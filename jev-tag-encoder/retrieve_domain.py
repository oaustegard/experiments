"""Round 3: SciFact retrieval with the corpus-fitted taxonomy (tags_scifact.txt) vs the general one.

Same queries, legs, scores and RRF as retrieve.py; the Jev vectors come from scifact_dom /
scifact_q_dom. Deltas are paired per query against the general-taxonomy run. Quantized codes use
the same cut schemes as quant.py, with bucket levels fit on the SciFact corpus vectors (no labels
involved; the general run fit them on the mixed corpus).
"""
import json

import numpy as np
from sklearn.cluster import KMeans

import jev
from common import DATA, boot, ci, fmt, jsonl, save
from embed import EMB
from quant import Quantizer
from retrieve import jev_scores, ndcgs, rrf, setup


def mean_ci(v, seed=7):
    return float(v.mean()), ci(boot(lambda i: v[i].mean(), len(v), seed=seed))[0]


def main():
    S = setup()
    doc_ids, qids, qrels, legs = S["doc_ids"], S["qids"], S["qrels"], S["legs"]
    nd, nq = len(doc_ids), len(qids)
    Dg, Qg = S["D"], S["Q"]["about"]
    Dd = jev.matrix("scifact_dom", "about", list(doc_ids))
    Qd = jev.matrix("scifact_q_dom", "about", qids)
    print(f"domain vectors: docs {(~np.isnan(Dd).any(1)).sum()}/{nd}, queries {(~np.isnan(Qd).any(1)).sum()}/{nq}")
    Dd, Qd = np.nan_to_num(Dd, nan=0.0), np.nan_to_num(Qd, nan=0.0)

    tags_d = jev.load_tags(jev.HERE / "tags_scifact.txt")
    diag = {}
    for name, D, Q, tags in (("general", Dg, Qg, jev.TAGS), ("domain", Dd, Qd, tags_d)):
        B = D >= 0.5
        fire = B.mean(0)
        codes = np.unique(np.packbits(B, axis=1), axis=0, return_counts=True)[1]
        # does a query share an active tag with at least one of its relevant docs?
        pos = {d: k for k, d in enumerate(doc_ids)}
        share = np.mean([any(((Q[i] >= 0.5) & B[pos[d]]).any() for d in qrels[q]) for i, q in enumerate(qids)])
        diag[name] = {"doc_active_mean": float(B.sum(1).mean()), "q_active_mean": float((Q >= 0.5).sum(1).mean()),
                      "tags_fire_ge_1pct": int((fire >= 0.01).sum()), "tags_never": int((fire == 0).sum()),
                      "unique_1bit_codes": len(codes), "largest_code_bucket": int(codes.max()),
                      "q_shares_tag_with_relevant_doc": float(share),
                      "top_fire": [(tags[i], round(float(fire[i]), 3)) for i in np.argsort(-fire)[:6]]}
        print(name, diag[name])

    # Does Jev put a doc under its own cluster's name? Refit build_domain_tags.py's KMeans (same seed) to get labels.
    E = np.load(EMB / "scifact_docs.npy")
    km = KMeans(256, n_init=4, random_state=98).fit(E)
    sizes = [c["size"] for c in json.loads((jev.HERE / "clusters_scifact.json").read_text())]
    assert np.bincount(km.labels_, minlength=256).tolist() == sizes, "KMeans refit does not reproduce the taxonomy"
    own = Dd[np.arange(nd), km.labels_]
    diag["domain"] |= {"own_cluster_p_ge_0.5": float((own >= 0.5).mean()),
                       "own_cluster_is_top1": float((Dd.argmax(1) == km.labels_).mean()),
                       "own_cluster_in_top5": float((np.argsort(-Dd, 1)[:, :5] == km.labels_[:, None]).any(1).mean()),
                       "zero_active_docs": float(((Dd >= 0.5).sum(1) == 0).mean())}
    print("domain agreement", {k: v for k, v in diag["domain"].items() if k.startswith(("own", "zero"))})

    # Ceiling for any labeller of this taxonomy: rank docs by the query embedding's cosine to the doc's cluster centroid
    C = km.cluster_centers_ / np.linalg.norm(km.cluster_centers_, axis=1, keepdims=True)
    qall = [r["_id"] for r in jsonl(DATA / "scifact" / "queries.jsonl")]
    Qe = np.load(EMB / "scifact_queries.npy")[[qall.index(q) for q in qids]]
    ceil_rank = list(np.argsort(-(Qe @ C.T)[:, km.labels_], axis=1, kind="stable"))

    base2 = ndcgs([rrf([legs["bm25"][i], legs["dense"][i]], nd) for i in range(nq)], qids, qrels, doc_ids)
    bm = ndcgs(legs["bm25"], qids, qrels, doc_ids)
    dense = ndcgs(legs["dense"], qids, qrels, doc_ids)
    runs = {}
    codings = {"general": (Dg, Qg, None), "domain": (Dd, Qd, None)}
    for b, s in [(1, "sign"), (2, "logit"), (3, "logit")]:
        q = Quantizer(b, s, Dd)
        codings[f"domain-{q.name}"] = (q(Dd), q(Qd), q)
    for cname, (D, Q, _) in codings.items():
        for sname, sc in jev_scores(Q, D, hamming=cname in ("general", "domain")).items():
            rank = list(np.argsort(-sc, axis=1, kind="stable"))
            runs[(cname, sname)] = {
                "alone": ndcgs(rank, qids, qrels, doc_ids),
                "rrf(bm25,dense,jev)": ndcgs([rrf([legs["bm25"][i], legs["dense"][i], rank[i]], nd) for i in range(nq)],
                                            qids, qrels, doc_ids),
                "rrf(bm25,jev)": ndcgs([rrf([legs["bm25"][i], rank[i]], nd) for i in range(nq)], qids, qrels, doc_ids),
                "rrf(dense,jev)": ndcgs([rrf([legs["dense"][i], rank[i]], nd) for i in range(nq)], qids, qrels, doc_ids),
            }
    refs = {"alone": None, "rrf(bm25,dense,jev)": base2, "rrf(bm25,jev)": bm, "rrf(dense,jev)": dense}
    out = {"diag": diag, "baselines": {"bm25": mean_ci(bm), "dense": mean_ci(dense), "rrf(bm25,dense)": mean_ci(base2)},
           "runs": {}}
    ceil = {"alone": ndcgs(ceil_rank, qids, qrels, doc_ids),
            "rrf(bm25,dense,jev)": ndcgs([rrf([legs["bm25"][i], legs["dense"][i], ceil_rank[i]], nd) for i in range(nq)],
                                        qids, qrels, doc_ids),
            "rrf(bm25,jev)": ndcgs([rrf([legs["bm25"][i], ceil_rank[i]], nd) for i in range(nq)], qids, qrels, doc_ids)}
    out["cluster_ceiling"] = {m: mean_ci(v if refs[m] is None else v - refs[m]) for m, v in ceil.items()}
    print("cluster-centroid ceiling:", {m: fmt(*v) for m, v in out["cluster_ceiling"].items()})
    print("\n| taxonomy/code | score | alone | RRF(bm25,dense,jev) − RRF(bm25,dense) | RRF(bm25,jev) − bm25 | "
          "RRF(dense,jev) − dense |")
    for (cname, sname), r in runs.items():
        row = {}
        for m, v in r.items():
            ref = refs[m]
            val = v if ref is None else v - ref
            row[m] = mean_ci(val)
            if cname != "general" and ("general", sname) in runs:
                dv = v - runs[("general", sname)][m]
                row[m + "|vs_general"] = mean_ci(dv)
        out["runs"][f"{cname}|{sname}"] = row
        print(f"| {cname} | {sname} | " + " | ".join(fmt(row[m][0], row[m][1]) for m in r) + " |")
    for sname in ("dot", "bernoulli", "hamming"):
        if ("domain", sname) in runs:
            r = out["runs"][f"domain|{sname}"]
            print(f"domain − general ({sname}): " + "  ".join(
                f"{m}: {fmt(*r[m + '|vs_general'])}" for m in runs[("domain", sname)]))
    save("round3_domain", out)


if __name__ == "__main__":
    main()
