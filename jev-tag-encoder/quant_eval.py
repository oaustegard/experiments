"""Steps 3 and 4 re-run on 1/2/3-bit Jev codes against the 2-decimal float vectors.

Step 3: zero-shot label scores (F1 at 0.5, oracle-global F1, micro/macro-AP) and the jev256+probe@500
arm. Step 4: Jev standalone nDCG@10 (dot, bernoulli; about and query phrasings) and the RRF legs.
Deltas are paired against the float vectors on the same bootstrap resamples.
"""
import numpy as np

import jev
from classify import fit_predict, pick_c
from common import DATA, ap_scores, arxiv, boot, ci, f1s, fmt, jsonl, label_scores, save
from quant import CUTS, Quantizer
from retrieve import jev_scores, ndcgs, rrf, setup


def main():
    fit_ids = [r["id"] for r in jsonl(DATA / "mixed.jsonl")]
    Pfit = jev.matrix("mixed", "about", fit_ids)
    Pfit = Pfit[~np.isnan(Pfit).any(1)]
    quants = [None] + [Quantizer(b, s, Pfit) for (b, s) in CUTS]
    names = ["float"] + [q.name for q in quants[1:]]
    for q in quants[1:]:
        print(f"{q.name}: cuts {q.cuts} levels {q.levels.round(3).tolist()}")
    out = {"levels": {q.name: q.levels.tolist() for q in quants[1:]}}

    # ---- step 3
    te_ids, Yte, _ = arxiv("test")
    tr_ids, Ytr, _ = arxiv("train")
    Pte = jev.matrix("arxiv", "about", te_ids)
    Ptr = jev.matrix("arxiv", "about", tr_ids)
    ok = ~np.isnan(Pte).any(1)
    Yte, Pte = Yte[ok], Pte[ok]
    have_tr = np.where(~np.isnan(Ptr).any(1))[0]
    out["occupancy_arxiv_test"] = {q.name: q.occupancy(Pte) for q in quants[1:]}
    n = len(Yte)
    grid = np.round(np.arange(0.05, 0.96, 0.05), 2)
    rows, s3 = [], {}
    for q, name in zip(quants, names):
        P, Pt = (Pte, Ptr) if q is None else (q(Pte), q(Ptr))
        S = label_scores(P)
        g = max(grid, key=lambda t: f1s(Yte, S >= t)[0])
        idx = np.random.default_rng(1000 * 500).choice(have_tr, 500, replace=False)
        C = pick_c(Pt[idx], Ytr[idx], 0)
        Sp = fit_predict(Pt[idx], Ytr[idx], P, C)

        def stat(i, S=S, g=g, Sp=Sp):
            return (*f1s(Yte[i], S[i] >= 0.5), f1s(Yte[i], S[i] >= g)[0], *ap_scores(Yte[i], S[i]),
                    f1s(Yte[i], Sp[i] >= 0.5)[0], ap_scores(Yte[i], Sp[i])[0])
        pt = stat(np.arange(n))
        bs = boot(stat, n, seed=4)
        s3[name] = {"point": list(pt), "ci": ci(bs), "_boot": bs, "oracle_thr": float(g)}
    cols = ["zs micro-F1@.5", "zs macro-F1@.5", "zs micro-F1@oracle", "zs micro-AP", "zs macro-AP",
            "probe500 micro-F1", "probe500 micro-AP"]
    print("\nstep 3 | " + " | ".join(cols))
    for name in names:
        r = s3[name]
        d = r["_boot"] - s3["float"]["_boot"]
        dci = ci(d)
        r["delta_vs_float"] = [[float(r["point"][k] - s3["float"]["point"][k])] + dci[k] for k in range(len(cols))]
        rows.append(f"| {name} | " + " | ".join(
            f"{r['point'][k]:.3f} ({r['delta_vs_float'][k][0]:+.3f} [{dci[k][0]:+.3f}, {dci[k][1]:+.3f}])"
            for k in range(len(cols))) + " |")
    print("\n".join(rows))
    for r in s3.values():
        r.pop("_boot")
    out["step3"] = {"cols": cols, "n_test": n, "arms": s3}

    # ---- step 4
    S = setup()
    doc_ids, qids, qrels, D = S["doc_ids"], S["qids"], S["qrels"], S["D"]
    legs = S["legs"]
    nd = len(doc_ids)
    nq = len(qids)
    out["occupancy_scifact_docs"] = {q.name: q.occupancy(D) for q in quants[1:]}
    base = ndcgs([rrf([legs["bm25"][i], legs["dense"][i]], nd) for i in range(nq)], qids, qrels, doc_ids)
    dense = ndcgs(legs["dense"], qids, qrels, doc_ids)
    s4 = {}
    for q, name in zip(quants, names):
        Dq = D if q is None else q(D)
        for qv, Q in S["Q"].items():
            Qq = Q if q is None else q(Q)
            for sname, sc in jev_scores(Qq, Dq, hamming=False).items():
                rank = list(np.argsort(-sc, axis=1, kind="stable"))
                alone = ndcgs(rank, qids, qrels, doc_ids)
                three = ndcgs([rrf([legs["bm25"][i], legs["dense"][i], rank[i]], nd) for i in range(nq)],
                              qids, qrels, doc_ids)
                two = ndcgs([rrf([legs["dense"][i], rank[i]], nd) for i in range(nq)], qids, qrels, doc_ids)
                s4[f"{name}|{qv}_{sname}"] = {"alone": alone, "rrf3_minus_rrf2": three - base,
                                              "rrf_dense_minus_dense": two - dense}
    print("\nstep 4 | code | leg | standalone nDCG@10 | RRF(bm25,dense,jev) - RRF(bm25,dense) | "
          "RRF(dense,jev) - dense")
    res4 = {}
    for key, v in s4.items():
        name, leg = key.split("|")
        fl = s4[f"float|{leg}"]
        row = {}
        for m, arr in v.items():
            c = ci(boot(lambda i, a=arr: a[i].mean(), nq, seed=5))[0]
            dv = arr - fl[m]
            dc = ci(boot(lambda i, a=dv: a[i].mean(), nq, seed=5))[0]
            row[m] = {"mean": float(arr.mean()), "ci": c, "delta_vs_float": float(dv.mean()), "delta_ci": dc}
        res4[key] = row
        print(f"| {name} | {leg} | " + " | ".join(
            f"{fmt(row[m]['mean'], row[m]['ci'])} (vs float {row[m]['delta_vs_float']:+.3f})" for m in row) + " |")
    out["step4"] = res4
    save("quant_eval", out)


if __name__ == "__main__":
    main()
