"""Step 4: BEIR SciFact (5,183 docs, 300 test queries). Jev vectors as a retrieval leg.

Doc side: "The document is about <tag>." Query side: the same questions over the query text
("about") and "The query asks about <tag>." ("query"). Scores between query probs q and doc probs d:
  dot       q . d
  bernoulli sum_t q_t log d_t + (1 - q_t) log(1 - d_t), d clipped to [0.01, 0.99]
  hamming   -popcount(bits(q) xor bits(d)), bits at 0.5 (ties broken by corpus order)
Baselines: BM25 (bm25s, English stopwords) and gemini-embedding-2 cosine. RRF k=60 over full rankings.
"""
import numpy as np

import jev
from common import DATA, boot, ci, fmt, jsonl, save
from data import MAX_CHARS
from embed import EMB

K = 10
RRF_K = 60


def ndcg_at_k(rank: np.ndarray, rel: set, all_ids: np.ndarray) -> float:
    top = all_ids[rank[:K]]
    dcg = sum(1 / np.log2(i + 2) for i, d in enumerate(top) if d in rel)
    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(rel), K)))
    return dcg / idcg


def rrf(ranks: list[np.ndarray], n_docs: int) -> np.ndarray:
    """ranks: per leg, doc indices in rank order. Returns fused ranking."""
    s = np.zeros(n_docs)
    for r in ranks:
        pos = np.empty(n_docs)
        pos[r] = np.arange(n_docs)
        s += 1 / (RRF_K + pos + 1)
    return np.argsort(-s, kind="stable")


def setup() -> dict:
    """Corpus, qrels, doc-side Jev matrix, and the BM25 + dense rankings (the Jev-independent legs)."""
    sf = DATA / "scifact"
    corpus = jsonl(sf / "corpus.jsonl")
    doc_ids = np.array([r["_id"] for r in corpus])
    qrels = {}
    for line in open(sf / "test.tsv").read().splitlines()[1:]:
        q, d, s = line.split("\t")
        if int(s) > 0:
            qrels.setdefault(q, set()).add(d)
    qids = sorted(qrels, key=int)
    queries = {r["_id"]: r["text"] for r in jsonl(sf / "queries.jsonl")}

    D = jev.matrix("scifact", "about", list(doc_ids))
    doc_ok = ~np.isnan(D).any(1)
    print(f"docs with Jev vectors: {doc_ok.sum()}/{len(doc_ok)}")
    D = np.nan_to_num(D, nan=0.0)  # a failed doc gets the all-zero vector: ranked last by every Jev score

    import bm25s
    tok = bm25s.tokenize([f"{r['title']}. {r['text']}"[:MAX_CHARS] for r in corpus], stopwords="en")
    bm = bm25s.BM25()
    bm.index(tok)
    qtok = bm25s.tokenize([queries[q] for q in qids], stopwords="en")
    bm_idx, _ = bm.retrieve(qtok, k=len(corpus))

    E = np.load(EMB / "scifact_docs.npy")
    qall = [r["_id"] for r in jsonl(sf / "queries.jsonl")]
    Qe = np.load(EMB / "scifact_queries.npy")[[qall.index(q) for q in qids]]
    dense_rank = np.argsort(-(Qe @ E.T), axis=1, kind="stable")
    Q = {qv: np.nan_to_num(jev.matrix("scifact_q", qv, qids), nan=0.0) for qv in ("about", "query")}
    return {"doc_ids": doc_ids, "qids": qids, "qrels": qrels, "D": D, "doc_ok": doc_ok, "Q": Q,
            "legs": {"bm25": [bm_idx[i] for i in range(len(qids))], "dense": list(dense_rank)}}


def jev_scores(Q: np.ndarray, D: np.ndarray, hamming: bool = True) -> dict[str, np.ndarray]:
    Dc = np.clip(D, 0.01, 0.99)
    out = {"dot": Q @ D.T, "bernoulli": Q @ np.log(Dc).T + (1 - Q) @ np.log(1 - Dc).T}
    if hamming:
        Db = D >= 0.5
        out["hamming"] = -np.stack([((q >= 0.5)[None, :] != Db).sum(-1) for q in Q])
    return out


def ndcgs(rankings, qids, qrels, doc_ids) -> np.ndarray:
    return np.array([ndcg_at_k(rankings[i], qrels[q], doc_ids) for i, q in enumerate(qids)])


def main():
    S = setup()
    doc_ids, qids, qrels, D, doc_ok = S["doc_ids"], S["qids"], S["qrels"], S["D"], S["doc_ok"]
    legs = dict(S["legs"])
    for qv, Q in S["Q"].items():
        for sname, sc in jev_scores(Q, D).items():
            legs[f"jev_{qv}_{sname}"] = list(np.argsort(-sc, axis=1, kind="stable"))

    n_docs = len(doc_ids)
    per_q = {name: np.array([ndcg_at_k(r[i], qrels[q], doc_ids) for i, q in enumerate(qids)])
             for name, r in legs.items()}
    base = [rrf([legs["bm25"][i], legs["dense"][i]], n_docs) for i in range(len(qids))]
    per_q["rrf(bm25,dense)"] = np.array([ndcg_at_k(base[i], qrels[q], doc_ids) for i, q in enumerate(qids)])
    for name in [n for n in legs if n.startswith("jev_")]:
        fused = [rrf([legs["bm25"][i], legs["dense"][i], legs[name][i]], n_docs) for i in range(len(qids))]
        per_q[f"rrf(bm25,dense,{name})"] = np.array([ndcg_at_k(fused[i], qrels[q], doc_ids)
                                                      for i, q in enumerate(qids)])
        f2 = [rrf([legs["bm25"][i], legs[name][i]], n_docs) for i in range(len(qids))]
        per_q[f"rrf(bm25,{name})"] = np.array([ndcg_at_k(f2[i], qrels[q], doc_ids) for i, q in enumerate(qids)])
        f3 = [rrf([legs["dense"][i], legs[name][i]], n_docs) for i in range(len(qids))]
        per_q[f"rrf(dense,{name})"] = np.array([ndcg_at_k(f3[i], qrels[q], doc_ids) for i, q in enumerate(qids)])

    nq = len(qids)
    out = {}
    print("| run | nDCG@10 [95% CI] | vs rrf(bm25,dense) |\n|---|---|---|")
    ref = per_q["rrf(bm25,dense)"]
    for name, v in per_q.items():
        c = ci(boot(lambda i: v[i].mean(), nq, seed=2))[0]
        d = v - ref
        dc = ci(boot(lambda i: d[i].mean(), nq, seed=2))[0]
        out[name] = {"ndcg10": float(v.mean()), "ci": c, "delta_vs_rrf2": float(d.mean()), "delta_ci": dc}
        print(f"| {name} | {fmt(v.mean(), c)} | {d.mean():+.3f} [{dc[0]:+.3f}, {dc[1]:+.3f}] |")
    # two-leg fusions against their single leg: does the tag leg help BM25 alone, or dense alone?
    for single in ("bm25", "dense"):
        for name in [n for n in per_q if n.startswith(f"rrf({single},jev")]:
            d = per_q[name] - per_q[single]
            dc = ci(boot(lambda i: d[i].mean(), nq, seed=2))[0]
            out[name][f"delta_vs_{single}"] = float(d.mean())
            out[name][f"delta_vs_{single}_ci"] = dc
            print(f"  {name} - {single}: {d.mean():+.3f} [{dc[0]:+.3f}, {dc[1]:+.3f}]")
    act_d = (D[doc_ok] >= 0.5).sum(1)
    Qa = jev.matrix("scifact_q", "about", qids)
    Qq = jev.matrix("scifact_q", "query", qids)
    stats = {"doc_active_mean": float(act_d.mean()), "doc_zero_active": int((act_d == 0).sum()),
             "q_about_active_mean": float(np.nanmean((Qa >= 0.5).sum(1))),
             "q_query_active_mean": float(np.nanmean((Qq >= 0.5).sum(1))),
             "q_about_zero_active": int(((Qa >= 0.5).sum(1) == 0).sum()),
             "q_query_zero_active": int(((Qq >= 0.5).sum(1) == 0).sum())}
    print(stats)
    save("step4_retrieve", {"n_queries": nq, "n_docs": n_docs, "docs_encoded": int(doc_ok.sum()),
                            "runs": out, "activity": stats})


if __name__ == "__main__":
    main()
