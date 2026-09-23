"""Round 4: interpretable retrieval on SciFact without dense vectors.

Two arms, both built from named quantities only:
  1. Score fusion of BM25 with the Round 3 tag vector: z(BM25) + λ·z(Bernoulli tag score), per query.
     λ is picked on a random half of the queries and scored on the other half.
  2. Jev as a pairwise reranker of a first stage's top K: state {"claim": query, "document": doc}, four
     Nouls (evidence, supports, refutes, same topic), one call per pair. The reranked top K keeps the
     first stage's order below it. First stages: BM25, and BM25 + λ·fitted tags at LAM_FIRST.

`python3 rerank.py encode` makes the calls (cache/rerank_pairs.jsonl, resumable);
`python3 rerank.py` scores everything into results/round4_rerank.json.
"""
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import bm25s
import numpy as np

import jev
from common import DATA, boot, ci, fmt, jsonl, save
from data import MAX_CHARS
from retrieve import jev_scores, ndcgs, setup

K = 20
NOULS = {
    "evidence": "The document provides evidence that supports or refutes the claim.",
    "supports": "The document supports the claim.",
    "refutes": "The document refutes the claim.",
    "topic": "The document is about the same topic as the claim.",
}
QS = {k: {"type": "noul", "instructions": v} for k, v in NOULS.items()}
PAIRS = jev.CACHE / "rerank_pairs.jsonl"
LAMBDAS = (0.1, 0.3, 0.5, 1.0, 2.0, 3.0)
LAM_FIRST = 1.0  # where the arm-1 curve flattens; read off all 300 queries, so the fused first stage is mildly tuned


def corpus_and_queries():
    sf = DATA / "scifact"
    corpus = jsonl(sf / "corpus.jsonl")
    queries = {r["_id"]: r["text"] for r in jsonl(sf / "queries.jsonl")}
    return corpus, queries


def bm25_scores(corpus, queries, qids) -> np.ndarray:
    """(n_queries, n_docs) BM25 scores, same tokenization as retrieve.setup()."""
    bm = bm25s.BM25()
    bm.index(bm25s.tokenize([f"{r['title']}. {r['text']}"[:MAX_CHARS] for r in corpus], stopwords="en",
                            show_progress=False), show_progress=False)
    idx, sc = bm.retrieve(bm25s.tokenize([queries[q] for q in qids], stopwords="en", show_progress=False),
                          k=len(corpus), show_progress=False)
    B = np.zeros((len(qids), len(corpus)))
    for i in range(len(qids)):
        B[i, idx[i]] = sc[i]
    return B


PAIRS_PQ = jev.VECTORS / "rerank_pairs.parquet"


def load_pairs() -> dict[str, dict]:
    """id "qid|docid" -> record. A fresh clone has no cache/ (gitignored); the committed parquet holds every call."""
    out = {}
    if PAIRS.exists():
        for line in PAIRS.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                out[r["id"]] = r
    elif PAIRS_PQ.exists():
        import pyarrow.parquet as pq
        for r in pq.read_table(PAIRS_PQ).to_pylist():
            out[r["id"]] = {"id": r["id"], "p": {k: r[k] for k in NOULS}, "in_tok": r["in_tok"],
                            "latency_s": r["latency_s"], "model": r["model"]}
    return out


def pairs_to_parquet() -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    recs = [r for r in load_pairs().values() if "p" in r]
    cols = {"id": [r["id"] for r in recs]} | {k: pa.array([r["p"][k] for r in recs], pa.float32()) for k in NOULS}
    cols |= {c: [r[c] for r in recs] for c in ("in_tok", "latency_s", "model")}
    pq.write_table(pa.table(cols), PAIRS_PQ, compression="zstd")


def zrows(M):
    return (M - M.mean(1, keepdims=True)) / (M.std(1, keepdims=True) + 1e-9)


def first_stages(S, corpus, queries):
    """name -> (B, T, list of full rankings per query)."""
    ids, qids = S["doc_ids"], S["qids"]
    B = bm25_scores(corpus, queries, qids)
    Dd = np.nan_to_num(jev.matrix("scifact_dom", "about", list(ids)), nan=0.0)
    Qd = np.nan_to_num(jev.matrix("scifact_q_dom", "about", qids), nan=0.0)
    T = jev_scores(Qd, Dd, hamming=False)["bernoulli"]
    fused = np.argsort(-(zrows(B) + LAM_FIRST * zrows(T)), 1, kind="stable")
    return B, T, {"bm25": S["legs"]["bm25"], "bm25+tags": list(fused)}


def encode(S, corpus, queries):
    text = {r["_id"]: f"{r['title']}. {r['text']}"[:MAX_CHARS] for r in corpus}
    ids = S["doc_ids"]
    done = load_pairs()
    _, _, stages = first_stages(S, corpus, queries)
    todo = sorted({(q, d) for rank in stages.values() for i, q in enumerate(S["qids"]) for d in ids[rank[i][:K]]
                   if "p" not in done.get(f"{q}|{d}", {})})
    print(f"{len(todo)} pairs to encode", flush=True)
    jev.CACHE.mkdir(exist_ok=True)
    lock, n, t0 = threading.Lock(), [0], time.time()

    def one(pair):
        q, d = pair
        try:
            res, dt, _ = jev.post({"claim": queries[q], "document": text[d]}, QS)
            u = res.get("usage", {})
            rec = {"id": f"{q}|{d}", "p": {k: float(res["answers"][k]["noul"]) for k in NOULS},
                   "in_tok": u.get("input_tokens"), "latency_s": dt, "model": res.get("model")}
        except (RuntimeError, jev.Blocked) as e:
            rec = {"id": f"{q}|{d}", "error": str(e)[:300]}
        with lock:
            with PAIRS.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            n[0] += 1
            if n[0] % 250 == 0 or n[0] == len(todo):
                print(f"[rerank] {n[0]}/{len(todo)} {time.time() - t0:.0f}s", flush=True)

    with ThreadPoolExecutor(8) as ex:  # pacing sets the rate
        list(ex.map(one, todo))
    pairs_to_parquet()
    print("ENCODE DONE", flush=True)


def paired(v, ref, seed=7):
    d = v - ref
    return float(v.mean()), float(d.mean()), ci(boot(lambda i: d[i].mean(), len(d), seed=seed))[0]


def main():
    S = setup()
    corpus, queries = corpus_and_queries()
    if sys.argv[1:] == ["encode"]:
        encode(S, corpus, queries)
        return
    ids, qids, qrels, legs = S["doc_ids"], S["qids"], S["qrels"], S["legs"]
    nq = len(qids)
    nd = ndcgs
    z = zrows
    B, _, stages = first_stages(S, corpus, queries)
    bm = nd(list(np.argsort(-B, 1, kind="stable")), qids, qrels, ids)
    bm_legacy = nd(legs["bm25"], qids, qrels, ids)
    assert abs(bm.mean() - bm_legacy.mean()) < 1e-3, (bm.mean(), bm_legacy.mean())
    out = {"k": K, "nouls": NOULS, "bm25": [float(bm.mean()), ci(boot(lambda i: bm[i].mean(), nq, seed=7))[0]]}

    # Arm 1: BM25 + λ·tags, λ tuned on half A, reported on half B and on all queries
    Dd = np.nan_to_num(jev.matrix("scifact_dom", "about", list(ids)), nan=0.0)
    Qd = np.nan_to_num(jev.matrix("scifact_q_dom", "about", qids), nan=0.0)
    tags = {"general": jev_scores(S["Q"]["about"], S["D"], hamming=False)["bernoulli"],
            "fitted": jev_scores(Qd, Dd, hamming=False)["bernoulli"]}
    half = np.random.default_rng(98).permutation(nq) < nq // 2
    out["fusion"] = {}
    for name, T in tags.items():
        runs = {lam: nd(list(np.argsort(-(z(B) + lam * z(T)), 1, kind="stable")), qids, qrels, ids) for lam in LAMBDAS}
        lam_a = max(runs, key=lambda lam: runs[lam][half].mean())
        v, d, c = paired(runs[lam_a][~half], bm[~half])
        out["fusion"][name] = {"by_lambda_all": {str(lam): paired(r, bm)[:2] for lam, r in runs.items()},
                               "lambda_tuned_on_A": lam_a, "heldout_B": {"ndcg": v, "delta": d, "ci": c}}
        print(f"BM25 + λ·{name}: " + "  ".join(f"{lam}: {r.mean():.3f}" for lam, r in runs.items())
              + f" | λ={lam_a} from half A, half B {fmt(d, c)}")

    # Arm 2: Jev pairwise rerank of BM25 top K
    pairs = load_pairs()
    ok = sum("p" in r for r in pairs.values())
    lat = np.array([r["latency_s"] for r in pairs.values() if "p" in r])
    toks = np.array([r["in_tok"] for r in pairs.values() if "p" in r])
    out["calls"] = {"ok": ok, "failed": len(pairs) - ok, "latency_p50": float(np.median(lat)),
                    "latency_p90": float(np.quantile(lat, 0.9)), "in_tok_mean": float(toks.mean())}
    print(out["calls"])

    def reranked(first, score_fn):
        ranks = []
        for i, q in enumerate(qids):
            top = first[i][:K]
            s = np.array([score_fn(pairs[f"{q}|{ids[j]}"]["p"], B[i, j], rank) if "p" in pairs.get(f"{q}|{ids[j]}", {})
                          else -1e9 for rank, j in enumerate(top)])
            ranks.append(np.concatenate([top[np.argsort(-s, kind="stable")], first[i][K:]]))
        return nd(ranks, qids, qrels, ids)

    rr = {
        "none (first stage)": None,
        "evidence": lambda p, b, r: p["evidence"],
        "supports": lambda p, b, r: p["supports"],
        "max(supports, refutes)": lambda p, b, r: max(p["supports"], p["refutes"]),
        "topic": lambda p, b, r: p["topic"],
        "evidence + topic": lambda p, b, r: p["evidence"] + p["topic"],
        "evidence + 0.1·first-stage rank prior": lambda p, b, r: p["evidence"] - 0.1 * r / K,
    }
    out["rerank"], out["oracle_topk"], out["recall_at_k"] = {}, {}, {}
    for sname, first in stages.items():
        out["rerank"][sname] = {}
        for name, fn in rr.items():
            r = nd(first, qids, qrels, ids) if fn is None else reranked(first, fn)
            v, d, c = paired(r, bm)
            out["rerank"][sname][name] = {"ndcg": v, "delta_vs_bm25": d, "ci": c,
                                          "ndcg_ci": ci(boot(lambda i, r=r: r[i].mean(), nq, seed=7))[0]}
            print(f"[{sname}] rerank top-{K} by {name}: {v:.3f}  vs BM25 {fmt(d, c)}")
        # oracle: relevant docs in the top K moved to the top (the ceiling for any top-K reranker)
        ranks = []
        for i, q in enumerate(qids):
            top = first[i][:K]
            rel = np.array([ids[j] in qrels[q] for j in top])
            ranks.append(np.concatenate([top[np.argsort(~rel, kind="stable")], first[i][K:]]))
        out["oracle_topk"][sname] = float(nd(ranks, qids, qrels, ids).mean())
        out["recall_at_k"][sname] = float(np.mean([len(set(ids[first[i][:K]]) & qrels[q]) / len(qrels[q])
                                                   for i, q in enumerate(qids)]))
        print(f"[{sname}] oracle rerank of top-{K}: {out['oracle_topk'][sname]:.3f}; "
              f"recall@{K} {out['recall_at_k'][sname]:.3f}")
    ev = {sname: reranked(first, rr["evidence"]) for sname, first in stages.items()}
    out["fused_vs_bm25_first_stage_evidence"] = paired(ev["bm25+tags"], ev["bm25"])
    print("evidence rerank, fused first stage minus BM25 first stage:", fmt(*out["fused_vs_bm25_first_stage_evidence"][1:]))

    # Noul calibration on the judged pairs: relevant vs not, among the top-K candidates
    rel_p, non_p = [], []
    for i, q in enumerate(qids):
        for j in legs["bm25"][i][:K]:  # calibration on the BM25 candidate set only
            r = pairs.get(f"{q}|{ids[j]}")
            if r and "p" in r:
                (rel_p if ids[j] in qrels[q] else non_p).append(r["p"])
    from sklearn.metrics import roc_auc_score
    out["pair_auc"] = {}
    for k in NOULS:
        y = [1] * len(rel_p) + [0] * len(non_p)
        s = [p[k] for p in rel_p] + [p[k] for p in non_p]
        out["pair_auc"][k] = {"auc": float(roc_auc_score(y, s)), "mean_rel": float(np.mean([p[k] for p in rel_p])),
                              "mean_nonrel": float(np.mean([p[k] for p in non_p])),
                              "frac_rel_ge_0.5": float(np.mean([p[k] >= 0.5 for p in rel_p])),
                              "frac_nonrel_ge_0.5": float(np.mean([p[k] >= 0.5 for p in non_p]))}
    out["pair_counts"] = {"relevant": len(rel_p), "nonrelevant": len(non_p)}
    print(out["pair_counts"], {k: round(v["auc"], 3) for k, v in out["pair_auc"].items()})
    save("round4_rerank", out)


if __name__ == "__main__":
    main()
