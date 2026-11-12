#!/usr/bin/env python3
"""Round 3: the round-1/2 comparisons scored through remex's own index search.

Rounds 1-2 scored MaxSim on decoded vectors (q . decode(x)). RemexIndex ranks
with remex's centered search estimator instead (scale ||r|| / kappa and the
stored mu.r; remex CLAUDE.md design decision 4), which is what a user of the
library gets. This re-runs the arms that carry the findings with
RemexIndex(similarity="maxsim", center=True), sentence-transformers 6.3
HierarchicalTokenPooling for the pooled arms, and seeds 42, 1, 2, 3.
Bytes per document are remex's own nbytes (codes + float32 ||r|| + float32 mu.r).

One JSON line per (arm, seed) in results3_<corpus>.jsonl; resumable.
Prints 'BENCH3 DONE <corpus>'.
"""
from __future__ import annotations

import json, sys, time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
WORK = HERE / ".work"
SEEDS = (42, 1, 2, 3)
ARMS = [("none", None, 1), ("none", None, 2), ("none", None, 4), ("none", None, 8),
        ("st pf2", 2, 2), ("st pf2", 2, 4), ("st pf4", 4, 2), ("st pf4", 4, 4), ("st pf4", 4, 8)]


def ndcg10(res, qids, dids, qrels):
    out = []
    for r, q in zip(res, qids):
        rel = qrels[q]
        dcg = sum(rel.get(dids[h["corpus_id"]], 0) / np.log2(i + 2) for i, h in enumerate(r[:10]))
        ideal = sorted(rel.values(), reverse=True)[:10]
        out.append(dcg / sum(g / np.log2(i + 2) for i, g in enumerate(ideal)))
    return out


def run(corpus, remex_path):
    sys.path.insert(0, remex_path)
    import remex
    from remex import RemexIndex
    from sentence_transformers.multi_vector_encoder.modules import HierarchicalTokenPooling
    W = WORK / corpus
    res = HERE / f"results3_{corpus}.jsonl"
    done = {(json.loads(l)["arm"], json.loads(l)["seed"]) for l in open(res)} if res.exists() else set()
    T = np.load(W / "tokens.npy"); off = np.load(W / "offsets.npy")
    QT = np.load(W / "q_tokens.npy"); qoff = np.load(W / "q_offsets.npy")
    dids = json.load(open(W / "doc_ids.json")); qids = json.load(open(W / "query_ids.json"))
    qrels = json.load(open(W / "qrels.json"))
    mats = [T[off[i]:off[i + 1]] for i in range(len(off) - 1)]
    qm = [QT[qoff[i]:qoff[i + 1]] for i in range(len(qoff) - 1)]
    pooled = {None: mats}
    for f in sorted({a[1] for a in ARMS if a[1]}):
        p = HierarchicalTokenPooling(pool_factor=f)
        pooled[f] = [t.numpy() for s in range(0, len(mats), 256)
                     for t in p.pool([torch.from_numpy(x) for x in mats[s:s + 256]])]
    print(f"[{corpus}] remex {remex.__version__}, {len(mats)} docs, {len(qm)} queries", flush=True)
    for name, f, bits in ARMS:
        arm = f"{name} {bits}-bit"
        for seed in SEEDS:
            if (arm, seed) in done:
                continue
            t0 = time.time()
            ix = RemexIndex(None, bits=bits, similarity="maxsim", center=True, dim=128, seed=seed)
            ix.add(pooled[f])
            nd = ndcg10(ix.search(qm, top_k=10), qids, dids, qrels)
            row = dict(arm=arm, seed=seed, bits=bits, pool_factor=f,
                       tokens_per_doc=round(ix.compressed.n / len(ix), 2),
                       bytes_per_doc=round(ix.nbytes / len(ix), 1), ndcg10=round(float(np.mean(nd)), 4),
                       perquery=[round(x, 5) for x in nd], remex=remex.__version__,
                       seconds=round(time.time() - t0, 1))
            with open(res, "a") as fh:
                fh.write(json.dumps(row) + "\n")
            print(f"  {arm:14s} seed {seed:2d}  B/doc {row['bytes_per_doc']:8.0f}  nDCG@10 {row['ndcg10']:.4f}"
                  f"  ({row['seconds']}s)", flush=True)
    print(f"BENCH3 DONE {corpus}", flush=True)


if __name__ == "__main__":
    torch.set_num_threads(4)
    remex_path = sys.argv[1]
    for c in [a for a in sys.argv[2:] if a in ("scifact", "nfcorpus")] or ("scifact", "nfcorpus"):
        run(c, remex_path)
