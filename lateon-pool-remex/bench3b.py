#!/usr/bin/env python3
"""Round 3b: the quantizer-matched cut and its uniform control, scored by RemexIndex search.

Same cuts as round 1/2 (bench.WardForest on true Ward; thresholds c * D_b from
.work/<corpus>/distortion.json), each stored in RemexIndex(similarity="maxsim",
center=True) at the matching bit width, seeds 42, 1, 2, 3. Rows append to
results3_<corpus>.jsonl next to round 3's. Prints 'BENCH3B DONE <corpus>'.
"""
from __future__ import annotations

import json, sys, time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bench  # noqa: E402
from bench3 import ndcg10, SEEDS  # noqa: E402

POINTS = [(1.0, 2), (4.0, 3), (2.0, 3), (1.0, 3)]


def run(corpus, remex_path):
    sys.path.insert(0, remex_path)
    import remex
    from remex import RemexIndex
    W = bench.WORK / corpus
    res = HERE / f"results3_{corpus}.jsonl"
    done = {(json.loads(l)["arm"], json.loads(l)["seed"]) for l in open(res)} if res.exists() else set()
    T0 = np.load(W / "tokens.npy"); off0 = np.load(W / "offsets.npy")
    QT = np.load(W / "q_tokens.npy"); qoff = np.load(W / "q_offsets.npy")
    dids = json.load(open(W / "doc_ids.json")); qids = json.load(open(W / "query_ids.json"))
    qrels = json.load(open(W / "qrels.json"))
    Db = {int(k): v for k, v in json.load(open(W / "distortion.json")).items()}
    qm = [QT[qoff[i]:qoff[i + 1]] for i in range(len(qoff) - 1)]
    forest = bench.WardForest(T0, off0)
    for c, b in POINTS:
        T, off = forest.cut(tau=c * Db[b])
        Tm, offm = forest.cut(f=forest.factor_for(len(T)))
        for fam, (TT, oo) in (("adapt", (T, off)), ("match", (Tm, offm))):
            arm = f"{fam} c{c:g}@{b} {b}-bit"
            mats = [TT[oo[i]:oo[i + 1]] for i in range(len(oo) - 1)]
            for seed in SEEDS:
                if (arm, seed) in done:
                    continue
                t0 = time.time()
                ix = RemexIndex(None, bits=b, similarity="maxsim", center=True, dim=128, seed=seed)
                ix.add(mats)
                nd = ndcg10(ix.search(qm, top_k=10), qids, dids, qrels)
                row = dict(arm=arm, seed=seed, bits=b, c=c, tokens_per_doc=round(ix.compressed.n / len(ix), 2),
                           bytes_per_doc=round(ix.nbytes / len(ix), 1), ndcg10=round(float(np.mean(nd)), 4),
                           perquery=[round(x, 5) for x in nd], remex=remex.__version__,
                           seconds=round(time.time() - t0, 1))
                with open(res, "a") as fh:
                    fh.write(json.dumps(row) + "\n")
                print(f"  {arm:20s} seed {seed:2d}  B/doc {row['bytes_per_doc']:8.0f}  nDCG@10 {row['ndcg10']:.4f}", flush=True)
    print(f"BENCH3B DONE {corpus}", flush=True)


if __name__ == "__main__":
    remex_path = sys.argv[1]
    for c in [a for a in sys.argv[2:] if a in ("scifact", "nfcorpus")] or ("scifact", "nfcorpus"):
        run(c, remex_path)
