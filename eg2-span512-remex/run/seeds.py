"""Seed spread of the headline arms: the query bootstrap in eval.py holds remex's random rotation fixed.

Usage: python3 -I run/seeds.py

full768_b2, rot512_b2 and rot512_b3 at remex seeds 0..4 (seed 42 is in results/recall.json).
Writes results/seeds.json: per arm and query set, recall@10 per seed, mean and range.
"""
import json
import sys
from pathlib import Path

import numpy as np
import remex

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval as E  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
SEEDS = [0, 1, 2, 3, 4]


def arm(X, Qs, bits, seed, truth):
    qz = remex.Quantizer(X.shape[1], bits=bits, seed=seed, mean=X.mean(0).astype(np.float32))
    cv = qz.encode(X)
    dirs = qz.centroids.astype(np.float32)[cv.indices]
    scale = qz._score_norms(cv).astype(np.float32)
    bias = np.asarray(qz._row_bias(cv), np.float32)
    score = lambda Q: (qz._rotate_rows(np.asarray(Q, np.float32)).astype(np.float32) @ dirs.T) * scale + bias
    return {k: float(E.recall(E.topk(score, Q), truth[k]).mean()) for k, Q in Qs.items()}


def main() -> int:
    X = np.load(HERE / "out" / "docs.npy")
    X = X[np.linalg.norm(X, axis=1) > 0]
    Qs = {k: np.load(HERE / "out" / "qt" / f"{k}.eg2.npy") for k in E.SETS}
    B = np.load(HERE / "out" / "basis_proj.npy")
    truth = {k: E.topk(lambda q: q @ X.T, Q) for k, Q in Qs.items()}
    Z, Qz = X @ B.T, {k: Q @ B.T for k, Q in Qs.items()}
    res = {}
    for name, (Xs, Qss, b) in {"full768_b2": (X, Qs, 2), "rot512_b2": (Z, Qz, 2), "rot512_b3": (Z, Qz, 3)}.items():
        runs = [arm(Xs, Qss, b, s, truth) for s in SEEDS]
        res[name] = {k: {"per_seed": [r[k] for r in runs], "mean": float(np.mean([r[k] for r in runs])),
                         "range": float(np.ptp([r[k] for r in runs]))} for k in E.SETS}
        print(name, json.dumps(res[name]), flush=True)
    (HERE / "results" / "seeds.json").write_text(json.dumps({"seeds": SEEDS, "arms": res}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
