"""Shared loaders and metrics for fold.py, eval.py and seeds.py."""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
OUT = HERE / "out"
RES = HERE / "results"
K = 10
SETS = ("synth", "real")
L = 63  # int8 code range of the shipped package (w8L63)


def unit(a):
    return a / np.linalg.norm(a, axis=1, keepdims=True)


def load():
    """X: unit 768-d model outputs, P: pooled 256-d states, per set; W: head weight [256, 768]."""
    X = {k: np.load(OUT / "vec" / f"{k}.x.npy") for k in ("docs",) + SETS}
    P = {k: np.load(OUT / "vec" / f"{k}.p.npy") for k in ("docs",) + SETS}
    src = np.array([json.loads(l)["post"] for l in (OUT / "q" / "synth_test.jsonl").open()])
    return X, P, np.load(OUT / "W.npy"), src


def svd_head(W):
    """W = A diag(S) B^T. Returns the folded head H = A diag(S) [256, 256] and the basis B [768, 256].

    The model computes e = p W, so e B = p H holds the same vector in the 256 coordinates of
    the head's row space, and B has orthonormal columns: lengths and angles are unchanged.
    """
    A, S, Bt = np.linalg.svd(W.astype(np.float64), full_matrices=False)
    return A * S, Bt.T, S


def int8_cols(H, levels=L):
    """The shipped weight format: int8 codes in +-levels with one fp32 scale per output column."""
    s = (np.abs(H).max(0) / levels + 1e-12).astype(np.float32)
    q = np.clip(np.round(H / s), -levels, levels).astype(np.int8)
    return q, s


def topk(score_fn, Q, k=K, chunk=250):
    out = np.empty((len(Q), k), np.int64)
    for s in range(0, len(Q), chunk):
        sc = score_fn(Q[s:s + chunk])
        part = np.argpartition(-sc, k, axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(sc, part, 1), axis=1)
        out[s:s + chunk] = np.take_along_axis(part, order, 1)
    return out


def recall(a, b):
    """Per-query share of b's top 10 that a returns."""
    return np.array([len(set(x) & set(y)) / K for x, y in zip(a, b)])


def hits(top, src):
    return (top == src[:, None]).any(1).astype(float)


def boot(x, n=2000, seed=0):
    """Mean and bootstrap 95% interval over queries."""
    rng = np.random.default_rng(seed)
    m = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)])
    return {"mean": float(x.mean()), "lo": float(np.quantile(m, 0.025)), "hi": float(np.quantile(m, 0.975))}
