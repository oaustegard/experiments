#!/usr/bin/env python3
"""SRHT near-orthogonal projection vs Haar vs Rademacher.

Rademacher planes are only *approximately* orthogonal (independent ±1 columns),
which cost ~2 recall@10 pts vs Haar (Part 7). A structured alternative:

    one round = D then H   where D is a seed-driven random ±1 diagonal and H is
    the Walsh-Hadamard transform. H/sqrt(n) is exactly orthogonal and D is
    orthogonal, so H·D is an EXACT orthogonal transform — but structured (only
    `dim` random bits). Stacking R rounds (H D H D ...) increases the effective
    randomness toward Haar-uniform while staying exactly orthogonal, O(d log d),
    integer sign-flips + float add/sub only (no QR, no transcendentals → fully
    portable & bit-deterministic across Python/JS).

Question: do 2-3 rounds recover Haar's ~2 pts that plain Rademacher gives up?

dim=768 isn't a power of two, so we zero-pad to 1024, transform, and take the
sign of the first 768 outputs per stack (standard subsampled-Hadamard). Metric:
recall@10 vs float-768 gold, self-retrieval, 3 seeds (matches portable_projection.py).
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / ".spokes" / "remax_kb"), str(ROOT / ".spokes" / "remax" / "src")]
from remax import hamming_distances
from remax.rotation import haar_rotation

DIM, PAD, TOPK = 768, 1024, 10
KS = [1, 2, 3, 4]
SEEDS = [0, 1, 2]


def fwht(a):
    """In-place iterative Walsh-Hadamard along axis 1. a: (n, PAD), PAD = 2^m."""
    a = a.copy()
    n = a.shape[1]
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            x = a[:, i:i + h].copy()
            y = a[:, i + h:i + h * 2].copy()
            a[:, i:i + h] = x + y
            a[:, i + h:i + h * 2] = x - y
        h *= 2
    return a


def srht_bits(X, k, seed, rounds):
    """(n, DIM) -> packed sign-bit codes (n, k*DIM//8) via R rounds of H·D per stack."""
    n = X.shape[0]
    Xp = np.zeros((n, PAD), dtype=np.float32)
    Xp[:, :DIM] = X
    stacks = []
    for j in range(k):
        rng = np.random.default_rng((seed << 16) ^ (j + 1))
        Y = Xp
        for _ in range(rounds):
            d = (rng.integers(0, 2, PAD) * 2 - 1).astype(np.float32)
            Y = fwht(Y * d)
        stacks.append((Y[:, :DIM] >= 0))          # (n, DIM) bool
    bits = np.concatenate(stacks, axis=1)          # (n, k*DIM)
    return np.packbits(bits.astype(np.uint8), axis=1)


def haar_bits(X, k, seed):
    rots = np.stack([haar_rotation(DIM, seed=int(s), dtype=np.float32)
                     for s in np.random.SeedSequence(seed).generate_state(k, dtype=np.uint32)])
    bits = np.concatenate([(X @ rots[j]) >= 0 for j in range(k)], axis=1)
    return np.packbits(bits.astype(np.uint8), axis=1)


def rademacher_bits(X, k, seed):
    rng = np.random.default_rng(seed)
    R = np.where(rng.integers(0, 2, (k, DIM, DIM)) == 0, -1.0, 1.0).astype(np.float32)
    bits = np.concatenate([(X @ R[j]) >= 0 for j in range(k)], axis=1)
    return np.packbits(bits.astype(np.uint8), axis=1)


def main():
    d = np.load(HERE / "embeddings.npz", allow_pickle=True)
    V = d["vecs"].astype(np.float32)
    V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
    N = V.shape[0]
    X = np.ascontiguousarray(V - V.mean(0))
    S = V @ V.T; np.fill_diagonal(S, -np.inf)
    gold = [set(np.argpartition(-S[i], TOPK)[:TOPK]) for i in range(N)]

    def recall(codes):
        r = 0.0
        for i in range(N):
            dist = hamming_distances(codes, codes[i]); dist[i] = 1 << 30
            r += len(set(np.argpartition(dist, TOPK)[:TOPK]) & gold[i]) / TOPK
        return r / N

    methods = {
        "haar":       lambda k, s: haar_bits(X, k, s),
        "rademacher": lambda k, s: rademacher_bits(X, k, s),
        "srht_r2":    lambda k, s: srht_bits(X, k, s, 2),
        "srht_r3":    lambda k, s: srht_bits(X, k, s, 3),
    }
    results = {m: {} for m in methods}
    for m, fn in methods.items():
        for k in KS:
            rs = [recall(fn(k, s)) for s in SEEDS]
            results[m][k] = (float(np.mean(rs)), float(np.std(rs)))
            print(f"{m:11s} k={k}  R@10={np.mean(rs):.4f} ±{np.std(rs):.4f}")
        print()

    (HERE / "srht_projection.json").write_text(json.dumps(
        {"dim": DIM, "pad": PAD, "n": N, "seeds": SEEDS,
         "results": {m: {str(k): results[m][k] for k in KS} for m in methods}}, indent=2))
    print("=== gap recovered vs Haar (k=2) ===")
    h2 = results["haar"][2][0]
    for m in ("rademacher", "srht_r2", "srht_r3"):
        print(f"  {m:11s} k=2: {results[m][2][0]:.4f}  ({results[m][2][0]-h2:+.4f} vs Haar)")
    print("wrote srht_projection.json")


if __name__ == "__main__":
    main()
