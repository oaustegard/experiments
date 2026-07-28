#!/usr/bin/env python3
"""Portable projection sweep: can a seed-only (no shipped rotation) projection
match Haar at equal total bytes?

Background: a .kbi ships its Haar rotation matrices because numpy's
PCG64+Ziggurat+LAPACK-QR construction isn't reproducible in the JS reader.
Two independent implementations produce *different* orthogonal matrices, and
mixing them (docs hashed with A, query with B) collapses recall to chance
(~50% bit-flip) — see the inline demo in the session. The escape: a projection
both languages compute bit-identically from a seed.

  - Rademacher ±1 planes: integer entries from a counter PRNG → bit-identical
    cross-language, zero float drift, NOTHING shipped (seed only).
  - iid Gaussian planes (Box-Muller from a counter PRNG): also portable, shown
    here via numpy as a stand-in.
  - Haar: current remax; orthogonal, but needs the matrix shipped.

Question: how much recall does dropping the shipped Haar matrix cost, and at
what k does a portable projection match Haar's deployed 768/k=2 point — at what
TOTAL bytes (per-chunk vectors + corpus-independent rotation sidecar)?

Metric: recall@10 vs float-768 cosine top-10, self-retrieval (consistent with
the rest of kb-k-sweep). Random-plane methods averaged over 3 seeds (they carry
more variance than Haar); Haar also over 3 master seeds for a fair comparison.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
import sys
sys.path[:0] = [str(ROOT / ".spokes" / "remax_kb"), str(ROOT / ".spokes" / "remax" / "src")]
from remax import StackedSignBitQuantizer, hamming_distances
from remax.rotation import haar_rotation

DIM, TOPK = 768, 10
KS = [1, 2, 3, 4, 6, 8]
SEEDS = [0, 1, 2]
N_CHUNKS_DEFAULT = 1779


def load():
    d = np.load(HERE / "embeddings.npz", allow_pickle=True)
    V = d["vecs"].astype(np.float32)
    V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
    return V


def make_rotations(method, k, seed):
    """Return (k, DIM, DIM) f32 stack for the given projection family."""
    if method == "haar":
        ss = np.random.SeedSequence(seed)
        sub = ss.generate_state(k, dtype=np.uint32)
        return np.stack([haar_rotation(DIM, seed=int(s), dtype=np.float32) for s in sub])
    rng = np.random.default_rng(seed)
    if method == "gaussian":
        return rng.standard_normal((k, DIM, DIM)).astype(np.float32)
    if method == "rademacher":
        return np.where(rng.integers(0, 2, (k, DIM, DIM)) == 0, -1.0, 1.0).astype(np.float32)
    raise ValueError(method)


def encode(rots, X, k):
    q = StackedSignBitQuantizer(d=DIM, k=k, seed=0)
    q.rotations_ = rots.astype(q.dtype)
    return q.encode(X)


def recall(codes, gold, N):
    r = 0.0
    for i in range(N):
        dist = hamming_distances(codes, codes[i]); dist[i] = 1 << 30
        top = np.argpartition(dist, TOPK)[:TOPK]
        r += len(set(top) & gold[i]) / TOPK
    return r / N


def main():
    V = load(); N = V.shape[0]
    X = np.ascontiguousarray(V - V.mean(0))
    S = V @ V.T; np.fill_diagonal(S, -np.inf)
    gold = [set(np.argpartition(-S[i], TOPK)[:TOPK]) for i in range(N)]

    methods = ["haar", "gaussian", "rademacher"]
    results = {m: {} for m in methods}
    for m in methods:
        for k in KS:
            rs = [recall(encode(make_rotations(m, k, s), X, k), gold, N) for s in SEEDS]
            results[m][k] = (float(np.mean(rs)), float(np.std(rs)))
            print(f"{m:11s} k={k}  R@10={np.mean(rs):.4f} ±{np.std(rs):.4f}")
        print()

    # byte accounting (per the muninn corpus size)
    def vec_bytes(k): return DIM * k // 8                      # per chunk
    def haar_sidecar_i8(k): return k * DIM * DIM               # bytes, corpus-independent
    def total_kbi_vectors(method, k):
        v = N * vec_bytes(k)
        side = haar_sidecar_i8(k) if method == "haar" else 0   # rademacher: seed only
        return v + side

    haar_dep = results["haar"][2][0]   # Haar at deployed k=2
    print(f"=== Haar deployed point: 768/k=2  R@10={haar_dep:.4f} ===")
    print(f"    total vec+sidecar(int8) bytes = {total_kbi_vectors('haar',2)/1024:.0f} KB\n")

    # smallest portable k matching Haar@k2
    for m in ("rademacher", "gaussian"):
        match = next((k for k in KS if results[m][k][0] >= haar_dep), None)
        print(f"{m}: smallest k matching Haar@k2 ({haar_dep:.4f}) -> "
              + (f"k={match} (R={results[m][match][0]:.4f}), "
                 f"vectors {total_kbi_vectors(m,match)/1024:.0f} KB, NO sidecar"
                 if match else "none within k<=8"))

    out = {"corpus_n": N, "dim": DIM, "topk": TOPK, "seeds": SEEDS,
           "results": {m: {str(k): results[m][k] for k in KS} for m in methods},
           "byte_model": {
               "haar_768_k2_total_KB": total_kbi_vectors("haar", 2) / 1024,
               "rademacher_per_k_vectors_KB": {str(k): N * vec_bytes(k) / 1024 for k in KS}}}
    (HERE / "portable_projection.json").write_text(json.dumps(out, indent=2))
    print("\nwrote portable_projection.json")


if __name__ == "__main__":
    main()
