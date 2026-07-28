#!/usr/bin/env python3
"""(dim, k) grid sweep on the Mac-search corpus — where do the bits go?

Follow-up to sweep.py. That run showed Matryoshka truncation 768->256 was the
dominant recall loss, not bit-depth. This sweeps both axes against a FIXED
reference (float-768 top-10) so dims and stacks are on one yardstick, and asks
the design question: at a matched byte budget (bytes/chunk = dim*k/8), is recall
better spent on more dimensions or more stacks?

Reuses the cached embeddings.npz from sweep.py — no re-embedding.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / ".spokes" / "remax_kb"), str(ROOT / ".spokes" / "remax" / "src")]

DIMS = [64, 128, 256, 384, 512, 768]   # all divisible by 8 (remax requirement)
KS = [1, 2, 4, 8, 16]
SEED, TOPK = 0, 10
EMB = HERE / "embeddings.npz"


def topk_float(X, k):
    sims = X @ X.T
    np.fill_diagonal(sims, -np.inf)
    return np.argpartition(-sims, k, axis=1)[:, :k]


def recall(pred, gt_sets):
    return sum(len(set(pred[i]) & gt_sets[i]) for i in range(len(gt_sets))) / (len(gt_sets) * TOPK)


def main():
    d = np.load(EMB, allow_pickle=True)
    vecs = d["vecs"].astype(np.float32)
    vecs = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12)
    N = vecs.shape[0]
    print(f"corpus {vecs.shape}")

    # fixed reference: float-768 top-10
    gt_sets = [set(r) for r in topk_float(vecs, TOPK)]

    from remax import StackedSignBitQuantizer, hamming_distances

    mean_full = vecs.mean(axis=0).astype(np.float32)
    centered = vecs - mean_full

    grid = {}          # (dim,k) -> R@10 vs float-768
    ceilings = {}      # dim -> float R@10 vs float-768
    for dim in DIMS:
        trunc = np.ascontiguousarray(centered[:, :dim])
        tn = trunc / (np.linalg.norm(trunc, axis=1, keepdims=True) + 1e-12)
        ceilings[dim] = recall(topk_float(tn, TOPK), gt_sets)
        for k in KS:
            q = StackedSignBitQuantizer(d=dim, k=k, seed=SEED)
            codes = q.encode(trunc)
            pred = np.empty((N, TOPK), dtype=np.int64)
            for i in range(N):
                dist = hamming_distances(codes, codes[i]).astype(np.int64)
                dist[i] = 1 << 30
                pred[i] = np.argpartition(dist, TOPK)[:TOPK]
            r = recall(pred, gt_sets)
            grid[(dim, k)] = r
            print(f"dim={dim:3d} k={k:2d}  {dim*k//8:4d}B/chunk  R@10={r:.4f}")
        print(f"  [dim={dim} float ceiling {ceilings[dim]:.4f}]")

    out = {
        "corpus": "muninn.austegard.com (Mac search corpus)",
        "embedder": "gemini-embedding-001 @ output_dim=768, RETRIEVAL_DOCUMENT",
        "n_chunks": N, "seed": SEED, "topk": TOPK,
        "reference": "float-768 cosine top-10 (fixed)",
        "float_ceilings": {str(dm): ceilings[dm] for dm in DIMS},
        "grid": [{"dim": dm, "k": k, "bytes_per_chunk": dm * k // 8,
                  "R_at_10": grid[(dm, k)]} for dm in DIMS for k in KS],
    }
    (HERE / "dim_sweep.json").write_text(json.dumps(out, indent=2))

    # iso-byte Pareto: best (dim,k) per byte budget
    by_bytes = {}
    for dm in DIMS:
        for k in KS:
            b = dm * k // 8
            by_bytes.setdefault(b, []).append((grid[(dm, k)], dm, k))
    print("\n=== best config per byte budget ===")
    for b in sorted(by_bytes):
        r, dm, k = max(by_bytes[b])
        ties = sorted(by_bytes[b], reverse=True)
        print(f"{b:4d}B: best dim={dm} k={k} R@10={r:.4f}"
              + (f"   (vs {ties[1][1]}d/{ties[1][2]}k {ties[1][0]:.4f})" if len(ties) > 1 else ""))

    print("wrote dim_sweep.json")


if __name__ == "__main__":
    main()
