#!/usr/bin/env python3
"""Does the rotation pendulum (my ITQ 'win' vs remax#46's ITQ 'loss') resolve to
PROTOCOL (in-corpus overfit) or EMBEDDER (specialized vs general)? And does a
DECORRELATED multi-rotation beat random SimHash where ITQ failed?

Metric = remax/bench's own: self-retrieval recall@10 of the float32 top-10
(exact_knn ground truth). Pure numpy on precomputed caches — no embedding.

Two embedders, same scientific domain:
  * SPECTER2 (specialized, 10k corpus)      — the setting remax#46 used
  * Jina-v5-nano (general) on SciFact (5183) — the general embedder

Protocols (the #46 control):
  * in-corpus : fit rotation on the EVAL split
  * transfer  : fit rotation on a DISJOINT split  (defeats in-corpus overfit)

Methods on the k-stack ladder (k in {1,2,4,8}, codes concatenated, Hamming):
  * simhash : k independent random orthogonal rotations (remax default; diverse)
  * itq     : k independent ITQ rotations (each min sign-MSE; aligned but corr.)
  * decorr  : k = QR(alpha*R_itq_shared + (1-alpha)*G_i) — shared alignment +
              per-stack random diversity; tests the diversity/alignment tradeoff
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
C = HERE / "cache"
K_LADDER = (1, 2, 4, 8)
RK = 10                      # recall@10, as in #46
N_SEEDS = 3
DECORR_ALPHA = 0.5


def l2(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return (X / np.where(n == 0, 1, n)).astype(np.float32)


def rand_orth(d, rng):
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    return Q.astype(np.float32)


def itq_rotation(Xc, rng, iters=30, pca_dim=None):
    """Rotation R (d x b) minimizing ||sign(Xc@R) - Xc@R||. Full-dim (b=d) by
    default; classic PCA-to-b variant if pca_dim set."""
    d = Xc.shape[1]
    b = pca_dim or d
    if pca_dim:
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        P = Vt[:b].T
        V = Xc @ P
    else:
        P, V = np.eye(d, dtype=np.float32), Xc
    R = rand_orth(b, rng)
    for _ in range(iters):
        B = np.sign(V @ R); B[B == 0] = 1
        U, _, Vt = np.linalg.svd(B.T @ V)
        R = (Vt.T @ U.T)
    return (P @ R).astype(np.float32)        # (d, b): maps centered vec -> b bits


def encode(Xc, rotations):
    """Concatenate sign bits across the stack list -> packed uint8 codes."""
    bits = np.concatenate([(Xc @ R) > 0 for R in rotations], axis=1)
    return np.packbits(bits.astype(np.uint8), axis=1)


def build_rotations(method, Xfit_c, k, rng, alpha=DECORR_ALPHA):
    d = Xfit_c.shape[1]
    if method == "simhash":
        return [rand_orth(d, rng) for _ in range(k)]
    if method == "itq":
        return [itq_rotation(Xfit_c, np.random.default_rng(rng.integers(1 << 30)))
                for _ in range(k)]
    if method == "decorr":
        R_itq = itq_rotation(Xfit_c, rng)
        out = []
        for _ in range(k):
            G = rng.standard_normal((d, d)).astype(np.float32)
            Q, _ = np.linalg.qr(alpha * R_itq + (1 - alpha) * G)
            out.append(Q.astype(np.float32))
        return out
    raise ValueError(method)


def hamming_topk(dcodes, qcode, k):
    # popcount LUT
    xor = np.bitwise_xor(dcodes, qcode[None, :])
    dist = _LUT[xor].sum(axis=1)
    return np.argpartition(dist, k)[:k]


_LUT = np.array([bin(b).count("1") for b in range(256)], dtype=np.uint16)


def recall_at_k(Xcorpus, Xq, q_self_idx, rotations, mean, rk):
    """Self-retrieval recall@rk of float32 top-rk, excluding identity."""
    Cc = Xcorpus - mean
    dcodes = encode(Cc, rotations)
    qcodes = encode(Xq - mean, rotations)
    # float32 GT (exclude self)
    sims = Xq @ Xcorpus.T                                   # (nq, N)
    for i, si in enumerate(q_self_idx):
        if si >= 0:
            sims[i, si] = -np.inf
    gt = np.argpartition(-sims, rk, axis=1)[:, :rk]
    hits = 0
    for i in range(Xq.shape[0]):
        cand = hamming_topk(dcodes, qcodes[i], rk + (1 if q_self_idx[i] >= 0 else 0))
        cand = [c for c in cand if c != q_self_idx[i]][:rk]
        hits += len(set(cand) & set(gt[i].tolist()))
    return hits / (Xq.shape[0] * rk)


def run_dataset(name, X, transfer_X):
    X = l2(X)
    rng0 = np.random.default_rng(0)
    idx = rng0.permutation(X.shape[0])
    half = X.shape[0] // 2
    eval_set = X[idx[:half]]
    disjoint = X[idx[half:]]                  # same-embedder disjoint split (transfer)
    nq = min(500, eval_set.shape[0])
    qidx = np.arange(nq)                       # queries = first nq of eval (in corpus)
    Xq = eval_set[qidx]
    print(f"\n##### {name}: eval corpus {eval_set.shape}, {nq} queries #####", flush=True)
    print(f"{'method':<9}{'protocol':<10}" + "".join(f"k={k:<6}" for k in K_LADDER), flush=True)
    fit_sets = {"in-corpus": eval_set, "transfer": disjoint}
    for method in ("simhash", "itq", "decorr"):
        for proto, Xfit in fit_sets.items():
            mean = Xfit.mean(0).astype(np.float32)
            Xfit_c = Xfit - mean
            row = []
            for k in K_LADDER:
                vals = []
                for s in range(N_SEEDS):
                    rng = np.random.default_rng(100 * s + k)
                    rots = build_rotations(method, Xfit_c, k, rng)
                    vals.append(recall_at_k(eval_set, Xq, qidx, rots, mean, RK))
                row.append(np.mean(vals))
            cells = "".join(f"{v:<8.3f}" for v in row)
            print(f"{method:<9}{proto:<10}{cells}", flush=True)


def main():
    spec = np.load(C / "specter2.npy")
    jina = np.load(C / "jina_scifact_corpus.npy")
    run_dataset("SPECTER2 (specialized)", spec, None)
    run_dataset("Jina-v5 SciFact (general)", jina, None)
    print("\nRead: itq in-corpus vs transfer gap = overfit. simhash is the ladder "
          "baseline. decorr>simhash under TRANSFER on the ladder = the open lead pays.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
