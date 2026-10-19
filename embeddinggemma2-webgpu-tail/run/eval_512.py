"""d=512 against d=768 for the 2-bit remex codes the tail produces, on real EG2 vectors.

Usage: python3 -I run/eval_512.py [--n 100000] [--nq 1000]

Corpus: the 100k EG2 Quora docs and 1,000 test queries in out/eg2q (embstore set
quora/embeddinggemma-2/fp32-768-first100k). Arms, each a top-10 per query:
  fp32_768   exact inner product on the full vectors (ground truth)
  fp32_512   exact, after truncating docs and queries to 512 dims and renormalizing
  rx2_768    remex Quantizer(768, bits=2, seed=42, mean) codes, ADC scores
  rx2_512    the same at d=512 on the truncated, renormalized vectors (mean of those)
Metrics: recall@10 of each arm against fp32_768 and against fp32_512, and top-10
overlap of rx2_512 with rx2_768 (truncation and quantization error add up, so the
codes are compared with each other, not only with floats). Paired bootstrap 95% CI
over queries for rx2_512 - rx2_768. Writes results/dim512.json.
"""
import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import remex

ROOT = Path(__file__).resolve().parent.parent
VEC = ROOT / "out" / "eg2q"
K = 10


def unit(a):
    return a / np.linalg.norm(a, axis=1, keepdims=True)


def topk_exact(Q, X, k=K):
    out = np.empty((len(Q), k), np.int64)
    for s in range(0, len(Q), 100):
        sc = Q[s:s + 100] @ X.T
        part = np.argpartition(-sc, k, axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(sc, part, 1), axis=1)
        out[s:s + 100] = np.take_along_axis(part, order, 1)
    return out


def topk_codes(X, Q, d):
    mu = X.mean(0).astype(np.float32)
    qz = remex.Quantizer(d, bits=2, seed=42, mean=mu)
    t0 = time.perf_counter()
    cv = qz.encode(X)
    enc_s = time.perf_counter() - t0
    idx = np.empty((len(Q), K), np.int64)
    for i, q in enumerate(Q):
        idx[i] = qz.search_adc(cv, q, k=K)[0]
    return idx, enc_s


def recall(a, b):
    """Per-query |a ∩ b| / K."""
    return np.array([len(set(x) & set(y)) / K for x, y in zip(a, b)])


def boot_ci(diff, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    m = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(n)]
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100000)
    ap.add_argument("--nq", type=int, default=1000)
    a = ap.parse_args()
    X = np.concatenate([np.load(f) for f in sorted(glob.glob(str(VEC / "shards" / "docs_*.npy")))])[: a.n]
    X = unit(X.astype(np.float32))
    Q = unit(np.load(VEC / "q_dense.npy").astype(np.float32)[: a.nq])
    X5, Q5 = unit(X[:, :512].copy()), unit(Q[:, :512].copy())

    gt768 = topk_exact(Q, X)
    gt512 = topk_exact(Q5, X5)
    rx768, enc768 = topk_codes(X, Q, 768)
    rx512, enc512 = topk_codes(X5, Q5, 512)

    r = {
        "n": len(X), "nq": len(Q), "k": K, "bits": 2,
        "bytes_per_vector": {"768": 768 * 2 // 8, "512": 512 * 2 // 8},
        "recall_at_10_vs_fp32_768": {
            "fp32_512": float(recall(gt512, gt768).mean()),
            "rx2_768": float(recall(rx768, gt768).mean()),
            "rx2_512": float(recall(rx512, gt768).mean()),
        },
        "recall_at_10_vs_fp32_512": {
            "rx2_512": float(recall(rx512, gt512).mean()),
        },
        "overlap_rx2_512_with_rx2_768": float(recall(rx512, rx768).mean()),
        "encode_s_native": {"768": round(enc768, 2), "512": round(enc512, 2)},
    }
    d = recall(rx512, gt768) - recall(rx768, gt768)
    r["rx2_512_minus_rx2_768_vs_fp32_768"] = {"mean": float(d.mean()), "ci95": boot_ci(d)}
    out = ROOT / "results" / "dim512.json"
    out.write_text(json.dumps(r, indent=1) + "\n")
    print(json.dumps(r, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
