"""Reproduce the jina-v5-nano-mirror PERFORMANCE.md 'Small, not fast' caveat,
now against the MERGED remax_kb hamming_scan (PR #16).

The caveat read: "At muninn scale a numpy BLAS float-cosine scan beats the
current popcount Hamming scan (0.05 vs 0.50 ms/query @600 docs)." That described
the pre-#16 LUT kernel. This script times the *shipped* kernel imported straight
from the remax_kb checkout (no reimplementation) against the same float-cosine
BLAS baseline, to refresh the number.

Run: OPENBLAS_NUM_THREADS=1 python3 repro_merged.py
"""
from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

import numpy as np

# Import the SHIPPED kernel directly (avoids remax_kb/__init__ -> bm25s/remax).
_HAMMING_PATH = Path(__file__).resolve().parents[2] / ".spokes/remax_kb/remax_kb/_hamming.py"
_spec = importlib.util.spec_from_file_location("_hamming_merged", _HAMMING_PATH)
_h = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_h)
hamming_scan = _h.hamming_scan
top_k = _h.top_k

print(f"shipped kernel: {_HAMMING_PATH}")
print(f"  bitwise_count fast path active: {_h._HAS_BITWISE_COUNT}  (numpy {np.__version__})")

BITS = 2048          # d=512 * k=4  -> the recommended remax config in PERFORMANCE.md
ROW_BYTES = BITS // 8  # 256 B/row
FLOAT_DIM = 768      # jina-v5-nano output width (fp32 vector DB baseline)
K = 10
SEED = 20260626
rng = np.random.default_rng(SEED)


def float_cosine_scan(corpus_f32, query_f32):
    """The PERFORMANCE.md baseline: normalized fp32 dot via BLAS (-sim ascending)."""
    return -(corpus_f32 @ query_f32)


def time_fn(fn, args, reps, warmup=3):
    for _ in range(warmup):
        fn(*args)
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn(*args)
        best = min(best, time.perf_counter() - t0)
    return best * 1e3  # ms


def main():
    ns = [600, 2_000, 10_000, 50_000, 100_000, 500_000, 1_000_000]
    reps = 40
    rows = []
    print(f"\n{'N':>9} {'Hamming (merged)':>18} {'float cosine BLAS':>18} {'winner':>10} {'speedup':>9}")
    for n in ns:
        codes = np.ascontiguousarray(rng.integers(0, 256, size=(n, ROW_BYTES), dtype=np.uint8))
        query = rng.integers(0, 256, size=ROW_BYTES, dtype=np.uint8)
        corpus_f32 = rng.standard_normal((n, FLOAT_DIM)).astype(np.float32)
        corpus_f32 /= np.linalg.norm(corpus_f32, axis=1, keepdims=True)
        corpus_f32 = np.ascontiguousarray(corpus_f32)
        query_f32 = rng.standard_normal(FLOAT_DIM).astype(np.float32)
        query_f32 /= np.linalg.norm(query_f32)

        ham = time_fn(lambda c, q: hamming_scan(c, q), (codes, query), reps)
        cos = time_fn(float_cosine_scan, (corpus_f32, query_f32), reps)
        # sanity: kernel still returns sane top-k
        _ = top_k(hamming_scan(codes, query), K)
        winner = "Hamming" if ham <= cos else "cosine"
        speedup = cos / ham
        rows.append({"n": n, "hamming_ms": round(ham, 4), "float_cosine_ms": round(cos, 4),
                     "winner": winner, "hamming_x_cosine": round(speedup, 2)})
        print(f"{n:>9} {ham:>15.3f}ms {cos:>15.3f}ms {winner:>10} {speedup:>8.2f}x")
        del codes, corpus_f32

    out = {
        "meta": {
            "what": "MERGED remax_kb hamming_scan (PR #16) vs BLAS float-cosine, "
                    "reproducing jina-v5-nano-mirror PERFORMANCE.md 'Small, not fast' caveat",
            "numpy": np.__version__, "bits": BITS, "row_bytes": ROW_BYTES,
            "float_dim": FLOAT_DIM, "reps": reps, "seed": SEED,
            "blas_threads": "OPENBLAS_NUM_THREADS=1",
            "metric": "best-of-reps ms per query, single query vs full corpus",
            "bitwise_count_active": bool(_h._HAS_BITWISE_COUNT),
        },
        "results": rows,
    }
    Path("repro_merged.json").write_text(json.dumps(out, indent=2))
    print("\nwrote repro_merged.json")


if __name__ == "__main__":
    main()
