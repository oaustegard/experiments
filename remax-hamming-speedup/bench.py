"""Benchmark candidate 1-bit Hamming scan kernels against BLAS float cosine.

remax_kb issue #15: the current uint8 popcount-LUT gather (`hamming_scan`) is
~10x slower than an equivalent BLAS float-cosine search at small N. This script
times the candidate replacements from the issue on the same synthetic corpus and
verifies every candidate returns the IDENTICAL top-k ranking as the LUT baseline.

Corpus shape mirrors the muninn micro-bench: d=512, k=4 stacked 1-bit codes ->
2048 bits = 256 bytes/row. Float baseline is 768-d fp32 (3072 bytes/row).

Run: python3 bench.py            # full sweep, writes results.json
     python3 bench.py --quick    # fewer N, fewer reps
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

BITS = 2048                  # d=512 * k=4
ROW_BYTES = BITS // 8        # 256 bytes/row, packed
FLOAT_DIM = 768              # fp32 baseline vector width
K = 10                       # top-k retrieved
SEED = 20260626

rng = np.random.default_rng(SEED)

# ── kernels ──────────────────────────────────────────────────────────────────

POPCOUNT_LUT = np.array([bin(b).count("1") for b in range(256)], dtype=np.uint16)


def lut_scan(codes_u8: np.ndarray, query_u8: np.ndarray) -> np.ndarray:
    """Baseline: current hamming_scan. (N,B) uint8 XOR -> LUT gather -> sum."""
    xor = np.bitwise_xor(codes_u8, query_u8[None, :])
    return POPCOUNT_LUT[xor].sum(axis=1, dtype=np.int32)


def bitcount_u8_scan(codes_u8: np.ndarray, query_u8: np.ndarray) -> np.ndarray:
    """Approach 1a: drop the LUT, popcount the uint8 XOR directly."""
    xor = np.bitwise_xor(codes_u8, query_u8[None, :])
    return np.bitwise_count(xor).sum(axis=1, dtype=np.int32)


def bitcount_u64_scan(codes_u64: np.ndarray, query_u64: np.ndarray) -> np.ndarray:
    """Approach 1b: view rows as uint64 (8x fewer elems), XOR, hw popcount, sum."""
    xor = np.bitwise_xor(codes_u64, query_u64[None, :])
    return np.bitwise_count(xor).sum(axis=1, dtype=np.int32)


def pm1_matmul_f32(corpus_pm1_f32: np.ndarray, query_pm1_f32: np.ndarray) -> np.ndarray:
    """Approach 2: +-1 codes ranked by BLAS matmul.

    dot = nbits - 2*Hamming, so SORTING by -dot == sorting by Hamming.
    Returns -dot as the rank key (ascending == nearest), an fp32 surrogate for
    Hamming distance (monotone, not equal).
    """
    return -(corpus_pm1_f32 @ query_pm1_f32)


def pm1_matmul_i8(corpus_pm1_i8: np.ndarray, query_pm1_i8: np.ndarray) -> np.ndarray:
    """Approach 2 variant: int8 +-1 corpus, accumulate in int16 via int32 matmul."""
    # numpy has no native int8 GEMM; promote to int16 to keep it integer + exact.
    return -(corpus_pm1_i8.astype(np.int16) @ query_pm1_i8.astype(np.int16))


def float_cosine(corpus_f32: np.ndarray, query_f32: np.ndarray) -> np.ndarray:
    """The thing we're trying to beat: normalized fp32 dot via BLAS. (-sim ascending)"""
    return -(corpus_f32 @ query_f32)


# ── data prep ────────────────────────────────────────────────────────────────

# Above this N the +-1 matmul corpora (8-32x the packed size) blow past RAM and
# aren't competitive anyway, so we skip them and time only popcount + cosine.
PM1_MAX_N = 100_000


def make_corpus(n: int, with_pm1: bool = True):
    codes_u8 = rng.integers(0, 256, size=(n, ROW_BYTES), dtype=np.uint8)
    codes_u8 = np.ascontiguousarray(codes_u8)
    query_u8 = rng.integers(0, 256, size=ROW_BYTES, dtype=np.uint8)

    # uint64 view (zero-copy; ROW_BYTES divisible by 8)
    codes_u64 = codes_u8.view(np.uint64)
    query_u64 = query_u8.view(np.uint64)

    d = dict(
        codes_u8=codes_u8, query_u8=query_u8,
        codes_u64=codes_u64, query_u64=query_u64,
    )

    if with_pm1:
        # +-1 unpack of the same bits: bit set -> +1, clear -> -1
        bits = np.unpackbits(codes_u8, axis=1)            # (N, 2048) uint8 {0,1}
        qbits = np.unpackbits(query_u8)                   # (2048,)
        corpus_pm1_i8 = (bits.astype(np.int8) * 2 - 1)
        query_pm1_i8 = (qbits.astype(np.int8) * 2 - 1)
        corpus_pm1_f32 = np.ascontiguousarray(corpus_pm1_i8.astype(np.float32))
        query_pm1_f32 = qbits.astype(np.float32) * 2 - 1
        d.update(
            corpus_pm1_i8=corpus_pm1_i8, query_pm1_i8=query_pm1_i8,
            corpus_pm1_f32=corpus_pm1_f32, query_pm1_f32=query_pm1_f32,
        )

    # independent fp32 baseline corpus (768-d, unit-norm)
    corpus_f32 = rng.standard_normal((n, FLOAT_DIM)).astype(np.float32)
    corpus_f32 /= np.linalg.norm(corpus_f32, axis=1, keepdims=True)
    corpus_f32 = np.ascontiguousarray(corpus_f32)
    query_f32 = rng.standard_normal(FLOAT_DIM).astype(np.float32)
    query_f32 /= np.linalg.norm(query_f32)
    d.update(corpus_f32=corpus_f32, query_f32=query_f32)
    return d


def topk(scores: np.ndarray, k: int) -> np.ndarray:
    k = min(k, scores.shape[0])
    cut = np.argpartition(scores, k - 1)[:k]
    return cut[np.argsort(scores[cut], kind="stable")]


# ── timing ───────────────────────────────────────────────────────────────────

def time_fn(fn, args, reps: int, warmup: int = 3) -> float:
    for _ in range(warmup):
        fn(*args)
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn(*args)
        best = min(best, time.perf_counter() - t0)
    return best  # seconds (best-of, to suppress scheduler noise)


def run(ns, reps):
    kernels = [
        ("lut_baseline",   lut_scan,        lambda d: (d["codes_u8"], d["query_u8"]),   True),
        ("bitcount_u8",    bitcount_u8_scan, lambda d: (d["codes_u8"], d["query_u8"]),  True),
        ("bitcount_u64",   bitcount_u64_scan, lambda d: (d["codes_u64"], d["query_u64"]), True),
        ("pm1_matmul_f32", pm1_matmul_f32,  lambda d: (d["corpus_pm1_f32"], d["query_pm1_f32"]), True),
        ("pm1_matmul_i8",  pm1_matmul_i8,   lambda d: (d["corpus_pm1_i8"], d["query_pm1_i8"]),   True),
        ("float_cosine",   float_cosine,    lambda d: (d["corpus_f32"], d["query_f32"]), False),
    ]
    results = []
    for n in ns:
        with_pm1 = n <= PM1_MAX_N
        d = make_corpus(n, with_pm1=with_pm1)
        ref_rank = None
        row = {"n": n}
        for name, fn, pick, is_hamming in kernels:
            if name.startswith("pm1_") and not with_pm1:
                continue
            args = pick(d)
            secs = time_fn(fn, args, reps)
            ms = secs * 1e3
            row[name] = round(ms, 4)
            # ranking-equivalence check vs LUT baseline (Hamming kernels only)
            if is_hamming:
                rank = topk(fn(*args), K)
                if name == "lut_baseline":
                    ref_rank = rank
                    row[name + "_match"] = True
                else:
                    row[name + "_match"] = bool(np.array_equal(rank, ref_rank))
        results.append(row)
        cols = ("lut_baseline","bitcount_u8","bitcount_u64",
                "pm1_matmul_f32","pm1_matmul_i8","float_cosine")
        print(f"N={n:>7}  " + "  ".join(
            f"{k}={row[k]:.3f}ms" for k in cols if k in row
        ))
        bad = [k for k in row if k.endswith("_match") and not row[k]]
        if bad:
            print(f"   !! ranking mismatch: {bad}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        ns, reps = [600, 10_000, 100_000], 20
    else:
        ns, reps = [600, 2_000, 10_000, 50_000, 100_000, 500_000, 1_000_000], 40
    print(f"numpy {np.__version__}  BITS={BITS} ROW_BYTES={ROW_BYTES} K={K} reps={reps}")
    results = run(ns, reps)
    out = {
        "meta": {
            "numpy": np.__version__, "bits": BITS, "row_bytes": ROW_BYTES,
            "float_dim": FLOAT_DIM, "k": K, "reps": reps, "seed": SEED,
            "metric": "best-of-reps milliseconds per query (single query vs full corpus)",
        },
        "results": results,
    }
    with open("results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote results.json")


if __name__ == "__main__":
    main()
