"""Where the numpy Hamming kernel's time actually goes, and the layout fix.

Two questions:
  1. Stage decomposition — is `np.bitwise_count` slow, or is `.sum(axis=1)`?
  2. Does the answer depend on code width? dab41dd6 used k=256 (4 uint64 words
     per row); remax-hamming-speedup ships d=512 k=4 (32 words). A 4-wide inner
     reduction amortises numpy's per-row pairwise-reduction machinery over four
     elements; a 32-wide one has 8x more to hide behind.

The fix is layout, not language: store the W words as W separate contiguous
columns (bit planes) and the row reduction becomes W-1 whole-array adds.
"""
import time
import numpy as np

rng = np.random.default_rng(0)


def t(f, reps=9):
    f()
    best = float('inf')
    for _ in range(reps):
        t0 = time.perf_counter(); f(); best = min(best, time.perf_counter() - t0)
    return best * 1e3


def build(n, code_bits, float_d):
    W = code_bits // 64
    C = np.ascontiguousarray(
        rng.integers(0, 2**63, size=(n, W), dtype=np.int64).view(np.uint64))
    X = rng.standard_normal((n, float_d), dtype=np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return C, C[0].copy(), X, X[0].copy(), W


def stages(n=42_500, code_bits=256):
    """Q1: which numpy op owns the time?"""
    C, qc, X, q, W = build(n, code_bits, 256)
    xr_ = np.empty_like(C)
    mb = C.nbytes / 1e6
    print(f"stage decomposition  n={n:,}  {code_bits}-bit codes ({mb:.2f} MB, {W} words/row)")
    for name, f in (('xor only', lambda: np.bitwise_xor(C, qc, out=xr_)),
                    ('bitwise_count only', lambda: np.bitwise_count(xr_, out=xr_)),
                    ('sum(axis=1) only', lambda: xr_.sum(axis=1)),
                    ('full expression', lambda: np.bitwise_count(C ^ qc).sum(axis=1))):
        v = t(f)
        print(f"    {name:<22}{v:8.3f} ms   {mb / v:6.1f} GB/s")


def soa_vs_naive(n, code_bits, float_d):
    """Q2: naive row expression vs SoA bit planes, against the BLAS float scan."""
    C, qc, X, q, W = build(n, code_bits, float_d)
    cols = [np.ascontiguousarray(C[:, j]) for j in range(W)]
    acc, tmp = np.empty(n, np.uint64), np.empty(n, np.uint64)
    ref = np.bitwise_count(C ^ qc).sum(axis=1).astype(np.int64)

    def naive():
        return np.bitwise_count(C ^ qc).sum(axis=1)

    def soa():
        np.bitwise_xor(cols[0], qc[0], out=acc); np.bitwise_count(acc, out=acc)
        for j in range(1, W):
            np.bitwise_xor(cols[j], qc[j], out=tmp); np.bitwise_count(tmp, out=tmp)
            np.add(acc, tmp, out=acc)
        return acc

    assert np.array_equal(np.asarray(soa()).astype(np.int64), ref)
    b, nv, sv = t(lambda: X @ q), t(naive), t(soa)
    mb = C.nbytes / 1e6
    print(f"  n={n:>8,}  codes {mb:6.2f} MB ({W} words/row)  float {X.nbytes/1e6:6.1f} MB"
          f"  ratio {X.nbytes / C.nbytes:.0f}x")
    print(f"      BLAS float      {b:8.3f} ms")
    print(f"      naive u64 expr  {nv:8.3f} ms  {b/nv:6.2f}x vs BLAS  ({mb/nv:5.2f} GB/s)")
    print(f"      SoA bit planes  {sv:8.3f} ms  {b/sv:6.2f}x vs BLAS  ({mb/sv:5.2f} GB/s)"
          f"  SoA/naive {nv/sv:.2f}x")


if __name__ == '__main__':
    stages()
    print("\n=== dab41dd6 config: k=256 sign bits, 32 B/row (4 words), float32 d=256 ===")
    for n in (42_500, 250_000):
        soa_vs_naive(n, 256, 256)
    print("\n=== remax-hamming-speedup config: d=512 k=4, 256 B/row (32 words), float32 d=768 ===")
    for n in (10_000, 50_000, 250_000):
        soa_vs_naive(n, 2048, 768)
