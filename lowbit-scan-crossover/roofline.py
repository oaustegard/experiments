"""How far is each kernel from the memory-bandwidth roofline at small n?

Compares, on identical data:
  f32 BLAS sgemv        (the 'BLAS baseline')
  numpy bitwise_count   (the kernel the crossover was measured with)
  C + VPOPCNTDQ         (the same arithmetic, SIMD)
  xor-only read probe   (achievable streaming bandwidth on the packed codes)
"""
import ctypes, time, sys
import numpy as np

K, W, REPS, TOPK = 256, 4, 9, 1000
lib = ctypes.CDLL('./hamkern.so')
lib.ham_scan.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_long]
lib.ham_scan.restype = None
lib.ham_scan_thresh.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_long, ctypes.c_int, ctypes.c_long]
lib.ham_scan_thresh.restype = ctypes.c_long
lib.band_probe.argtypes = [ctypes.c_void_p, ctypes.c_long]
lib.band_probe.restype = ctypes.c_uint64

rng = np.random.default_rng(0)


def make(n):
    X = rng.standard_normal((n, K), dtype=np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    C = np.packbits(X > 0, axis=1, bitorder='little').view(np.uint64)
    return X, np.ascontiguousarray(C), X[0].copy(), C[0].copy()


def t(fn, reps=REPS):
    fn()
    return min(_timed(fn) for _ in range(reps)) * 1e3


def _timed(fn):
    t0 = time.perf_counter(); fn(); return time.perf_counter() - t0


def run(n):
    X, C, q, qc = make(n)
    out = np.empty(n, dtype=np.uint16)
    ids = np.empty(n, dtype=np.uint32)
    cp, qp, op, ip = (a.ctypes.data for a in (C, qc, out, ids))

    ms = {}
    ms['f32 sgemv'] = t(lambda: X @ q)
    ms['np popcount'] = t(lambda: np.bitwise_count(C ^ qc).sum(axis=1))
    ms['C vpopcnt'] = t(lambda: lib.ham_scan(cp, qp, op, n))
    ms['C fused thr'] = t(lambda: lib.ham_scan_thresh(cp, qp, ip, n, 96, n))
    ms['xor read'] = t(lambda: lib.band_probe(cp, n * W))

    # sanity: C kernel agrees with numpy
    lib.ham_scan(cp, qp, op, n)
    assert np.array_equal(out.astype(np.int64),
                          np.bitwise_count(C ^ qc).sum(axis=1).astype(np.int64))

    code_MB, f32_MB = C.nbytes / 1e6, X.nbytes / 1e6
    print(f"\nn={n:,}   codes {code_MB:.2f} MB   float32 {f32_MB:.1f} MB")
    print(f"  {'kernel':<14}{'ms':>9}{'GB/s over its own bytes':>27}{'vs sgemv':>11}")
    for name, v in ms.items():
        byts = f32_MB if name == 'f32 sgemv' else code_MB
        print(f"  {name:<14}{v:>9.3f}{byts / 1e3 / (v / 1e3):>27.1f}{ms['f32 sgemv'] / v:>11.2f}x")
    return ms


if __name__ == '__main__':
    ns = [int(x) for x in sys.argv[1:]] or [5_000, 42_500, 250_000, 1_000_000]
    for n in ns:
        run(n)
