"""Two honest counter-cases for BLAS at small n:
  (a) tiny n, where per-call overhead beats both kernels
  (b) batched queries, where sgemv becomes GEMM and BLAS goes compute-bound
      while a naive Hamming scan re-streams the corpus per query.
"""
import ctypes, time, sys
import numpy as np

K, W = 256, 4
lib = ctypes.CDLL('./hamkern.so')
lib.ham_scan.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_long]
lib.ham_scan.restype = None

rng = np.random.default_rng(0)


def make(n, b=1):
    X = rng.standard_normal((n, K), dtype=np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    C = np.ascontiguousarray(np.packbits(X > 0, axis=1, bitorder='little').view(np.uint64))
    Q = np.ascontiguousarray(X[:b].T.copy())          # (K, b) for GEMM
    QC = np.ascontiguousarray(C[:b])
    return X, C, Q, QC


def t(fn, reps=15):
    fn()
    return min((lambda: (lambda t0: (fn(), time.perf_counter() - t0)[1])(time.perf_counter()))()
               for _ in range(reps)) * 1e3


print("== (a) tiny n: where does per-call overhead dominate? ==")
print(f"  {'n':>8}{'sgemv ms':>11}{'C ham ms':>11}{'speedup':>10}")
for n in (100, 500, 1_000, 2_000, 5_000):
    X, C, Q, QC = make(n)
    q = np.ascontiguousarray(Q[:, 0].copy())
    out = np.empty(n, dtype=np.uint16)
    cp, qp, op = C.ctypes.data, QC.ctypes.data, out.ctypes.data
    a = t(lambda: X @ q)
    b = t(lambda: lib.ham_scan(cp, qp, op, n))
    print(f"  {n:>8,}{a:>11.4f}{b:>11.4f}{a/b:>10.2f}x")

print("\n== (b) batched queries at n=42,500: BLAS GEMM vs per-query Hamming ==")
n = 42_500
print(f"  {'batch':>6}{'GEMM ms/q':>12}{'C ham ms/q':>12}{'speedup':>10}")
for b in (1, 8, 64, 256, 1024):
    X, C, Q, QC = make(n, b)
    out = np.empty(n, dtype=np.uint16)
    cp, op = C.ctypes.data, out.ctypes.data
    qptrs = [QC[i:i+1].ctypes.data for i in range(b)]
    ga = t(lambda: X @ Q, reps=5) / b

    def ham_batch():
        for p in qptrs:
            lib.ham_scan(cp, p, op, n)
    ha = t(ham_batch, reps=5) / b
    print(f"  {b:>6}{ga:>12.4f}{ha:>12.4f}{ga/ha:>10.2f}x")
