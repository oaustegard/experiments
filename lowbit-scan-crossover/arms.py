"""The two disconfirming arms the adversarial pass demanded.

(1) ISA arm  — same C source at three -march levels. v3 = AVX2, no VPOPCNTDQ.
               v2 = SSE4.2 + scalar popcnt. The BLAS baseline keeps AVX-512
               either way, so this is a conservative test for the code side.
(2) Cache arm — evict L3 (300 MB scratch stream) between every timed rep, so
               neither the 43.5 MB float32 matrix nor the 1.36 MB codes are warm.
"""
import ctypes, time, sys, os
import numpy as np

K, W = 256, 4
rng = np.random.default_rng(0)
SCRATCH = np.empty(300 * 1024 * 1024 // 8, dtype=np.uint64)   # > 260 MB L3


def flush():
    SCRATCH[::8] += 1                       # touch every cache line


def make(n):
    X = rng.standard_normal((n, K), dtype=np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    C = np.ascontiguousarray(np.packbits(X > 0, axis=1, bitorder='little').view(np.uint64))
    return X, C, X[0].copy(), np.ascontiguousarray(C[:1])


def t(fn, reps=9, cold=False):
    fn()
    ts = []
    for _ in range(reps):
        if cold:
            flush()
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return (float(np.median(ts)) if cold else min(ts)) * 1e3


LIBS = {}
for tag, flags in (('v4+vpopcnt', '-march=native'),
                   ('v3 (AVX2)', '-march=x86-64-v3'),
                   ('v2 (SSE4.2)', '-march=x86-64-v2')):
    so = f'ham_{tag.split()[0]}.so'
    os.system(f'gcc -O3 {flags} -funroll-loops -shared -fPIC -o {so} hamkern.c')
    n_vp = int(os.popen(f"objdump -d {so} | grep -c vpopcnt").read() or 0)
    lib = ctypes.CDLL(f'./{so}')
    lib.ham_scan.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_long]
    lib.ham_scan.restype = None
    LIBS[tag] = (lib, n_vp)

for n in (42_500, 250_000):
    X, C, q, qc = make(n)
    out = np.empty(n, dtype=np.uint16)
    cp, qp, op = C.ctypes.data, qc.ctypes.data, out.ctypes.data
    ref = np.bitwise_count(C ^ qc[0]).sum(axis=1).astype(np.int64)

    for cold in (False, True):
        label = 'COLD (L3 evicted each rep)' if cold else 'warm (best-of-9)'
        base = t(lambda: X @ q, cold=cold)
        print(f"\nn={n:,}  {label}")
        print(f"  {'kernel':<16}{'ms':>9}{'GB/s':>9}{'vs sgemv':>11}")
        print(f"  {'f32 sgemv':<16}{base:>9.3f}{X.nbytes/1e6/base:>9.1f}{1.0:>10.2f}x")
        for tag, (lib, n_vp) in LIBS.items():
            v = t(lambda: lib.ham_scan(cp, qp, op, n), cold=cold)
            lib.ham_scan(cp, qp, op, n)
            assert np.array_equal(out.astype(np.int64), ref), tag
            print(f"  {'C ' + tag:<16}{v:>9.3f}{C.nbytes/1e6/v:>9.1f}{base/v:>10.2f}x"
                  f"   [{n_vp} vpopcnt]")
        npv = t(lambda: np.bitwise_count(C ^ qc[0]).sum(axis=1), cold=cold)
        print(f"  {'np popcount':<16}{npv:>9.3f}{C.nbytes/1e6/npv:>9.1f}{base/npv:>10.2f}x")
