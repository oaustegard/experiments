"""Measure the 100M-row claim directly instead of extrapolating to it.

The blog post "Three Gigs to Search a Hundred Million Papers" claims
"single-threaded brute force returns top-100 candidates in well under a
second". 100M x 32 B = 3.2 GB, which fits in a 16 GB container, so this needs
no extrapolation.

Finding: the scan is ~316 ms and fine; `np.argpartition` top-100 over the
resulting 100M-element score array costs another ~714 ms and blows the budget.
Fusing selection into the scan (emit ids under a distance threshold, never
materialise scores) lands at ~395 ms total.

Needs ~4.5 GB RSS at n=1e8. Pass a smaller n to check the shape cheaply.
"""
import ctypes
import sys
import time

import numpy as np

lib = ctypes.CDLL('./hamkern.so')
lib.ham_scan.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_long]
lib.ham_scan.restype = None
lib.ham_scan_thresh.argtypes = ([ctypes.c_void_p] * 3
                                + [ctypes.c_long, ctypes.c_int, ctypes.c_long])
lib.ham_scan_thresh.restype = ctypes.c_long

CAP = 1 << 20
K = 100


def build(n, chunk=4_000_000):
    """Random codes, filled in chunks so peak RSS stays near the array itself."""
    rng = np.random.default_rng(0)
    C = np.empty((n, 4), np.uint64)
    for i in range(0, n, chunk):
        j = min(i + chunk, n)
        C[i:j] = rng.integers(0, 2**63, size=(j - i, 4), dtype=np.int64).view(np.uint64)
    return C


def t(fn, reps=3):
    fn()                                    # WARM FIRST — see the note below
    best = float('inf')
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); best = min(best, time.perf_counter() - t0)
    return best * 1e3


def run(n):
    C = build(n)
    out = np.empty(n, np.uint16)
    ids = np.empty(CAP, np.uint32)
    qc = C[0].copy()
    cp, qp, op, ip = (a.ctypes.data for a in (C, qc, out, ids))

    scan = t(lambda: lib.ham_scan(cp, qp, op, n))
    lib.ham_scan(cp, qp, op, n)

    # The unwarmed version of this call reports ~5.4 s at n=1e8 — page faults on
    # the fresh score array plus an 800 MB index allocation. Warm it is ~714 ms.
    part = t(lambda: np.argpartition(out, K)[:K])

    def counting():
        h = np.bincount(out, minlength=257)
        thr = int(np.searchsorted(np.cumsum(h), K))
        cand = np.flatnonzero(out <= thr)
        return cand[np.argsort(out[cand], kind='stable')[:K]]

    count = t(counting)

    thr = int(np.searchsorted(np.cumsum(np.bincount(out, minlength=257)), K))
    fused = t(lambda: lib.ham_scan_thresh(cp, qp, ip, n, thr, CAP))
    m = lib.ham_scan_thresh(cp, qp, ip, n, thr, CAP)

    # Exactness: ties at the boundary make the *index* set arbitrary, so compare
    # the distance multiset, which is what "top-k" actually pins down.
    exact_d = np.sort(out[np.argpartition(out, K)[:K]])
    fused_d = np.sort(out[ids[:m]])[:K]

    gb = C.nbytes / 1e9
    print(f"\nn={n:,}   codes {gb:.2f} GB   threshold d<={thr}, {m} candidates")
    print(f"  scan                          {scan:8.1f} ms   ({gb / (scan / 1e3):.1f} GB/s)")
    print(f"  + np.argpartition top-{K}     {part:8.1f} ms   -> {(scan + part) / 1000:.3f} s")
    print(f"  + counting-sort top-{K}       {count:8.1f} ms   -> {(scan + count) / 1000:.3f} s")
    print(f"  fused scan+threshold          {fused:8.1f} ms   -> {fused / 1000:.3f} s")
    print(f"  fused top-{K} is exact (distance multiset): "
          f"{np.array_equal(exact_d, fused_d)}")
    del C, out


if __name__ == '__main__':
    for n in ([int(x) for x in sys.argv[1:]] or [16_000_000, 100_000_000]):
        run(n)
