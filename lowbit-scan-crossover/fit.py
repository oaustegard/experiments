"""Fit t(n) = a + b*n to a scan benchmark, and derive whether a crossover
can exist at all.

Family model for any linear scan:

    t(n) = a + (bytes_per_candidate * n) / (bandwidth * efficiency)

`n` enters only as a multiplier on the linear term, identically for every
kernel. So two kernels can cross only if at least one has a > 0. With a = 0 on
both sides the ratio is constant in n and no corpus size flips the ordering.

Run this before reporting any crossover. If a ~ 0 on both sides, a reported
crossover is an artifact of the two n values that bracket it. If a > 0, the
constant IS the finding, and it is a property of the harness, not the corpus.
"""
import numpy as np


def fit(pts, bytes_per_row=None, drop_tail=False):
    """pts: [(n, ms), ...]. Returns (a_ms, b_ns_per_row, max_abs_resid_ms)."""
    pts = sorted(pts)
    if drop_tail:                       # last point is usually DRAM-superlinear
        pts = pts[:-1]
    n = np.array([p[0] for p in pts], float)
    t = np.array([p[1] for p in pts], float)
    (a, b), *_ = np.linalg.lstsq(np.vstack([np.ones_like(n), n]).T, t, rcond=None)
    resid = t - (a + b * n)
    gbs = (bytes_per_row / (b * 1e6)) if bytes_per_row else float('nan')
    return a, b * 1e6, float(np.max(np.abs(resid))), gbs


def crossover(a_compressed, b_compressed_ns, b_baseline_ns):
    """n* where the compressed kernel overtakes the baseline, or None."""
    d = (b_baseline_ns - b_compressed_ns) * 1e-6      # ms per row
    if a_compressed <= 0 or d <= 0:
        return None
    return a_compressed / d


SERIES = {
    # dab41dd6, 2026-08-09, 1 vCPU container. k=256 sign bits, 32 B/row.
    'dab41dd6 numpy hamming': (
        [(42_500, 5.1), (250_000, 12.7), (1_000_000, 36.4),
         (4_000_000, 150.6), (16_000_000, 745.9)], 32, 2),
    'dab41dd6 float32 BLAS': (
        [(42_500, 3.5), (250_000, 22.6), (1_000_000, 92.5)], 1024, 0),
    # this session, 4 vCPU Xeon, same expression.
    'here numpy hamming': (
        [(42_500, 1.152), (250_000, 7.087), (1_000_000, 33.329)], 32, 0),
    'here float32 BLAS': (
        [(42_500, 1.950), (250_000, 19.859), (1_000_000, 77.869)], 1024, 0),
    'here C hamming': (
        [(42_500, 0.052), (250_000, 0.395), (1_000_000, 1.456)], 32, 0),
}

if __name__ == '__main__':
    out = {}
    print(f"  {'series':<26}{'a (ms)':>10}{'b (ns/row)':>13}{'GB/s':>8}{'max resid':>11}")
    for name, (pts, bpr, ndrop) in SERIES.items():
        p = pts[:len(pts) - ndrop] if ndrop else pts
        a, b, r, gbs = fit(p, bpr)
        out[name] = (a, b)
        print(f"  {name:<26}{a:>10.3f}{b:>13.2f}{gbs:>8.2f}{r:>11.2f}")

    print("\n  crossover n* = a_ham / (b_f32 - b_ham)")
    for tag, ham, f32 in (('dab41dd6', 'dab41dd6 numpy hamming', 'dab41dd6 float32 BLAS'),
                          ('here numpy', 'here numpy hamming', 'here float32 BLAS'),
                          ('here C', 'here C hamming', 'here float32 BLAS')):
        n_star = crossover(out[ham][0], out[ham][1], out[f32][1])
        lo = min(p[0] for p in SERIES[ham][0])
        if not n_star:
            note = "no positive solution — the kernels never cross"
        elif n_star < lo:
            note = (f"n* = {n_star:,.0f} — BELOW the fitted range (n>={lo:,}); "
                    "extrapolation artifact, not a crossover")
        else:
            note = f"n* = {n_star:,.0f}"
        print(f"    {tag:<12}{note}")
