#!/usr/bin/env python3
"""What does the Gaussian marginal approximation cost the exact-norm arm?

TurboQuant (arXiv:2504.19874 sec. 3) is explicit that after rotation the
coordinates of a **unit** vector follow a Beta distribution, and it fits its
per-coordinate Lloyd-Max quantizer to that Beta rather than to a Gaussian:

    "[because the distribution of coordinates of unit vec]tors follows a Beta
     distribution, we design optimal Lloyd-Max quantizer for each coordinate
     by solving a continuous k-means problem."

This experiment's exact-norm arm instead quantizes the unit direction with a
*Gaussian* Lloyd-Max quantizer at sigma = 1/sqrt(d).  That is the right
asymptotic limit — one coordinate of a uniform unit vector in R^d has density
proportional to (1 - x^2)^((d-3)/2), which converges to N(0, 1/d) — but it is
an approximation, and it is an approximation applied to *remex's own arm*.  If
it were expensive it would handicap remex, and since the headline result may
be a remex loss, that has to be measured rather than asserted.

It is not expensive: at d >= 768 the excess MSE is under 0.01%, and even at
d = 100 it is under 0.2%.

Discretization guard: the reference Lloyd here runs on a discretized density,
so it is only trustworthy while the grid resolves the level spacing.  The
script self-checks by running the same discretized solver on a *Gaussian*
density, where the exact answer is known in closed form, and refuses to report
any bit width where that check is off by more than 1%.  Without the guard the
8-bit row silently reports the Beta optimum as twice its true value and the
comparison inverts.

    python3 beta_check.py
"""
from __future__ import annotations

import numpy as np

import grids

BITS = (1, 2, 3, 4, 6, 8)
DIMS = (100, 768, 1024)
GRID_N = 2_000_001
SELF_CHECK_TOL = 0.01


def unit_coord_density(d: int, n: int = GRID_N):
    """Density of one coordinate of a uniformly random unit vector in R^d."""
    x = np.linspace(-1.0, 1.0, n)
    logf = ((d - 3) / 2) * np.log(np.clip(1.0 - x ** 2, 1e-300, None))
    f = np.exp(logf - logf.max())
    return x, f / np.trapezoid(f, x)


def gaussian_density(sigma: float, n: int = GRID_N, span: float = 12.0):
    x = np.linspace(-span * sigma, span * sigma, n)
    f = np.exp(-0.5 * (x / sigma) ** 2)
    return x, f / np.trapezoid(f, x)


def mse_of_levels(x, f, levels):
    bnd = 0.5 * (levels[:-1] + levels[1:])
    idx = np.searchsorted(bnd, x)
    return float(np.trapezoid(f * (x - levels[idx]) ** 2, x))


def lloyd_discrete(x, f, k, iters=2000):
    """Lloyd on a discretized density; returns (levels, mse)."""
    cdf = np.cumsum(f)
    cdf /= cdf[-1]
    levels = np.interp(np.linspace(0, 1, k + 2)[1:-1], cdf, x)
    for _ in range(iters):
        bnd = 0.5 * (levels[:-1] + levels[1:])
        idx = np.searchsorted(bnd, x)
        w = np.bincount(idx, weights=f, minlength=k)
        s = np.bincount(idx, weights=f * x, minlength=k)
        new = np.where(w > 0, s / np.maximum(w, 1e-300), levels)
        if np.allclose(new, levels, rtol=0, atol=1e-16):
            levels = new
            break
        levels = new
    return levels, mse_of_levels(x, f, levels)


def main():
    print("Cost of the Gaussian marginal approximation on the exact-norm arm.")
    print("MSE is reported x d, so the numbers are comparable to the unit-variance")
    print("Lloyd-Max table. 'excess' = how much worse Gaussian-fitted levels are")
    print("than Beta-optimal levels, on the true Beta source.\n")
    print(f"{'d':>6} {'bits':>5} {'beta-optimal':>14} {'gaussian levels':>16} "
          f"{'excess':>9}")
    skipped = []
    for d in DIMS:
        x, f = unit_coord_density(d)
        sigma = 1.0 / np.sqrt(d)
        gx, gf = gaussian_density(sigma)
        for b in BITS:
            # self-check: same solver, Gaussian source, known closed form
            _, mse_disc = lloyd_discrete(gx, gf, 1 << b)
            closed = grids.lloyd_max_1d(b)[1] * sigma ** 2
            if abs(mse_disc - closed) / closed > SELF_CHECK_TOL:
                skipped.append((d, b, mse_disc / closed))
                continue
            _, m_opt = lloyd_discrete(x, f, 1 << b)
            lv = grids.lloyd_max_1d(b)[0].astype(np.float64) * sigma
            m_gauss = mse_of_levels(x, f, lv)
            print(f"{d:>6} {b:>5} {m_opt * d:>14.6f} {m_gauss * d:>16.6f} "
                  f"{(m_gauss / m_opt - 1) * 100:>8.3f}%")
    if skipped:
        print("\nSkipped — discretized reference solver failed its own "
              "Gaussian self-check (grid too coarse for this many levels):")
        for d, b, ratio in skipped:
            print(f"  d={d} bits={b}: solver/closed-form = {ratio:.2f}x")
        print("These rows are a limit of the reference solver, not a finding "
              "about the approximation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
