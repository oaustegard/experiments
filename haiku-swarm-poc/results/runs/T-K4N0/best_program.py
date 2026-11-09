# EVOLVE-BLOCK-START
"""Multi-start constructor for 26 circles in a unit square.

Explicit staggered-row seeds plus random seeds -> joint SLSQP optimisation of
centers and radii -> exact LP radii -> strict repair so the result is always
valid. A time-bounded basin-hopping search with Metropolis acceptance refines
the best arrangement, with moves that relocate the smallest circle, swap the
two smallest, or jitter a few centers.
"""
import math
import time

import numpy as np
from scipy.optimize import linprog, minimize

N = 26
_IU = np.triu_indices(N, k=1)
PI, PJ = _IU[0], _IU[1]
NP = PI.size

ROW_CONFIGS = [
    (5, 5, 6, 5, 5),
    (5, 6, 5, 6, 4),
    (6, 5, 6, 5, 4),
    (4, 6, 5, 6, 5),
    (5, 5, 5, 5, 6),
    (4, 5, 6, 5, 6),
    (6, 4, 6, 4, 6),
    (4, 5, 4, 5, 4, 4),
    (5, 6, 5, 6, 4),
    (6, 5, 5, 5, 5),
    (4, 6, 6, 6, 4),
    (5, 4, 6, 4, 4, 3),
    (6, 5, 4, 5, 6),
    (5, 6, 4, 6, 5),
]

TIME_BUDGET = 90.0  # seconds for the whole constructor


def _row_centers(counts):
    """Evenly spaced rows; row counts must sum to N."""
    m = len(counts)
    pts = []
    for k, cnt in enumerate(counts):
        y = (k + 0.5) / m
        for j in range(cnt):
            pts.append([(j + 0.5) / cnt, y])
    c = np.array(pts, dtype=float)
    assert c.shape == (N, 2)
    return c


def _wall_ub(c):
    return np.maximum(np.minimum(np.minimum(c[:, 0], c[:, 1]),
                                 np.minimum(1 - c[:, 0], 1 - c[:, 1])), 0.0)


def _pair_d(c):
    return np.sqrt(np.sum((c[PI] - c[PJ]) ** 2, axis=1))


def _lp_radii(c):
    """Maximal radii for fixed centers (LP). Returns None on failure."""
    n = c.shape[0]
    A = np.zeros((NP + n, n))
    A[np.arange(NP), PI] = 1.0
    A[np.arange(NP), PJ] = 1.0
    A[NP + np.arange(n), np.arange(n)] = 1.0
    b = np.concatenate([_pair_d(c), _wall_ub(c)])
    try:
        res = linprog(-np.ones(n), A_ub=A, b_ub=b,
                      bounds=[(0.0, None)] * n, method="highs")
    except Exception:
        return None
    if res.status != 0 or res.x is None:
        return None
    return np.maximum(res.x, 0.0)


def _repair(c, r):
    """Force exact validity. Radii only ever decrease, so one pass suffices."""
    r = np.maximum(np.asarray(r, dtype=float), 0.0)
    r = np.minimum(r, _wall_ub(c))
    d = _pair_d(c)
    for k in range(NP):
        i, j = PI[k], PJ[k]
        s = r[i] + r[j]
        if s > d[k] and s > 0.0:
            f = d[k] / s
            r[i] *= f
            r[j] *= f
    return r * (1.0 - 1e-9)


def _slsqp(c0, r0, maxiter=250):
    """Joint optimisation of centers and radii. Returns raw (centers, radii)."""
    n = N
    c0 = np.asarray(c0, dtype=float)
    r0 = np.asarray(r0, dtype=float)
    z0 = np.concatenate([c0.ravel(), r0])

    def unpack(z):
        return z[:2 * n].reshape(n, 2), z[2 * n:]

    def cons(z):
        P, R = unpack(z)
        dx = P[PI, 0] - P[PJ, 0]
        dy = P[PI, 1] - P[PJ, 1]
        s = R[PI] + R[PJ]
        gp = dx * dx + dy * dy - s * s
        gw = np.concatenate([P[:, 0] - R, 1 - P[:, 0] - R,
                             P[:, 1] - R, 1 - P[:, 1] - R])
        return np.concatenate([gp, gw])

    def jac(z):
        P, R = unpack(z)
        dx = P[PI, 0] - P[PJ, 0]
        dy = P[PI, 1] - P[PJ, 1]
        s = R[PI] + R[PJ]
        M = np.zeros((NP + 4 * n, 3 * n))
        k = np.arange(NP)
        M[k, 2 * PI] = 2.0 * dx
        M[k, 2 * PJ] = -2.0 * dx
        M[k, 2 * PI + 1] = 2.0 * dy
        M[k, 2 * PJ + 1] = -2.0 * dy
        M[k, 2 * n + PI] = -2.0 * s
        M[k, 2 * n + PJ] = -2.0 * s
        i = np.arange(n)
        b = NP
        M[b + i, 2 * i] = 1.0
        M[b + i, 2 * n + i] = -1.0
        b += n
        M[b + i, 2 * i] = -1.0
        M[b + i, 2 * n + i] = -1.0
        b += n
        M[b + i, 2 * i + 1] = 1.0
        M[b + i, 2 * n + i] = -1.0
        b += n
        M[b + i, 2 * i + 1] = -1.0
        M[b + i, 2 * n + i] = -1.0
        return M

    def fun(z):
        return -np.sum(z[2 * n:])

    def gfun(z):
        g = np.zeros_like(z)
        g[2 * n:] = -1.0
        return g

    bounds = [(0.0, 1.0)] * (2 * n) + [(0.0, 0.5)] * n
    try:
        res = minimize(fun, z0, jac=gfun, method="SLSQP", bounds=bounds,
                       constraints=[{"type": "ineq", "fun": cons, "jac": jac}],
                       options={"maxiter": maxiter, "ftol": 1e-12})
        z = res.x
        if not np.all(np.isfinite(z)):
            raise ValueError("non-finite SLSQP result")
    except Exception:
        return c0.copy(), r0.copy()
    P, R = unpack(z)
    return P.copy(), np.maximum(R, 0.0)


def _finalize(c, r_raw):
    """Clip centers, take the better of exact LP radii and repaired SLSQP radii."""
    c = np.clip(np.asarray(c, dtype=float), 0.0, 1.0)
    cands = []
    r_lp = _lp_radii(c)
    if r_lp is not None:
        cands.append(_repair(c, r_lp))
    cands.append(_repair(c, r_raw))
    r = max(cands, key=lambda v: float(np.sum(v)))
    return c, r, float(np.sum(r))


def _run(c0, r0, maxiter=250):
    c, r_raw = _slsqp(np.clip(c0, 0.0, 1.0), r0, maxiter=maxiter)
    return _finalize(c, r_raw)


def construct_packing():
    t0 = time.time()
    rng = np.random.default_rng(2025)

    starts = [_row_centers(cfg) for cfg in ROW_CONFIGS]
    for t in range(24):
        base = starts[t % len(ROW_CONFIGS)]
        starts.append(np.clip(base + rng.normal(0.0, 0.02, base.shape), 0.02, 0.98))
    for _ in range(24):
        starts.append(rng.uniform(0.05, 0.95, size=(N, 2)))

    best_c, best_r, best_s = None, None, -1.0
    for c0 in starts:
        r0 = _lp_radii(c0)
        if r0 is None:
            r0 = np.full(N, 0.05)
        c, r, s = _run(c0, r0)
        if s > best_s:
            best_c, best_r, best_s = c, r, s

    # Basin hopping with Metropolis acceptance on the current state.
    cur_c, cur_r, cur_s = best_c, best_r, best_s
    temp0 = 0.01
    it = 0
    while time.time() - t0 < TIME_BUDGET and it < 400:
        frac = min(1.0, (time.time() - t0) / TIME_BUDGET)
        temp = temp0 * (1.0 - frac) + 1e-5
        cand = cur_c.copy()
        u = rng.random()
        if u < 0.2:
            idx = int(np.argmin(cur_r))
            cand[idx] = rng.uniform(0.05, 0.95, size=2)
        elif u < 0.3:
            order = np.argsort(cur_r)
            a, b = int(order[0]), int(order[1])
            cand[[a, b]] = cand[[b, a]]
            cand[a] += rng.normal(0.0, 0.01, size=2)
            cand[b] += rng.normal(0.0, 0.01, size=2)
        else:
            k = int(rng.integers(1, 4))
            idx = rng.choice(N, size=k, replace=False)
            step = 0.03 * (1.0 - frac) + 0.005
            cand[idx] += rng.normal(0.0, step, size=(k, 2))
        cand = np.clip(cand, 0.02, 0.98)
        r_init = _lp_radii(cand)
        if r_init is None:
            r_init = cur_r
        c, r, s = _run(cand, r_init)
        accept = s > cur_s
        if not accept and s > -1.0:
            delta = s - cur_s
            accept = rng.random() < math.exp(min(0.0, delta / temp))
        if accept:
            cur_c, cur_r, cur_s = c, r, s
        if s > best_s + 1e-10:
            best_c, best_r, best_s = c, r, s
        it += 1

    return best_c, best_r, float(np.sum(best_r))


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_packing():
    """Run the circle packing constructor for n=26"""
    centers, radii, sum_radii = construct_packing()
    return centers, radii, sum_radii


if __name__ == "__main__":
    centers, radii, sum_radii = run_packing()
    print(f"Sum of radii: {sum_radii}")
