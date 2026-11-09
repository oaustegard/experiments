# EVOLVE-BLOCK-START
import time
from itertools import combinations

import numpy as np

try:
    from scipy.optimize import minimize as _sp_minimize
except Exception:  # pragma: no cover
    _sp_minimize = None

_SQRT3 = float(np.sqrt(3.0))
_TRI_AREA = _SQRT3 / 4.0
_VERTS = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, _SQRT3 / 2.0]])
_N = 11
_IDX = np.array(list(combinations(range(_N), 3)))


def _signed_areas(p):
    """Signed triangle areas (normalized by the big triangle's area) for all triples."""
    i, j, k = _IDX[:, 0], _IDX[:, 1], _IDX[:, 2]
    cr = (p[j, 0] - p[i, 0]) * (p[k, 1] - p[i, 1]) - (p[k, 0] - p[i, 0]) * (p[j, 1] - p[i, 1])
    return 0.5 * cr / _TRI_AREA


def _min_area(p):
    return float(np.min(np.abs(_signed_areas(p))))


def _project(p):
    """Clip points into the triangle through barycentric coordinates."""
    c = 2.0 * p[:, 1] / _SQRT3
    b = p[:, 0] - 0.5 * c
    a = 1.0 - b - c
    w = np.clip(np.stack([a, b, c], axis=1), 0.0, None)
    w /= np.maximum(w.sum(axis=1, keepdims=True), 1e-300)
    return w @ _VERTS


def _boundary_system(n):
    """Linear inequalities G z + h >= 0 keeping points inside the triangle (z=[x0,y0,...,t])."""
    G = np.zeros((3 * n, 2 * n + 1))
    h = np.zeros(3 * n)
    for q in range(n):
        G[3 * q, 2 * q + 1] = 1.0                      # y >= 0
        G[3 * q + 1, 2 * q] = _SQRT3                   # sqrt3 x - y >= 0
        G[3 * q + 1, 2 * q + 1] = -1.0
        G[3 * q + 2, 2 * q] = -_SQRT3                  # sqrt3 (1 - x) - y >= 0
        G[3 * q + 2, 2 * q + 1] = -1.0
        h[3 * q + 2] = _SQRT3
    return G, h


_G, _H = _boundary_system(_N)


def _slsqp_polish(p0, maxiter=300):
    """Maximize t s.t. s*area >= t with fixed orientation signs (epigraph form)."""
    n = _N
    m = len(_IDX)
    i, j, k = _IDX[:, 0], _IDX[:, 1], _IDX[:, 2]
    s = np.sign(_signed_areas(p0))
    s[s == 0] = 1.0
    c = 0.5 / _TRI_AREA
    rows = np.arange(m)

    def cons(z):
        p = z[:-1].reshape(n, 2)
        return np.concatenate([s * _signed_areas(p) - z[-1], _G @ z + _H])

    def cons_jac(z):
        p = z[:-1].reshape(n, 2)
        J = np.zeros((m + 3 * n, 2 * n + 1))
        xi, yi = p[i, 0], p[i, 1]
        xj, yj = p[j, 0], p[j, 1]
        xk, yk = p[k, 0], p[k, 1]
        J[rows, 2 * i] = s * c * (yj - yk)
        J[rows, 2 * i + 1] = s * c * (xk - xj)
        J[rows, 2 * j] = s * c * (yk - yi)
        J[rows, 2 * j + 1] = s * c * (xi - xk)
        J[rows, 2 * k] = s * c * (yi - yj)
        J[rows, 2 * k + 1] = s * c * (xj - xi)
        J[:m, -1] = -1.0
        J[m:, :] = _G
        return J

    t0 = float(np.min(s * _signed_areas(p0)))
    z0 = np.concatenate([p0.ravel(), [t0]])
    grad = np.zeros(2 * n + 1)
    grad[-1] = -1.0
    res = _sp_minimize(
        lambda z: -z[-1],
        z0,
        jac=lambda z: grad,
        constraints=[{"type": "ineq", "fun": cons, "jac": cons_jac}],
        method="SLSQP",
        options={"maxiter": maxiter, "ftol": 1e-14},
    )
    return _project(res.x[:-1].reshape(n, 2))


def _hill_climb(p, rng, iters=20000):
    """Fallback stochastic local search (used only if SciPy is unavailable)."""
    best = _min_area(p)
    step = 0.05
    for it in range(iters):
        q = p.copy()
        r = rng.integers(_N)
        q[r] += rng.normal(0.0, step, 2)
        q = _project(q)
        v = _min_area(q)
        if v > best:
            p, best = q, v
        if it % 2000 == 1999:
            step *= 0.6
    return p


def _sym_samples(rng, m):
    """Mirror-symmetric (x -> 1-x) configurations: 5 mirrored pairs + 1 axis point."""
    half = np.array([[0.0, 0.0], [0.5, 0.0], [0.5, _SQRT3 / 2.0]])
    w = rng.dirichlet([0.7, 0.7, 0.7], size=(m, 5))
    left = w @ half
    right = left.copy()
    right[..., 0] = 1.0 - right[..., 0]
    ax = np.stack([np.full(m, 0.5), rng.random(m) * _SQRT3 / 2.0], axis=1)[:, None, :]
    return np.concatenate([left, right, ax], axis=1)


def heilbronn_triangle11() -> np.ndarray:
    """
    Construct an arrangement of n points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 11.

    Returns:
        points: np.ndarray of shape (11,2) with the x,y coordinates of the points.
    """
    n = _N
    rng = np.random.default_rng(12345)
    deadline = time.time() + 20.0

    # ---- Stage 1: cheap random screening (boundary-biased barycentric samples) ----
    stage2_deadline = time.time() + 7.0   # reserve the rest of the budget for basin hopping
    m = 20000
    w = rng.dirichlet([0.6, 0.6, 0.6], size=(m, n))   # (m, n, 3)
    samples = np.concatenate([w @ _VERTS, _sym_samples(rng, m)], axis=0)  # (2m, n, 2)
    i, j, k = _IDX[:, 0], _IDX[:, 1], _IDX[:, 2]
    cr = (samples[:, j, 0] - samples[:, i, 0]) * (samples[:, k, 1] - samples[:, i, 1]) \
        - (samples[:, k, 0] - samples[:, i, 0]) * (samples[:, j, 1] - samples[:, i, 1])
    scores = np.min(0.5 * np.abs(cr) / _TRI_AREA, axis=1)
    # interleave the best generic and best symmetric starts
    ord_a = np.argsort(-scores[:m])[:100]
    ord_b = m + np.argsort(-scores[m:])[:100]
    order = np.empty(200, dtype=int)
    order[0::2] = ord_a
    order[1::2] = ord_b

    best_pts = samples[order[0]].copy()
    best_val = _min_area(best_pts)

    # ---- Stage 2: SLSQP polishing of the most promising starts ----
    n_starts = 200
    for idx in order[:n_starts]:
        if time.time() > stage2_deadline:
            break
        start = samples[idx]
        try:
            if _sp_minimize is not None:
                cand = _slsqp_polish(start)
            else:
                cand = _hill_climb(start.copy(), rng)
        except Exception:
            continue
        if not np.all(np.isfinite(cand)):
            continue
        val = _min_area(cand)
        if val > best_val:
            best_val, best_pts = val, cand.copy()

    # ---- Stage 3: basin hopping around the incumbent (re-derives orientation signs) ----
    if _sp_minimize is not None:
        try:
            cand = _slsqp_polish(best_pts, maxiter=500)
            val = _min_area(cand)
            if val > best_val:
                best_val, best_pts = val, cand
        except Exception:
            pass

        cur_pts, cur_val = best_pts.copy(), best_val
        sigmas = (0.01, 0.025, 0.05, 0.1)
        stall = 0
        for it in range(1500):
            if time.time() > deadline:
                break
            trial = cur_pts.copy()
            if rng.random() < 0.3:
                # relocate one or two points to random positions in the triangle
                for r in rng.choice(n, size=int(rng.integers(1, 3)), replace=False):
                    trial[r] = rng.dirichlet([1.0, 1.0, 1.0]) @ _VERTS
            else:
                trial += rng.normal(0.0, sigmas[it % len(sigmas)], trial.shape)
            trial = _project(trial)
            try:
                cand = _slsqp_polish(trial, maxiter=200)
            except Exception:
                continue
            if not np.all(np.isfinite(cand)):
                continue
            val = _min_area(cand)
            if val > cur_val * 0.97:
                cur_pts, cur_val = cand.copy(), val
            if val > best_val + 1e-12:
                best_val, best_pts = val, cand.copy()
                stall = 0
            else:
                stall += 1
            if stall >= 60:  # restart the walk from the global best
                cur_pts, cur_val = best_pts.copy(), best_val
                stall = 0

        # final high-accuracy polish of the best configuration
        try:
            cand = _slsqp_polish(best_pts, maxiter=500)
            val = _min_area(cand)
            if val > best_val:
                best_val, best_pts = val, cand
        except Exception:
            pass

    points = _project(np.asarray(best_pts, dtype=float))
    return points


# EVOLVE-BLOCK-END
