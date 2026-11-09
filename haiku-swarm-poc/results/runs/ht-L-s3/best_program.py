# EVOLVE-BLOCK-START
import itertools

import numpy as np
from scipy.optimize import minimize

_N = 11
_TRI = np.array(list(itertools.combinations(range(_N), 3)))
_I, _J, _K = _TRI[:, 0], _TRI[:, 1], _TRI[:, 2]
_T = len(_TRI)
_ROWS = np.arange(_T)
_IDX = np.arange(_N)
_SQ3 = np.sqrt(3.0)
_MARGIN = 1e-9


def _fold(p):
    """Map points into the reference triangle {u>=0, v>=0, u+v<=1}."""
    p = np.array(p, dtype=float).reshape(-1, 2)
    p = np.clip(p, _MARGIN, None)
    s = p.sum(axis=1)
    m = s > 1.0 - _MARGIN
    p[m] = p[m] / s[m][:, None] * (1.0 - _MARGIN)
    return p


def _dets(p):
    xa, ya = p[_I, 0], p[_I, 1]
    xb, yb = p[_J, 0], p[_J, 1]
    xc, yc = p[_K, 0], p[_K, 1]
    return (xb - xa) * (yc - ya) - (yb - ya) * (xc - xa)


def _dets_jac(p):
    xa, ya = p[_I, 0], p[_I, 1]
    xb, yb = p[_J, 0], p[_J, 1]
    xc, yc = p[_K, 0], p[_K, 1]
    J = np.zeros((_T, 2 * _N))
    J[_ROWS, 2 * _I] = yb - yc
    J[_ROWS, 2 * _I + 1] = xc - xb
    J[_ROWS, 2 * _J] = yc - ya
    J[_ROWS, 2 * _J + 1] = xa - xc
    J[_ROWS, 2 * _K] = ya - yb
    J[_ROWS, 2 * _K + 1] = xb - xa
    return J


def _min_area(p):
    return float(np.min(np.abs(_dets(p))) / 2.0)


def _random_points(rng, n=_N):
    u = rng.random((n, 2))
    m = u.sum(axis=1) > 1.0
    u[m] = 1.0 - u[m]
    return u


def _sym_full(base, v_axis):
    """Mirror about the altitude x=0.5, i.e. (u,v) -> (1-u-v, v).
    Five base points, their mirrors, and one on-axis point u=(1-v)/2."""
    mirrored = np.column_stack([1.0 - base[:, 0] - base[:, 1], base[:, 1]])
    axis = np.array([[0.5 * (1.0 - v_axis), v_axis]])
    return np.vstack([base, mirrored, axis])


def _hill(p, rng, iters, patience):
    p = _fold(p)
    best = _min_area(p)
    sigma = 0.25
    since = 0
    for _ in range(iters):
        q = p.copy()
        i = rng.integers(_N)
        if rng.random() < 0.7:
            q[i] += rng.normal(0.0, sigma, 2)
        else:
            q[i] = _random_points(rng, 1)[0]
        q = _fold(q)
        val = _min_area(q)
        if val > best * (1.0 + 1e-9):
            since = 0
        else:
            since += 1
        if val >= best:
            p, best = q, val
        sigma = max(0.005, sigma * 0.998)
        if since > patience:
            break
    return p, best


def _hill_sym(rng, iters):
    """Hill climb in the reduced mirror-symmetric space (5 pairs + axis param)."""
    base = _random_points(rng, 5)
    v_ax = float(rng.random())
    p = _fold(_sym_full(base, v_ax))
    best = _min_area(p)
    sigma = 0.25
    for _ in range(iters):
        nb, nv = base.copy(), v_ax
        if rng.random() < 0.8:
            i = rng.integers(5)
            nb[i] += rng.normal(0.0, sigma, 2)
        else:
            nv = float(np.clip(v_ax + rng.normal(0.0, 0.5 * sigma), 0.0, 1.0))
        nb = _fold(nb)
        q = _fold(_sym_full(nb, nv))
        val = _min_area(q)
        if val >= best:
            base, v_ax, p, best = nb, nv, q, val
        sigma = max(0.005, sigma * 0.998)
    return p, best


def _polish_once(p):
    """SLSQP epigraph: maximize t s.t. s_k * det_k >= t, signs frozen at p."""
    p = _fold(p)
    base = _min_area(p)
    signs = np.where(_dets(p) < 0.0, -1.0, 1.0)

    def cons_area(z):
        return signs * _dets(z[:-1].reshape(_N, 2)) - z[-1]

    def cons_area_jac(z):
        J = _dets_jac(z[:-1].reshape(_N, 2)) * signs[:, None]
        return np.hstack([J, -np.ones((_T, 1))])

    def cons_lin(z):
        uv = z[:-1].reshape(_N, 2)
        return np.concatenate([uv[:, 0], uv[:, 1], 1.0 - uv.sum(axis=1)])

    A = np.zeros((3 * _N, 2 * _N + 1))
    A[_IDX, 2 * _IDX] = 1.0
    A[_N + _IDX, 2 * _IDX + 1] = 1.0
    A[2 * _N + _IDX, 2 * _IDX] = -1.0
    A[2 * _N + _IDX, 2 * _IDX + 1] = -1.0

    z0 = np.concatenate([p.ravel(), [base]])
    bounds = [(0.0, 1.0)] * (2 * _N + 1)
    try:
        res = minimize(
            lambda z: -z[-1],
            z0,
            jac=lambda z: np.concatenate([np.zeros(2 * _N), [-1.0]]),
            method="SLSQP",
            bounds=bounds,
            constraints=[
                {"type": "ineq", "fun": cons_area, "jac": cons_area_jac},
                {"type": "ineq", "fun": cons_lin, "jac": lambda z: A},
            ],
            options={"maxiter": 400, "ftol": 1e-12},
        )
        cand = _fold(res.x[:-1].reshape(_N, 2))
        if _min_area(cand) > base:
            return cand
    except Exception:
        pass
    return p


def _polish(p, rounds=3):
    """Repeat the epigraph polish, re-deriving orientation signs each round."""
    best = _fold(p)
    bv = _min_area(best)
    cur = best
    for _ in range(rounds):
        cand = _polish_once(cur)
        v = _min_area(cand)
        if v > bv * (1.0 + 1e-10):
            best, bv, cur = cand, v, cand
        else:
            break
    return best, bv


def _basin_hop(p, v, rng, iters=30):
    """Perturb a few points of the incumbent, re-polish, keep strict gains."""
    best_p, best_v = _fold(p), v
    for it in range(iters):
        q = best_p.copy()
        k = int(rng.integers(1, 4))
        idx = rng.choice(_N, size=k, replace=False)
        scale = 0.12 * (0.85 ** (it // 5))
        q[idx] += rng.normal(0.0, scale, (k, 2))
        cand, cv = _polish(_fold(q))
        if cv > best_v * (1.0 + 1e-10):
            best_p, best_v = cand, cv
    return best_p, best_v


def _to_equilateral(p):
    u = p[:, 0]
    v = p[:, 1]
    x = u + 0.5 * v
    y = v * _SQ3 / 2.0
    pts = np.column_stack([x, y])
    pts[:, 1] = np.clip(pts[:, 1], 0.0, _SQ3 / 2.0)
    return pts


def heilbronn_triangle11() -> np.ndarray:
    """
    Construct 11 points in the equilateral triangle (0,0),(1,0),(0.5,sqrt(3)/2)
    maximizing the minimum triangle area. Deterministic (fixed seed).

    Search runs in affine coordinates (u, v) on {u,v>=0, u+v<=1}; area ratios
    are affine-invariant, so the score is unchanged by the map.
    """
    rng = np.random.default_rng(20240615)

    best_p, best_v = None, -1.0
    try:
        # Breadth: random hill climbs plus symmetric-space hill climbs.
        starts = []
        for k in range(32):
            if k % 8 == 7:
                p, v = _hill_sym(rng, 1500)
            else:
                p, v = _hill(_random_points(rng), rng, 1500, 500)
            starts.append((v, p))

        # Polish every start (unconstrained, so symmetry can break).
        polished = []
        for _, p in starts:
            cand, cv = _polish(p)
            polished.append((cv, cand))
        polished.sort(key=lambda t: -t[0])

        # Keep a handful of distinct basins.
        pop = []
        for cv, cand in polished:
            if all(abs(cv - pv) > 1e-9 for pv, _ in pop):
                pop.append((cv, cand))
            if len(pop) == 4:
                break

        # Basin-hop each distinct incumbent; keep the best overall.
        hopped = [(polished[0][0], polished[0][1])]
        for cv, cand in pop:
            hp, hv = _basin_hop(cand, cv, rng, 30)
            hopped.append((hv, hp))
        hopped.sort(key=lambda t: -t[0])
        best_v, best_p = hopped[0][0], hopped[0][1]
    except Exception:
        pass

    if best_p is None:
        best_p = _random_points(np.random.default_rng(0))

    points = _to_equilateral(_fold(best_p))
    return np.asarray(points, dtype=float).reshape(_N, 2)


# EVOLVE-BLOCK-END
