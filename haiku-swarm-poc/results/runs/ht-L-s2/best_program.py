# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations
from scipy.optimize import minimize

_SQ3 = np.sqrt(3.0)
_N = 11
_TRIPLES = np.array(list(combinations(range(_N), 3)), dtype=int)
_I, _J, _K = _TRIPLES[:, 0], _TRIPLES[:, 1], _TRIPLES[:, 2]
_M = len(_TRIPLES)
_SEED = 20240611

_N_RANDOM_STARTS = 24
_N_SYM_STARTS = 40
_TOP_K = 4
_BH_ITERS = 80
_SA_T0 = 2e-4
_LS_ITERS = 200
_POLISH_ROUNDS = 6


def _signed_cross(P):
    """Twice the signed area of every triple (i, j, k)."""
    xi, yi = P[_I, 0], P[_I, 1]
    xj, yj = P[_J, 0], P[_J, 1]
    xk, yk = P[_K, 0], P[_K, 1]
    return (xj - xi) * (yk - yi) - (xk - xi) * (yj - yi)


def _min_area(P):
    """Smallest triangle area over all triples."""
    return 0.5 * float(np.min(np.abs(_signed_cross(P))))


def _project_inside(P):
    """Map points into the triangle using clipped barycentric coordinates."""
    P = np.asarray(P, dtype=float).copy()
    lC = np.clip(2.0 * P[:, 1] / _SQ3, 0.0, None)
    lB = np.clip(P[:, 0] - 0.5 * lC, 0.0, None)
    lA = np.clip(1.0 - lB - lC, 0.0, None)
    s = lA + lB + lC
    s[s == 0] = 1.0
    lB, lC = lB / s, lC / s
    out = np.empty_like(P)
    out[:, 0] = lB + 0.5 * lC
    out[:, 1] = lC * _SQ3 / 2.0
    return out


def _random_in_triangle(rng, n):
    u = rng.random(n)
    v = rng.random(n)
    m = u + v > 1.0
    u[m] = 1.0 - u[m]
    v[m] = 1.0 - v[m]
    return np.column_stack([u + 0.5 * v, v * _SQ3 / 2.0])


def _symmetric_start(rng):
    """Mirror-symmetric about x = 0.5: 5 mirrored pairs plus one axis point."""
    half = _random_in_triangle(rng, 5)
    axis = np.array([[0.5, rng.random() * _SQ3 / 2.0]])
    pts = np.vstack([half, np.column_stack([1.0 - half[:, 0], half[:, 1]]), axis])
    return _project_inside(pts)


def _cross_jacobian_rows(P, s):
    """Jacobian of s * signed_cross w.r.t. the 22 coordinates."""
    J = np.zeros((_M, 2 * _N))
    xi, yi = P[_I, 0], P[_I, 1]
    xj, yj = P[_J, 0], P[_J, 1]
    xk, yk = P[_K, 0], P[_K, 1]
    rows = np.arange(_M)
    J[rows, 2 * _I] = yj - yk
    J[rows, 2 * _I + 1] = xk - xj
    J[rows, 2 * _J] = yk - yi
    J[rows, 2 * _J + 1] = -(xk - xi)
    J[rows, 2 * _K] = -(yj - yi)
    J[rows, 2 * _K + 1] = xj - xi
    return s[:, None] * J


def _slsqp_refine(P0):
    """Maximize t s.t. s_ijk * cross_ijk >= t and points inside the triangle."""
    P0 = _project_inside(P0)
    s = np.sign(_signed_cross(P0))
    s[s == 0] = 1.0
    z0 = np.concatenate([P0.ravel(), [_min_area(P0) * 2.0]])

    def obj(z):
        return -z[-1]

    def obj_grad(z):
        g = np.zeros_like(z)
        g[-1] = -1.0
        return g

    def cons(z):
        P = z[:-1].reshape(_N, 2)
        D = s * _signed_cross(P) - z[-1]
        x, y = P[:, 0], P[:, 1]
        return np.concatenate([D, y, _SQ3 * x - y, _SQ3 * (1.0 - x) - y])

    def cons_jac(z):
        P = z[:-1].reshape(_N, 2)
        Jt = np.zeros((_M + 3 * _N, 2 * _N + 1))
        Jt[:_M, :2 * _N] = _cross_jacobian_rows(P, s)
        Jt[:_M, -1] = -1.0
        base = _M
        for p in range(_N):
            Jt[base + p, 2 * p + 1] = 1.0
            Jt[base + _N + p, 2 * p] = _SQ3
            Jt[base + _N + p, 2 * p + 1] = -1.0
            Jt[base + 2 * _N + p, 2 * p] = -_SQ3
            Jt[base + 2 * _N + p, 2 * p + 1] = -1.0
        return Jt

    try:
        res = minimize(
            obj, z0, jac=obj_grad, method="SLSQP",
            constraints=[{"type": "ineq", "fun": cons, "jac": cons_jac}],
            options={"maxiter": 400, "ftol": 1e-13},
        )
        P = _project_inside(res.x[:-1].reshape(_N, 2))
        if not np.all(np.isfinite(P)):
            return P0
        return P
    except Exception:
        return P0


def _refine_full(P0, rounds=_POLISH_ROUNDS):
    """Repeat SLSQP with re-derived orientation signs until no improvement."""
    cur = _project_inside(P0)
    best, best_val = cur, _min_area(cur)
    for _ in range(rounds):
        cand = _slsqp_refine(best)
        val = _min_area(cand)
        if val > best_val + 1e-12:
            best, best_val = cand, val
        else:
            break
    return best, best_val


def _smallest_triangle_points(P):
    a = np.abs(_signed_cross(P))
    t = int(np.argmin(a))
    return _TRIPLES[t]


def _perturb(P, rng, scale=1.0):
    """Relocate one point, jitter bottleneck vertices, or jitter a random subset."""
    Q = P.copy()
    r = rng.random()
    if r < 0.2:
        p = int(rng.integers(_N))
        Q[p] = _random_in_triangle(rng, 1)[0]
    elif r < 0.6:
        idx = _smallest_triangle_points(P)
        extra = int(rng.integers(_N))
        idx = np.unique(np.append(idx, extra))
        Q[idx] += rng.normal(scale=0.03 * scale, size=(len(idx), 2))
    else:
        k = int(rng.integers(2, 5))
        idx = rng.choice(_N, size=k, replace=False)
        Q[idx] += rng.normal(scale=0.04 * scale, size=(k, 2))
    return _project_inside(Q)


def _anneal(P0, v0, rng, iters):
    """Basin hopping with near-greedy annealing acceptance."""
    cur_P, cur_val = P0, v0
    best_P, best_val = P0, v0
    for it in range(iters):
        T = _SA_T0 * (1.0 - it / max(iters, 1)) + 1e-12
        try:
            cand, val = _refine_full(_perturb(cur_P, rng))
        except Exception:
            continue
        if val >= cur_val or rng.random() < np.exp((val - cur_val) / T):
            cur_P, cur_val = cand, val
            if val > best_val + 1e-12:
                best_P, best_val = cand, val
    return best_P, best_val


def _local_search(P0, v0, rng, iters):
    """Greedy search around the incumbent using small-scale perturbations only."""
    best_P, best_val = P0, v0
    for _ in range(iters):
        Q = best_P.copy()
        k = int(rng.integers(1, 4))
        idx = rng.choice(_N, size=k, replace=False)
        Q[idx] += rng.normal(scale=rng.choice([0.005, 0.01]), size=(k, 2))
        Q = _project_inside(Q)
        try:
            cand, val = _refine_full(Q)
        except Exception:
            continue
        if val > best_val + 1e-12:
            best_P, best_val = cand, val
    return best_P, best_val


def heilbronn_triangle11() -> np.ndarray:
    """
    Construct 11 points on or inside the equilateral triangle with vertices
    (0,0), (1,0), (0.5, sqrt(3)/2), maximizing the minimum triangle area.

    Deterministic pipeline: mirror-symmetric-biased multistart with SLSQP
    maximin polish, basin hopping concentrated on the top few layouts, then a
    final small-scale greedy local search on the incumbent.
    """
    rng = np.random.default_rng(_SEED)
    pool = []

    for _ in range(_N_RANDOM_STARTS):
        try:
            P, v = _refine_full(_random_in_triangle(rng, _N))
            pool.append((v, P))
        except Exception:
            continue

    for _ in range(_N_SYM_STARTS):
        try:
            P, v = _refine_full(_symmetric_start(rng))
            pool.append((v, P))
        except Exception:
            continue

    pool.sort(key=lambda t: -t[0])

    best_P, best_val = None, -np.inf
    if pool:
        best_val, best_P = pool[0]

    for v0, P0 in pool[:_TOP_K]:
        try:
            cand, val = _anneal(P0, v0, rng, _BH_ITERS)
        except Exception:
            continue
        if val > best_val + 1e-12:
            best_P, best_val = cand, val

    if best_P is not None and best_val > 0.0:
        try:
            cand, val = _local_search(best_P, best_val, rng, _LS_ITERS)
            if val > best_val + 1e-12:
                best_P, best_val = cand, val
        except Exception:
            pass

    if best_P is None or best_val <= 0.0:
        best_P = _random_in_triangle(np.random.default_rng(_SEED), _N)

    return _project_inside(best_P).astype(float)


# EVOLVE-BLOCK-END
