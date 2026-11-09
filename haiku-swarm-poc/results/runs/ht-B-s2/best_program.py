import itertools

import numpy as np
from scipy.optimize import minimize

# EVOLVE-BLOCK-START

_N = 11
_SQ3 = np.sqrt(3.0)
_H = 0.5 * _SQ3
_CENT = np.array([0.5, _SQ3 / 6.0])
_TRIPLES = np.array(list(itertools.combinations(range(_N), 3)), dtype=int)
_TI, _TJ, _TK = _TRIPLES[:, 0], _TRIPLES[:, 1], _TRIPLES[:, 2]
_M = len(_TRIPLES)
_ROWS = np.arange(_M)

# Linear inside-triangle constraints in full coordinates w = (x0, y0, x1, y1, ...):
#   y >= 0,  sqrt3*x - y >= 0,  sqrt3*(1 - x) - y >= 0
_GIN = np.zeros((3 * _N, 2 * _N))
_BIN = np.zeros(3 * _N)
for _p in range(_N):
    _GIN[_p, 2 * _p + 1] = 1.0
    _GIN[_N + _p, 2 * _p] = _SQ3
    _GIN[_N + _p, 2 * _p + 1] = -1.0
    _GIN[2 * _N + _p, 2 * _p] = -_SQ3
    _GIN[2 * _N + _p, 2 * _p + 1] = -1.0
    _BIN[2 * _N + _p] = _SQ3

# Mirror symmetry about x = 0.5: points p and 5+p are mirror pairs, point 10 on the axis.
_NP = 5
_NRED = 2 * _NP + 1
_A_SYM = np.zeros((2 * _N, _NRED))
_C_SYM = np.zeros(2 * _N)
for _p in range(_NP):
    _a, _b = _p, _NP + _p
    _A_SYM[2 * _a, 2 * _p] = 1.0
    _A_SYM[2 * _a + 1, 2 * _p + 1] = 1.0
    _A_SYM[2 * _b, 2 * _p] = -1.0
    _C_SYM[2 * _b] = 1.0
    _A_SYM[2 * _b + 1, 2 * _p + 1] = 1.0
_C_SYM[2 * 10] = 0.5
_A_SYM[2 * 10 + 1, 2 * _NP] = 1.0

_A_ID = np.eye(2 * _N)
_C_ZERO = np.zeros(2 * _N)


def _from_bary(L):
    """Barycentric (l1, l2, l3) -> Cartesian; vertices (0,0), (1,0), (0.5, sqrt3/2)."""
    x = L[:, 1] + 0.5 * L[:, 2]
    y = _H * L[:, 2]
    return np.stack([x, y], axis=1)


def _random_points(rng):
    return _from_bary(rng.dirichlet(np.ones(3), size=_N))


def _boundary_points(rng, nb):
    """Random layout with the first nb points forced onto a random triangle edge."""
    P = _random_points(rng)
    for p in range(nb):
        t = rng.random()
        e = int(rng.integers(3))
        if e == 0:
            L = [1.0 - t, t, 0.0]
        elif e == 1:
            L = [1.0 - t, 0.0, t]
        else:
            L = [0.0, 1.0 - t, t]
        P[p] = _from_bary(np.array([L]))[0]
    return P


def _project(P):
    """Map points into the triangle by clipping barycentric coordinates."""
    P = np.asarray(P, dtype=float).reshape(_N, 2)
    l3 = P[:, 1] / _H
    l2 = P[:, 0] - 0.5 * l3
    l1 = 1.0 - l2 - l3
    L = np.clip(np.stack([l1, l2, l3], axis=1), 0.0, None)
    L /= L.sum(axis=1, keepdims=True)
    return _from_bary(L)


def _is_inside(P, tol=1e-9):
    P = np.asarray(P, dtype=float)
    if not np.all(np.isfinite(P)):
        return False
    l3 = P[:, 1] / _H
    l2 = P[:, 0] - 0.5 * l3
    l1 = 1.0 - l2 - l3
    return bool(np.all(np.stack([l1, l2, l3]) >= -tol))


def _cross(P):
    i, j, k = _TI, _TJ, _TK
    return (P[j, 0] - P[i, 0]) * (P[k, 1] - P[i, 1]) - (P[k, 0] - P[i, 0]) * (P[j, 1] - P[i, 1])


def _min_area(P):
    a = 0.5 * float(np.min(np.abs(_cross(P))))
    return a if np.isfinite(a) else -np.inf


def _solve(v0, A, c, maxiter=300):
    """
    Max-min SLSQP over reduced variables v, full coordinates w = A v + c.
    Maximize t s.t. s_T * cross_T(w) / 2 >= t for all triples T, plus inside-triangle
    constraints. Sign pattern is taken from the start point. Falls back to v0 on failure.
    """
    v0 = np.asarray(v0, dtype=float)
    nv = A.shape[1]
    w0 = A @ v0 + c
    cr0 = _cross(w0.reshape(_N, 2))
    s = np.where(cr0 >= 0.0, 1.0, -1.0)
    h = 0.5 * s
    t0 = 0.5 * float(np.min(np.abs(cr0)))
    z0 = np.concatenate([v0, [t0]])
    i, j, k = _TI, _TJ, _TK

    def fobj(z):
        return -z[-1]

    def fgrad(z):
        g = np.zeros(nv + 1)
        g[-1] = -1.0
        return g

    def cons(z):
        w = A @ z[:-1] + c
        P = w.reshape(_N, 2)
        return np.concatenate([h * _cross(P) - z[-1], _GIN @ w + _BIN])

    def jac(z):
        w = A @ z[:-1] + c
        P = w.reshape(_N, 2)
        xi, yi = P[i, 0], P[i, 1]
        xj, yj = P[j, 0], P[j, 1]
        xk, yk = P[k, 0], P[k, 1]
        Jc = np.zeros((_M, 2 * _N))
        Jc[_ROWS, 2 * i] = yj - yk
        Jc[_ROWS, 2 * i + 1] = xk - xj
        Jc[_ROWS, 2 * j] = yk - yi
        Jc[_ROWS, 2 * j + 1] = xi - xk
        Jc[_ROWS, 2 * k] = yi - yj
        Jc[_ROWS, 2 * k + 1] = xj - xi
        Jw = np.vstack([Jc * h[:, None], _GIN])
        J = np.zeros((_M + 3 * _N, nv + 1))
        J[:, :nv] = Jw @ A
        J[:_M, -1] = -1.0
        return J

    try:
        res = minimize(
            fobj, z0, jac=fgrad, method="SLSQP",
            constraints=[{"type": "ineq", "fun": cons, "jac": jac}],
            options={"maxiter": maxiter, "ftol": 1e-13},
        )
        v = np.asarray(res.x[:-1], dtype=float)
        if not np.all(np.isfinite(v)):
            return v0
        return v
    except Exception:
        return v0


def _polish(P0, rounds=2):
    """Unconstrained max-min SLSQP; sign pattern re-derived each round."""
    best = _project(P0)
    best_a = _min_area(best)
    for _ in range(rounds):
        v = _solve(best.ravel(), _A_ID, _C_ZERO, maxiter=500)
        cand = _project(v.reshape(_N, 2))
        a = _min_area(cand)
        if a > best_a + 1e-13:
            best, best_a = cand, a
        else:
            break
    return best


def _sym_vec(P):
    return np.concatenate([np.asarray(P, dtype=float)[:_NP].ravel(), [P[10, 1]]])


def _sym_map(v):
    return _project((_A_SYM @ np.asarray(v, dtype=float) + _C_SYM).reshape(_N, 2))


def _sym_solve(v0):
    return _sym_map(_solve(v0, _A_SYM, _C_SYM))


def _sym_start(rng, boundary):
    P = _boundary_points(rng, 3) if boundary else _random_points(rng)
    return _sym_vec(P)


def _sym_hop(P0, rng, iters):
    """Annealed basin hopping in mirror-symmetric space; returns the best symmetric layout."""
    cur_P = _sym_map(_sym_vec(P0))
    cur_a = _min_area(cur_P)
    best_P, best_a = cur_P, cur_a
    for it in range(iters):
        frac = it / max(iters, 1)
        sigma = 0.004 + 0.03 * (1.0 - frac)
        v = _sym_vec(cur_P) + rng.normal(scale=sigma, size=_NRED)
        Q = _sym_solve(v)
        a = _min_area(Q)
        thr = 5e-4 * (1.0 - frac)
        if a > cur_a - thr or rng.random() < 0.05:
            cur_P, cur_a = Q, a
        if a > best_a + 1e-12:
            best_P, best_a = Q, a
        if cur_a < best_a - 3e-3:
            cur_P, cur_a = best_P, best_a
    return best_P, best_a


def _binding_kick(P, rng, sigma):
    """Perturb one to three vertices drawn from the six smallest-area triangles."""
    cr = np.abs(_cross(P))
    order = np.argsort(cr)[:6]
    pts = np.unique(_TRIPLES[order].ravel())
    m = int(rng.integers(1, min(3, len(pts)) + 1))
    idx = rng.choice(pts, size=m, replace=False)
    Q = P.copy()
    Q[idx] += rng.normal(scale=sigma, size=(m, 2))
    return Q


def _isotropic_kick(P, rng, sigma):
    m = int(rng.integers(1, 4))
    idx = rng.choice(_N, size=m, replace=False)
    Q = P.copy()
    Q[idx] += rng.normal(scale=sigma, size=(m, 2))
    return Q


def _asym_hop(P0, rng, iters, sigma_hi=0.07, sigma_lo=0.005, patience=60):
    """Annealed unconstrained basin hopping with binding-triangle kicks; stall-based stop."""
    cur_P = _project(P0)
    cur_a = _min_area(cur_P)
    best_P, best_a = cur_P, cur_a
    stall = 0
    for it in range(iters):
        frac = it / max(iters, 1)
        sigma = sigma_lo + sigma_hi * (1.0 - frac)
        if rng.random() < 0.7:
            Q = _binding_kick(cur_P, rng, sigma)
        else:
            Q = _isotropic_kick(cur_P, rng, sigma)
        Q = _polish(_project(Q), rounds=2)
        a = _min_area(Q)
        if a > cur_a - 2e-4 * (1.0 - frac) or rng.random() < 0.02:
            cur_P, cur_a = Q, a
        if a > best_a + 1e-12:
            best_P, best_a = Q, a
            stall = 0
        else:
            stall += 1
            if stall >= patience:
                break
        if cur_a < best_a - 5e-3:
            cur_P, cur_a = best_P, best_a
    return best_P, best_a


def _finalize(P):
    """Return the inside-verified layout with the larger min area (inward-shrunk safety copy)."""
    Q = _project(np.asarray(P, dtype=float))
    R = _project(_CENT + (Q - _CENT) * (1.0 - 1e-9))
    best, best_a = None, -np.inf
    for C in (Q, R):
        if not _is_inside(C, 1e-9):
            continue
        a = _min_area(C)
        if a > best_a:
            best, best_a = C, a
    if best is None:
        best = _project(_random_points(np.random.default_rng(20240611)))
    return np.asarray(best, dtype=float).reshape(_N, 2)


def heilbronn_triangle11() -> np.ndarray:
    """
    Construct 11 points in the equilateral triangle maximizing the minimum triangle area.
    Deterministic (fixed seed).

    Pipeline:
      1. Mirror-symmetric multistart (48 random/boundary-biased starts).
      2. Annealed symmetric hopping from the top three distinct seeds.
      3. Release symmetry and polish the best symmetric layouts.
      4. Asymmetric binding-kick hopping (stall-stopped) from each released optimum.
      5. Short unconstrained random restarts, then stall-stopped hopping from the top
         three distinct pool entries.
      6. Low-amplitude refinement of the overall best, then margin-safe final repair.
    """
    rng = np.random.default_rng(20240611)

    best_P = _project(_random_points(rng))
    best_a = _min_area(best_P)
    pool = []

    try:
        # Phase 1: mirror-symmetric multistart.
        sym_cands = []
        for k in range(48):
            P = _sym_solve(_sym_start(rng, boundary=(k % 2 == 1)))
            sym_cands.append((_min_area(P), P))
        sym_cands.sort(key=lambda q: -q[0])

        # Phase 2: annealed symmetric hopping from diverse seeds.
        seeds = []
        for a, P in sym_cands:
            if all(abs(a - sa) > 1e-7 for sa, _ in seeds):
                seeds.append((a, P))
            if len(seeds) == 3:
                break
        sym_pool = []
        for _, P0 in seeds:
            Q, a = _sym_hop(P0, rng, 30)
            sym_pool.append((a, Q))
        sym_pool.extend(sym_cands[:3])
        sym_pool.sort(key=lambda q: -q[0])

        # Phase 3: release symmetry and polish the best symmetric layouts.
        released = []
        for _, P0 in sym_pool[:3]:
            Q = _polish(P0, rounds=2)
            a = _min_area(Q)
            released.append((a, Q))
            pool.append((a, Q))
            if a > best_a:
                best_a, best_P = a, Q

        # Phase 4: asymmetric hopping from each top symmetric optimum (stall-stopped).
        for _, P0 in released:
            Q, a = _asym_hop(P0, rng, 200, patience=60)
            pool.append((a, Q))
            if a > best_a:
                best_a, best_P = a, Q

        # Phase 5: short unconstrained random restarts, then hop from distinct top layouts.
        for _ in range(6):
            Q = _polish(_random_points(rng), rounds=2)
            a = _min_area(Q)
            pool.append((a, Q))
            if a > best_a:
                best_a, best_P = a, Q

        pool.sort(key=lambda q: -q[0])
        starts = []
        for a, P in pool:
            if all(abs(a - sa) > 1e-7 for sa, _ in starts):
                starts.append((a, P))
            if len(starts) == 3:
                break
        for _, P0 in starts:
            Q, a = _asym_hop(P0, rng, 150, patience=60)
            if a > best_a:
                best_a, best_P = a, Q

        # Phase 6: low-amplitude refinement of the overall best, then sign-rederiving polish.
        Q, a = _asym_hop(best_P, rng, 120, sigma_hi=0.02, sigma_lo=0.001, patience=50)
        if a > best_a:
            best_a, best_P = a, Q
        Q = _polish(best_P, rounds=3)
        a = _min_area(Q)
        if a > best_a:
            best_a, best_P = a, Q
    except Exception:
        pass

    return _finalize(best_P)


# EVOLVE-BLOCK-END
