import numpy as np
from itertools import combinations
from scipy.optimize import minimize

# EVOLVE-BLOCK-START
_N = 11
_SQ3 = np.sqrt(3.0)
_VERTS = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, _SQ3 / 2.0]])
_TRI_AREA = _SQ3 / 4.0
_TRI = np.array(list(combinations(range(_N), 3)), dtype=int)
_I, _J, _K = _TRI[:, 0], _TRI[:, 1], _TRI[:, 2]
_M = _TRI.shape[0]
_ROWS = np.arange(_M)
_NV = 2 * _N + 1          # 22 coordinates + epigraph variable t
_SEED = 20240611
_TOL = 1e-9
_AXIS = 10                # point fixed on the mirror axis x = 0.5

_N_SYM = 50               # symmetric multistart seeds
_N_SYM_SEEDS = 4          # symmetric basins refined by basin hopping
_N_SYM_HOP = 30           # symmetric basin-hop steps per seed
_SYM_SIGMA = 0.03
_N_SYM_POLISH = 8         # best symmetric layouts polished in full space
_N_RANDOM = 24
_N_BOUND = 6              # boundary-heavy full-space starts
_N_LATTICE = 4
_N_ILS = 120
_N_ELITE = 6
_N_FINAL = 3              # elite layouts given a final full-space polish
_RESTART_EVERY = 30
_N_CAND = 400             # candidate points for guided relocation
_T0 = 2e-4                # cooler Metropolis temperature on normalized min area

_BMAT = np.array([
    [_VERTS[0, 0], _VERTS[1, 0], _VERTS[2, 0]],
    [_VERTS[0, 1], _VERTS[1, 1], _VERTS[2, 1]],
    [1.0, 1.0, 1.0],
])
_BINV = np.linalg.inv(_BMAT)

# Containment constraints on the full variable vector: A_IN @ z + B_IN >= 0
_A_IN = np.zeros((3 * _N, _NV))
_B_IN = np.zeros(3 * _N)
for _p in range(_N):
    _ix, _iy = 2 * _p, 2 * _p + 1
    _r = 3 * _p
    _A_IN[_r, _iy] = 1.0                       # y >= 0
    _A_IN[_r + 1, _ix] = _SQ3                  # sqrt3*x - y >= 0
    _A_IN[_r + 1, _iy] = -1.0
    _B_IN[_r + 2] = _SQ3                       # sqrt3*(1-x) - y >= 0
    _A_IN[_r + 2, _ix] = -_SQ3
    _A_IN[_r + 2, _iy] = -1.0

# Mirror-symmetric reduction (x -> 1-x).
# zr = [x_0..x_4, y_0..y_4, y_axis, t]; z = _T_SYM @ zr + _C_SYM.
_NS = 12
_T_SYM = np.zeros((_NV, _NS))
_C_SYM = np.zeros(_NV)
for _i in range(5):
    _T_SYM[2 * _i, _i] = 1.0
    _T_SYM[2 * _i + 1, 5 + _i] = 1.0
    _T_SYM[2 * (5 + _i), _i] = -1.0
    _C_SYM[2 * (5 + _i)] = 1.0
    _T_SYM[2 * (5 + _i) + 1, 5 + _i] = 1.0
_C_SYM[2 * _AXIS] = 0.5
_T_SYM[2 * _AXIS + 1, 10] = 1.0
_T_SYM[_NV - 1, _NS - 1] = 1.0

_T_ID = np.eye(_NV)
_C_ID = np.zeros(_NV)


def _signed_areas(P):
    xi, yi = P[_I, 0], P[_I, 1]
    xj, yj = P[_J, 0], P[_J, 1]
    xk, yk = P[_K, 0], P[_K, 1]
    return 0.5 * ((xj - xi) * (yk - yi) - (xk - xi) * (yj - yi))


def _project(P):
    """Project points into the triangle via barycentric clipping."""
    P = np.asarray(P, dtype=float).reshape(_N, 2)
    homog = np.hstack([P, np.ones((_N, 1))])
    lam = np.clip(homog @ _BINV.T, 0.0, None)
    lam = lam / np.sum(lam, axis=1, keepdims=True)
    return lam @ _VERTS


def _violation(P):
    x, y = P[:, 0], P[:, 1]
    v1 = np.max(-y)
    v2 = np.max(-(_SQ3 * x - y))
    v3 = np.max(-(_SQ3 * (1.0 - x) - y))
    return float(max(0.0, v1, v2, v3))


def _score(P):
    """Normalized min triangle area for a feasible layout, else -1."""
    if not np.all(np.isfinite(P)) or _violation(P) > _TOL:
        return -1.0
    return float(np.min(np.abs(_signed_areas(P)))) / _TRI_AREA


def _random_points(rng, n=_N):
    """Uniform random points in the triangle via the reflection trick."""
    u = rng.random(n)
    v = rng.random(n)
    m = u + v > 1.0
    u[m] = 1.0 - u[m]
    v[m] = 1.0 - v[m]
    return np.column_stack([u + 0.5 * v, _SQ3 / 2.0 * v])


def _boundary_seed(rng, n=_N):
    """Boundary-heavy start: points on the three edges, lightly jittered."""
    pts = np.empty((n, 2))
    for p in range(n):
        e = int(rng.integers(3))
        t = rng.random()
        A = _VERTS[e]
        B = _VERTS[(e + 1) % 3]
        pts[p] = A + t * (B - A)
    pts = pts + rng.normal(0.0, 0.005, size=pts.shape)
    return _project(pts)


def _symmetric_seed(rng):
    """Mirror-symmetric start: 5 free points, their mirrors, one axis point."""
    half = _random_points(rng, 5)
    mirror = half.copy()
    mirror[:, 0] = 1.0 - mirror[:, 0]
    axis = np.array([[0.5, rng.random() * _SQ3 / 2.0]])
    return np.vstack([half, mirror, axis])


def _lattice_seed(rng):
    """Triangular-lattice seed: 10 lattice points plus one jittered extra point."""
    B = np.array([1.0, 0.0])
    C = np.array([0.5, _SQ3 / 2.0])
    pts = [(i / 3.0) * B + (j / 3.0) * C
           for i in range(4) for j in range(4) if i + j <= 3]
    P = np.vstack([np.array(pts), _random_points(rng, 1)])
    P = P + rng.normal(0.0, 0.01, size=P.shape)
    return _project(P)


def _guided_relocation(P, idx, rng):
    """Place point idx at the candidate that maximizes the resulting min area.

    Triangles not involving idx are fixed; candidates are sampled uniformly
    in the triangle and scored by the min over all triangles with idx.
    """
    others = [j for j in range(_N) if j != idx]
    pairs = np.array(list(combinations(others, 2)), dtype=int)
    keep = ~np.any(_TRI == idx, axis=1)
    base = float(np.min(np.abs(_signed_areas(P)[keep])))
    C = _random_points(rng, _N_CAND)
    A = P[pairs[:, 0]]
    B = P[pairs[:, 1]]
    dxA = A[None, :, 0] - C[:, None, 0]
    dyA = A[None, :, 1] - C[:, None, 1]
    dxB = B[None, :, 0] - C[:, None, 0]
    dyB = B[None, :, 1] - C[:, None, 1]
    areas = 0.5 * np.abs(dxA * dyB - dyA * dxB)
    scores = np.minimum(base, np.min(areas, axis=1))
    return C[int(np.argmax(scores))]


def _area_jac(P, s):
    """Jacobian of s*signed_area wrt [coords, t] (M x NV), area in raw units."""
    xi, yi = P[_I, 0], P[_I, 1]
    xj, yj = P[_J, 0], P[_J, 1]
    xk, yk = P[_K, 0], P[_K, 1]
    Jm = np.zeros((_M, _NV))
    Jm[_ROWS, 2 * _I] = s * 0.5 * (yj - yk)
    Jm[_ROWS, 2 * _I + 1] = s * 0.5 * (xk - xj)
    Jm[_ROWS, 2 * _J] = s * 0.5 * (yk - yi)
    Jm[_ROWS, 2 * _J + 1] = s * 0.5 * (xi - xk)
    Jm[_ROWS, 2 * _K] = s * 0.5 * (yi - yj)
    Jm[_ROWS, 2 * _K + 1] = s * 0.5 * (xj - xi)
    Jm[:, -1] = -1.0
    return Jm


def _run_slsqp(zr0, T, c, s, maxiter=300):
    """Max-min SLSQP in reduced variables zr (last entry = epigraph t, normalized).

    Full variables z = T @ zr + c. Orientation signs s are fixed for this solve.
    """
    nr = T.shape[1]
    A_red = _A_IN @ T
    B_red = _A_IN @ c + _B_IN
    g = np.zeros(nr)
    g[-1] = -1.0

    def area_fun(zr):
        z = T @ zr + c
        return s * _signed_areas(z[:-1].reshape(_N, 2)) / _TRI_AREA - z[-1]

    def area_jac(zr):
        z = T @ zr + c
        J = _area_jac(z[:-1].reshape(_N, 2), s)
        J[:, :-1] /= _TRI_AREA
        return J @ T

    def in_fun(zr):
        return A_red @ zr + B_red

    def in_jac(zr):
        return A_red

    def obj(zr):
        return -zr[-1]

    def obj_jac(zr):
        return g

    cons = [
        {"type": "ineq", "fun": area_fun, "jac": area_jac},
        {"type": "ineq", "fun": in_fun, "jac": in_jac},
    ]
    try:
        res = minimize(obj, zr0, jac=obj_jac, constraints=cons, method="SLSQP",
                       options={"maxiter": maxiter, "ftol": 1e-13})
        if not np.all(np.isfinite(res.x)):
            return None
        z = T @ res.x + c
        return _project(z[:-1].reshape(_N, 2))
    except Exception:
        return None


def _pack_full(P, t):
    return np.concatenate([P.ravel(), [t]])


def _pack_sym(P, t):
    return np.concatenate([P[:5, 0], P[:5, 1], [P[_AXIS, 1], t]])


def _unfold_sym(zr):
    z = _T_SYM @ zr + _C_SYM
    return _project(z[:-1].reshape(_N, 2))


def _optimize(P0, T, c, pack, rounds=3):
    """Re-solve with re-derived orientation signs until the layout stabilizes."""
    best = None
    best_v = -1.0
    P = _project(P0)
    for _ in range(rounds):
        s = np.sign(_signed_areas(P))
        s[s == 0] = 1.0
        t0 = float(np.min(s * _signed_areas(P))) / _TRI_AREA
        Pn = _run_slsqp(pack(P, t0), T, c, s)
        if Pn is None:
            break
        v = _score(Pn)
        if v > best_v + 1e-12:
            best_v, best = v, Pn
            P = Pn
        else:
            break
    return best


def _solve(P0):
    """Full unsymmetric max-min polish."""
    return _optimize(P0, _T_ID, _C_ID, _pack_full)


def _solve_sym(P0):
    """Mirror-symmetric reduced max-min search (12 variables)."""
    return _optimize(P0, _T_SYM, _C_SYM, _pack_sym)


def _add_elite(elite, v, P):
    """Keep the top distinct configurations by score."""
    for e in elite:
        if abs(e[0] - v) < 1e-9:
            return
    elite.append((v, P.copy()))
    elite.sort(key=lambda e: -e[0])
    del elite[_N_ELITE:]


def heilbronn_triangle11() -> np.ndarray:
    """
    Construct 11 points on or inside the triangle (0,0),(1,0),(0.5,sqrt(3)/2)
    maximizing the minimum triangle area. Deterministic (fixed seed).

    Returns:
        points: np.ndarray of shape (11,2) with the x,y coordinates of the points.
    """
    rng = np.random.default_rng(_SEED)

    best_P = _project(_random_points(rng))
    best_v = _score(best_P)
    elite = []
    _add_elite(elite, best_v, best_P)

    def consider(P):
        nonlocal best_P, best_v
        if P is None:
            return None
        P = _project(P)
        v = _score(P)
        if v < 0.0:
            return None
        if v > best_v:
            best_v = v
            best_P = P.copy()
        _add_elite(elite, v, P)
        return v, P

    try:
        # Stage A: mirror-symmetric multistart in the 12-variable reduced space.
        sym_results = []
        for _ in range(_N_SYM):
            res = consider(_solve_sym(_symmetric_seed(rng)))
            if res is not None:
                sym_results.append(res)
        sym_results.sort(key=lambda e: -e[0])

        # Stage B: symmetric basin hopping on the top symmetric basins.
        sym_refined = []
        for v0, P0 in sym_results[:_N_SYM_SEEDS]:
            cur_v, cur_P = v0, P0.copy()
            for _ in range(_N_SYM_HOP):
                zr = _pack_sym(cur_P, 0.0)
                zr[:-1] += rng.normal(0.0, _SYM_SIGMA, size=_NS - 1)
                res = consider(_solve_sym(_unfold_sym(zr)))
                if res is not None and res[0] > cur_v + 1e-12:
                    cur_v, cur_P = res[0], res[1].copy()
            sym_refined.append((cur_v, cur_P))
        sym_refined.sort(key=lambda e: -e[0])

        # Stage C: full unsymmetric polish of the best symmetric layouts.
        for _, P in (sym_refined + sym_results)[:_N_SYM_POLISH]:
            consider(_solve(P))

        # Stage D: random, boundary-heavy and lattice multistart on the full problem.
        for _ in range(_N_RANDOM):
            consider(_solve(_random_points(rng)))
        for _ in range(_N_BOUND):
            consider(_solve(_boundary_seed(rng)))
        for _ in range(_N_LATTICE):
            consider(_solve(_lattice_seed(rng)))

        # Stage E: full-space ILS with Metropolis acceptance and elite restarts.
        cur_P, cur_v = best_P.copy(), best_v
        for it in range(_N_ILS):
            frac = it / max(1, _N_ILS - 1)
            temp = _T0 * (1.0 - frac) + 1e-6

            if it > 0 and it % _RESTART_EVERY == 0 and elite:
                pick = elite[rng.integers(len(elite))]
                cur_v, cur_P = pick[0], pick[1].copy()

            P0 = cur_P.copy()
            mode = it % 5
            if mode == 0:
                idx = rng.integers(_N)
                P0[idx] = _random_points(rng, 1)[0]
            elif mode == 1:
                idxs = rng.choice(_N, size=2, replace=False)
                P0[idxs] = _random_points(rng, 2)
            elif mode == 2 or mode == 3:
                areas = np.abs(_signed_areas(cur_P))
                worst = np.argsort(areas)[:4]
                pts = np.unique(_TRI[worst].ravel())
                scale = 0.05 if mode == 2 else 0.02
                P0[pts] += rng.normal(0.0, scale, size=(pts.size, 2))
            else:
                # Guided relocation: move a vertex of the smallest triangle
                # to the position that maximizes the resulting min area.
                tri = _TRI[int(np.argmin(np.abs(_signed_areas(cur_P))))]
                idx = int(rng.choice(tri))
                P0[idx] = _guided_relocation(cur_P, idx, rng)

            res = consider(_solve(P0))
            if res is None:
                continue
            v, Pc = res
            if v > cur_v or rng.random() < np.exp((v - cur_v) / temp):
                cur_v, cur_P = v, Pc.copy()

        # Stage F: final full-space polish of the incumbent and top elite layouts.
        consider(_solve(best_P))
        for _, P in elite[:_N_FINAL]:
            consider(_solve(P))
    except Exception:
        if best_P is None:
            best_P = _project(_random_points(np.random.default_rng(0)))

    return np.asarray(_project(best_P), dtype=float).reshape(_N, 2)
# EVOLVE-BLOCK-END
