# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations
from scipy.optimize import minimize

_N = 11
_S3 = np.sqrt(3.0)
_TRIPLES = np.array(list(combinations(range(_N), 3)))
_I, _J, _K = _TRIPLES[:, 0], _TRIPLES[:, 1], _TRIPLES[:, 2]
_CENTROID = np.array([0.5, _S3 / 6.0])


def _signed_areas(P):
    X, Y = P[:, 0], P[:, 1]
    return 0.5 * ((X[_J] - X[_I]) * (Y[_K] - Y[_I]) - (X[_K] - X[_I]) * (Y[_J] - Y[_I]))


def _min_area(P):
    v = float(np.min(np.abs(_signed_areas(P))))
    return v if np.isfinite(v) else 0.0


def _bound_vals(P):
    X, Y = P[:, 0], P[:, 1]
    return np.concatenate([Y, _S3 * X - Y, _S3 * (1.0 - X) - Y])


def _inside_pt(p, tol=0.0):
    x, y = p[0], p[1]
    return (y >= -tol) and (_S3 * x - y >= -tol) and (_S3 * (1.0 - x) - y >= -tol)


def _project(P):
    """Shrink outside points toward the centroid until they lie inside."""
    Q = np.array(P, dtype=float).reshape(_N, 2).copy()
    for i in range(_N):
        if not np.all(np.isfinite(Q[i])):
            Q[i] = _CENTROID
        if _inside_pt(Q[i]):
            continue
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if _inside_pt(_CENTROID + (Q[i] - _CENTROID) * mid):
                lo = mid
            else:
                hi = mid
        Q[i] = _CENTROID + (Q[i] - _CENTROID) * lo
    return Q


def _random_in_triangle(rng, n):
    u = rng.random(n)
    v = rng.random(n)
    m = u + v > 1.0
    u[m] = 1.0 - u[m]
    v[m] = 1.0 - v[m]
    A = np.array([0.0, 0.0])
    B = np.array([1.0, 0.0])
    C = np.array([0.5, _S3 / 2.0])
    return A + u[:, None] * (B - A) + v[:, None] * (C - A)


def _free_map(z):
    return z[:2 * _N].reshape(_N, 2)


def _sym_map(z):
    A = z[:10].reshape(5, 2)
    P = np.empty((_N, 2))
    P[:5] = A
    P[5:10, 0] = 1.0 - A[:, 0]
    P[5:10, 1] = A[:, 1]
    P[10] = [0.5, z[10]]
    return P


def _sym_start(rng):
    x = rng.random(5) * 0.5
    y = rng.random(5) * _S3 * x
    y0 = rng.random() * _S3 / 2.0
    return np.concatenate([np.column_stack([x, y]).ravel(), [y0]])


def _epi(z0, to_pts, maxiter=250):
    """Epigraph SLSQP: maximize t s.t. sign*area >= t and points inside triangle."""
    try:
        P0 = _project(to_pts(z0))
        s = np.sign(_signed_areas(P0))
        s[s == 0] = 1.0
        t0 = max(_min_area(P0), 1e-6)
        zz0 = np.concatenate([np.asarray(z0, dtype=float), [t0]])

        def area_con(zz):
            return s * _signed_areas(to_pts(zz[:-1])) - zz[-1]

        def bound_con(zz):
            return _bound_vals(to_pts(zz[:-1]))

        res = minimize(lambda zz: -zz[-1], zz0, method="SLSQP",
                       constraints=[{"type": "ineq", "fun": area_con},
                                    {"type": "ineq", "fun": bound_con}],
                       options={"maxiter": maxiter, "ftol": 1e-13})
        P = _project(to_pts(res.x[:-1]))
        if np.all(np.isfinite(P)):
            return P, _min_area(P)
    except Exception:
        pass
    P = _project(to_pts(z0))
    return P, _min_area(P)


def _sym_hop(z, rng, iters):
    """Basin hopping in the 11-parameter mirror-symmetric space (fixed iterations)."""
    cur_z = np.asarray(z, dtype=float).copy()
    _, cur_val = _epi(cur_z, _sym_map, maxiter=200)
    best_z, best_val = cur_z, cur_val
    for _ in range(iters):
        Q = _project(_sym_map(cur_z))
        a = np.abs(_signed_areas(Q))
        tri = _TRIPLES[int(np.argmin(a))]
        zn = cur_z.copy()
        if rng.random() < 0.6:
            idx = tri[rng.integers(3)]
            slot = idx if idx < 5 else (idx - 5 if idx < 10 else 10)
            if slot < 10:
                p = _random_in_triangle(rng, 1)[0]
                x = min(p[0], 1.0 - p[0])
                s0 = slot if slot < 5 else slot - 5
                zn[2 * s0] = x * 0.999
                zn[2 * s0 + 1] = p[1]
            else:
                zn[10] = rng.random() * _S3 / 2.0
        else:
            zn = zn + rng.normal(scale=0.02, size=zn.shape)
        _, v2 = _epi(zn, _sym_map, maxiter=200)
        if v2 >= cur_val - 1e-12:
            cur_z, cur_val = zn, v2
        if v2 > best_val + 1e-12:
            best_z, best_val = zn, v2
    return best_z, best_val


def _free_ils(P, rng, iters):
    """Iterated local search over all 22 coordinates, focused on the tightest triangle."""
    cur_P, cur_val = _epi(P.ravel().copy(), _free_map, maxiter=300)
    best_P, best_val = cur_P, cur_val
    for _ in range(iters):
        Q = _project(cur_P)
        a = np.abs(_signed_areas(Q))
        tri = _TRIPLES[int(np.argmin(a))]
        Pn = cur_P.copy()
        r = rng.random()
        if r < 0.5:
            idx = tri[rng.integers(3)]
            Pn[idx] = _random_in_triangle(rng, 1)[0]
        elif r < 0.8:
            for idx in tri:
                Pn[idx] = Pn[idx] + rng.normal(scale=0.03, size=2)
        else:
            Pn = Pn + rng.normal(scale=0.01, size=Pn.shape)
        Pn = _project(Pn)
        Pc, vc = _epi(Pn.ravel().copy(), _free_map, maxiter=200)
        if vc >= cur_val - 1e-12:
            cur_P, cur_val = Pc, vc
        if vc > best_val + 1e-12:
            best_P, best_val = Pc, vc
    return best_P, best_val


def _kick_polish(P, val, rng, rounds):
    """Small-noise kicks around the incumbent; accept only strict improvements."""
    best_P, best_val = P, val
    scales = (0.002, 0.005, 0.01)
    for r in range(rounds):
        sc = scales[r % len(scales)]
        Pn = _project(best_P + rng.normal(scale=sc, size=best_P.shape))
        Pc, vc = _epi(Pn.ravel().copy(), _free_map, maxiter=250)
        if vc > best_val + 1e-12:
            best_P, best_val = Pc, vc
    return best_P, best_val


def _feasible_guard(P):
    """Final safety: force every point inside the triangle."""
    Q = _project(np.asarray(P, dtype=float).reshape(_N, 2))
    Q = np.clip(Q, [0.0, 0.0], [1.0, _S3 / 2.0])
    return _project(Q)


def _fallback():
    G = []
    for i in range(5):
        for j in range(5 - i):
            k = 4 - i - j
            G.append((i * np.array([0.0, 0.0]) + j * np.array([1.0, 0.0])
                      + k * np.array([0.5, _S3 / 2.0])) / 4.0)
    return _project(np.array(G[:_N]))


def heilbronn_triangle11() -> np.ndarray:
    """
    Construct 11 points in the equilateral triangle maximizing the minimum triangle area.

    Method: fixed-seed multistart mixing 60 mirror-symmetric and 40 free-form epigraph
    SLSQP starts; symmetric basin hopping on the top 6 symmetric basins; free-form
    polishing and tight-triangle ILS on the best candidates; kick-polish. All loops use
    fixed iteration counts (no wall-clock dependence), so output is deterministic.
    """
    try:
        rng = np.random.default_rng(12345)

        sym_cands = []
        for _ in range(60):
            z0 = _sym_start(rng)
            _, v = _epi(z0, _sym_map, maxiter=250)
            sym_cands.append((v if np.isfinite(v) else -np.inf, z0))
        sym_cands.sort(key=lambda p: -p[0])

        pool = []
        for v, z in sym_cands[:6]:
            zb, _ = _sym_hop(z, rng, iters=8)
            Pb, vb = _epi(zb, _sym_map, maxiter=250)
            if np.isfinite(vb):
                pool.append((vb, Pb))

        free = []
        for _ in range(40):
            P0 = _random_in_triangle(rng, _N)
            Pf, vf = _epi(P0.ravel().copy(), _free_map, maxiter=300)
            if np.isfinite(vf):
                free.append((vf, Pf))
        for v, P in pool:
            Pf, vf = _epi(P.ravel().copy(), _free_map, maxiter=300)
            free.append((vf, Pf))
            Pn = _project(P + rng.normal(scale=0.005, size=P.shape))
            Pc, vc = _epi(Pn.ravel().copy(), _free_map, maxiter=250)
            free.append((vc, Pc))
        free = [c for c in free if np.isfinite(c[0]) and c[0] > 0]
        free.sort(key=lambda c: -c[0])
        if not free:
            raise ValueError("no free candidates")

        best_val, best_P = free[0][0], free[0][1]
        for _, P0 in free[:4]:
            Pf, vf = _free_ils(P0, rng, iters=25)
            if vf > best_val + 1e-12:
                best_P, best_val = Pf, vf

        for _ in range(2):
            Pf, vf = _free_ils(best_P, rng, iters=15)
            if vf > best_val + 1e-12:
                best_P, best_val = Pf, vf
            else:
                break

        best_P, best_val = _kick_polish(best_P, best_val, rng, rounds=12)

        if not np.isfinite(best_val) or best_val <= 0:
            raise ValueError("degenerate result")
        return _feasible_guard(best_P)
    except Exception:
        return _fallback()


# EVOLVE-BLOCK-END
