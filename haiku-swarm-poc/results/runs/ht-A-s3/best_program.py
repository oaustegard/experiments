# EVOLVE-BLOCK-START
import itertools
import numpy as np

try:
    from scipy.optimize import minimize
except Exception:  # pragma: no cover
    minimize = None

_S3 = np.sqrt(3.0)
_VERTS = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, _S3 / 2.0]])


def _triples(n):
    t = np.array(list(itertools.combinations(range(n), 3)))
    return t[:, 0], t[:, 1], t[:, 2]


def _signed_areas(p, I, J, K):
    xi, yi = p[I, 0], p[I, 1]
    xj, yj = p[J, 0], p[J, 1]
    xk, yk = p[K, 0], p[K, 1]
    return 0.5 * ((xj - xi) * (yk - yi) - (xk - xi) * (yj - yi))


def _random_in_triangle(rng, m):
    """Uniform samples in the triangle, shape (m, 2)."""
    r1 = np.sqrt(rng.rand(m))
    r2 = rng.rand(m)
    return ((1 - r1)[:, None] * _VERTS[0]
            + (r1 * (1 - r2))[:, None] * _VERTS[1]
            + (r1 * r2)[:, None] * _VERTS[2])


def _sym_in_triangle(rng, n):
    """Mirror-symmetric (about x=0.5) random configuration: (n-1)/2 pairs + 1 axis point."""
    k = (n - 1) // 2
    left = _random_in_triangle(rng, k)
    left[:, 0] = np.where(left[:, 0] > 0.5, 1.0 - left[:, 0], left[:, 0])
    right = left.copy()
    right[:, 0] = 1.0 - right[:, 0]
    axis = np.array([[0.5, rng.rand() * _S3 / 2.0]])
    return np.vstack([left, right, axis])


def _project_inside(p):
    """Clip points into the triangle (barycentric clipping)."""
    A, B, C = _VERTS
    T = np.array([[B[0] - A[0], C[0] - A[0]], [B[1] - A[1], C[1] - A[1]]])
    bary = np.linalg.solve(T, (p - A).T).T  # (u, v)
    u = np.clip(bary[:, 0], 0, 1)
    v = np.clip(bary[:, 1], 0, 1)
    s = u + v
    over = s > 1
    u[over] /= s[over]
    v[over] /= s[over]
    return A + u[:, None] * (B - A) + v[:, None] * (C - A)


def _optimize(p0, I0, J0, K0, G, h, n, iters=4, M=90):
    """Max-min area via SLSQP with frozen orientation signs.

    Only the M currently-smallest triangles are constrained (cheaper SLSQP);
    if that lets an omitted triangle shrink, retry with all triangles.
    """
    nv = 2 * n + 1
    p = p0.copy()
    for _ in range(iters):
        a_full = _signed_areas(p, I0, J0, K0)
        t0 = float(np.min(np.abs(a_full)))
        if M < len(I0):
            sel = np.argpartition(np.abs(a_full), M)[:M]
        else:
            sel = np.arange(len(I0))
        I, J, K = I0[sel], J0[sel], K0[sel]
        m = len(I)
        rows = np.arange(m)
        sgn = np.where(a_full[sel] >= 0, 1.0, -1.0)
        z0 = np.concatenate([p.ravel(), [t0]])

        def cons(z, sgn=sgn):
            q = z[:2 * n].reshape(n, 2)
            area_c = sgn * _signed_areas(q, I, J, K) - z[-1]
            lin = G @ z[:2 * n] - h
            return np.concatenate([area_c, lin])

        def jac(z, sgn=sgn):
            q = z[:2 * n].reshape(n, 2)
            xi, yi = q[I, 0], q[I, 1]
            xj, yj = q[J, 0], q[J, 1]
            xk, yk = q[K, 0], q[K, 1]
            Jm = np.zeros((m + G.shape[0], nv))
            Jm[rows, 2 * I] = sgn * 0.5 * (yj - yk)
            Jm[rows, 2 * I + 1] = sgn * 0.5 * (xk - xj)
            Jm[rows, 2 * J] = sgn * 0.5 * (yk - yi)
            Jm[rows, 2 * J + 1] = sgn * 0.5 * (xi - xk)
            Jm[rows, 2 * K] = sgn * 0.5 * (yi - yj)
            Jm[rows, 2 * K + 1] = sgn * 0.5 * (xj - xi)
            Jm[:m, -1] = -1.0
            Jm[m:, :2 * n] = G
            return Jm

        grad_obj = np.zeros(nv)
        grad_obj[-1] = -1.0
        res = minimize(
            lambda z: -z[-1], z0, jac=lambda z: grad_obj, method="SLSQP",
            constraints=[{"type": "ineq", "fun": cons, "jac": jac}],
            options={"maxiter": 300, "ftol": 1e-14},
        )
        q = _project_inside(res.x[:2 * n].reshape(n, 2))
        if not np.all(np.isfinite(q)):
            break
        # Monotone acceptance: stop once re-linearising no longer helps
        newv = float(np.min(np.abs(_signed_areas(q, I0, J0, K0))))
        if newv <= t0 * (1 + 1e-9):
            if newv < t0 * (1 - 1e-7) and M < len(I0):
                M = len(I0)  # omitted triangles interfered: retry with all
                continue
            break
        p = q
    return p


def heilbronn_triangle11() -> np.ndarray:
    """
    Construct an arrangement of n points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 11.

    Returns:
        points: np.ndarray of shape (11,2) with the x,y coordinates of the points.
    """
    n = 11
    I, J, K = _triples(n)

    def min_area(p):
        return float(np.min(np.abs(_signed_areas(p, I, J, K))))

    rng = np.random.RandomState(12345)

    # Fallback: best random configuration (always valid, nonzero area)
    best_p = _random_in_triangle(rng, n)
    best_v = min_area(best_p)

    # Linear containment constraints  G x >= h  on flattened (x0,y0,x1,y1,...)
    G = np.zeros((3 * n, 2 * n))
    h = np.zeros(3 * n)
    for i in range(n):
        G[3 * i, 2 * i + 1] = 1.0                       # y >= 0
        G[3 * i + 1, 2 * i] = _S3                       # sqrt3*x - y >= 0
        G[3 * i + 1, 2 * i + 1] = -1.0
        G[3 * i + 2, 2 * i] = -_S3                      # sqrt3*(1-x) - y >= 0
        G[3 * i + 2, 2 * i + 1] = -1.0
        h[3 * i + 2] = -_S3

    pool = []  # (value, points) of every locally-optimised start
    n_starts = 450
    for si in range(n_starts):
        # Light pre-screen (best of 40) keeps starts diverse; only every
        # 4th start is mirror-symmetric (the optimum is likely asymmetric)
        sym = (si % 4 == 0)
        cand_best, cand_v = None, -1.0
        for _c in range(40):
            c = _sym_in_triangle(rng, n) if sym else _random_in_triangle(rng, n)
            v = min_area(c)
            if v > cand_v:
                cand_best, cand_v = c, v
        if cand_v > best_v:
            best_p, best_v = cand_best, cand_v

        if minimize is None:
            continue
        try:
            # Cheap screening pass (2 re-linearisations); top basins are polished below
            p = _optimize(cand_best, I, J, K, G, h, n, iters=2)
            v = min_area(p)
            if np.isfinite(v):
                pool.append((v, p))
            if np.isfinite(v) and v > best_v:
                best_p, best_v = p, v
        except Exception:
            continue

    # Basin hopping around the incumbent: jitter and point relocation
    if minimize is not None:
        scales = (0.01, 0.02, 0.04, 0.08)
        pool.sort(key=lambda t: -t[0])
        # Fully converge the most promising screened starts before ranking/dedup
        for ri in range(min(20, len(pool))):
            try:
                rp = _optimize(pool[ri][1], I, J, K, G, h, n, iters=10)
                rv = min_area(rp)
            except Exception:
                continue
            if np.isfinite(rv) and rv > pool[ri][0]:
                pool[ri] = (rv, rp)
                if rv > best_v:
                    best_p, best_v = rp, rv
        pool.sort(key=lambda t: -t[0])
        seeds, last = [], None
        for sv, sp in pool:
            if last is None or abs(sv - last) > 1e-5 * max(sv, 1e-12):
                seeds.append(sp)
                last = sv
            if len(seeds) == 8:
                break
        if not seeds:
            seeds = [best_p]
        per = 75
        cur_p, cur_v = best_p, best_v
        loc_p, loc_v = cur_p, cur_v
        for it in range(600):
            if it % per == 0:  # start a new round from the next distinct basin
                loc_p = seeds[(it // per) % len(seeds)]
                loc_v = min_area(loc_p)
                cur_p, cur_v = loc_p, loc_v
            c = cur_p.copy()
            if it % 2 == 0:
                c = c + rng.normal(0.0, scales[(it // 2) % len(scales)], c.shape)
            else:
                k = 1 + (it // 2) % 2
                idx = rng.choice(n, size=k, replace=False)
                c[idx] = _random_in_triangle(rng, k)
            c = _project_inside(c)
            try:
                p = _optimize(c, I, J, K, G, h, n)
                v = min_area(p)
            except Exception:
                continue
            if not np.isfinite(v):
                continue
            if v > best_v:
                best_p, best_v = p, v
            if v > loc_v:
                loc_p, loc_v = p, v
                cur_p, cur_v = p, v
            elif v > cur_v or (v > 0.8 * loc_v and rng.rand() < 0.15):
                cur_p, cur_v = p, v
            else:
                cur_p, cur_v = loc_p, loc_v

        # Extra fine-scale search with an independent RNG (cannot perturb the
        # stream above; best_p only ever changes on strict improvement).
        rng2 = np.random.RandomState(777)
        fine = (0.003, 0.006, 0.012)
        bases = [best_p] + seeds[:3]
        for it in range(160):
            base = bases[0] if (it % 2 == 0 or len(bases) == 1) else bases[1 + (it // 2) % (len(bases) - 1)]
            c = base + rng2.normal(0.0, fine[it % 3], base.shape)
            if it % 5 == 4:
                c[rng2.randint(n)] = _random_in_triangle(rng2, 1)[0]
            c = _project_inside(c)
            try:
                p = _optimize(c, I, J, K, G, h, n)
                v = min_area(p)
            except Exception:
                continue
            if np.isfinite(v) and v > best_v:
                best_p, best_v = p, v
                bases[0] = best_p

    # Final polish: extra re-linearisation passes on the incumbent
    if minimize is not None:
        try:
            p = _optimize(best_p, I, J, K, G, h, n, iters=12)
            v = min_area(p)
            if np.isfinite(v) and v > best_v:
                best_p, best_v = p, v
        except Exception:
            pass

    return _project_inside(best_p)


# EVOLVE-BLOCK-END
