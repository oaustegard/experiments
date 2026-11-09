# EVOLVE-BLOCK-START
import numpy as np


def heilbronn_triangle11() -> np.ndarray:
    """
    Construct an arrangement of n points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 11.

    Returns:
        points: np.ndarray of shape (11,2) with the x,y coordinates of the points.
    """
    from itertools import combinations
    from scipy.optimize import minimize

    n = 11
    s3 = np.sqrt(3.0)
    h = s3 / 2.0
    rng = np.random.default_rng(0)

    tri = np.array(list(combinations(range(n), 3)))
    I, J, K = tri[:, 0], tri[:, 1], tri[:, 2]
    m = len(tri)
    rows = np.arange(m)

    def signed_areas(P):
        x, y = P[:, 0], P[:, 1]
        return 0.5 * ((x[J] - x[I]) * (y[K] - y[I]) - (x[K] - x[I]) * (y[J] - y[I]))

    def min_area(P):
        return float(np.min(np.abs(signed_areas(P))))

    def project_inside(P):
        # Shrink toward centroid just enough to guarantee containment.
        x, y = P[:, 0], P[:, 1]
        l3 = y / h
        l2 = x - 0.5 * l3
        l1 = 1.0 - l2 - l3
        lam_min = min(l1.min(), l2.min(), l3.min())
        if lam_min >= 0:
            return P
        a = (1.0 / 3.0) / (1.0 / 3.0 - lam_min) * (1 - 1e-12)
        c = np.array([0.5, h / 3.0])
        return c + a * (P - c)

    def random_start():
        alpha = rng.choice([0.4, 0.7, 1.0])
        w = rng.dirichlet([alpha] * 3, size=n)
        V = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, h]])
        return w @ V

    # Linear containment constraints: y>=0, s3*x-y>=0, s3*(1-x)-y>=0
    A_lin = np.zeros((3 * n, 2 * n + 1))
    b_lin = np.zeros(3 * n)
    for p in range(n):
        A_lin[3 * p, n + p] = 1.0
        A_lin[3 * p + 1, p] = s3
        A_lin[3 * p + 1, n + p] = -1.0
        A_lin[3 * p + 2, p] = -s3
        A_lin[3 * p + 2, n + p] = -1.0
        b_lin[3 * p + 2] = s3

    def optimize_from(P0):
        sgn = np.sign(signed_areas(P0))
        sgn[sgn == 0] = 1.0
        t0 = float(np.min(sgn * signed_areas(P0)))
        z0 = np.concatenate([P0[:, 0], P0[:, 1], [t0]])

        def cons(z):
            x, y, t = z[:n], z[n:2 * n], z[-1]
            P = np.stack([x, y], axis=1)
            area_c = sgn * signed_areas(P) - t
            lin_c = A_lin @ z + b_lin
            return np.concatenate([area_c, lin_c])

        def cons_jac(z):
            x, y = z[:n], z[n:2 * n]
            Jm = np.zeros((m, 2 * n + 1))
            Jm[rows, I] = 0.5 * (y[J] - y[K]) * sgn
            Jm[rows, J] = 0.5 * (y[K] - y[I]) * sgn
            Jm[rows, K] = 0.5 * (y[I] - y[J]) * sgn
            Jm[rows, n + I] = 0.5 * (x[K] - x[J]) * sgn
            Jm[rows, n + J] = 0.5 * (x[I] - x[K]) * sgn
            Jm[rows, n + K] = 0.5 * (x[J] - x[I]) * sgn
            Jm[:, -1] = -1.0
            return np.vstack([Jm, A_lin])

        grad = np.zeros(2 * n + 1)
        grad[-1] = -1.0
        res = minimize(
            lambda z: -z[-1], z0, jac=lambda z: grad, method="SLSQP",
            constraints=[{"type": "ineq", "fun": cons, "jac": cons_jac}],
            options={"maxiter": 300, "ftol": 1e-15},
        )
        z = res.x
        return project_inside(np.stack([z[:n], z[n:2 * n]], axis=1))

    best_P = project_inside(random_start())
    best_val = min_area(best_P)
    pool = [(best_val, best_P)]

    # Phase 1: broad multistart (explores many sign patterns / chambers)
    for _ in range(200):
        try:
            P = optimize_from(random_start())
            if not np.all(np.isfinite(P)):
                continue
            v = min_area(P)
            pool.append((v, P))
            if v > best_val:
                best_val, best_P = v, P
        except Exception:
            continue

    # Phase 2: basin hopping from the top candidates. Perturbation can flip
    # near-degenerate triangles, letting SLSQP move to neighbouring chambers.
    pool.sort(key=lambda t: -t[0])
    for v0, P0 in pool[:5]:
        cur_P, cur_v = P0, v0
        for _ in range(40):
            sigma = rng.choice([0.005, 0.015, 0.04])
            try:
                start = project_inside(cur_P + rng.normal(0.0, sigma, cur_P.shape))
                P = optimize_from(start)
                if not np.all(np.isfinite(P)):
                    continue
                v = min_area(P)
                if v > cur_v:
                    cur_v, cur_P = v, P
                    if v > best_val:
                        best_val, best_P = v, P
            except Exception:
                continue

    # Phase 3: extra search that only ever improves on the Phase 1-2 result
    # (the RNG prefix above is unchanged, so best_val can't regress).
    def hop(cur_P, cur_v, iters, tol, reset_every=None):
        nonlocal best_val, best_P
        for it in range(iters):
            if reset_every and it % reset_every == 0:
                cur_P, cur_v = best_P, best_val
            try:
                start = cur_P.copy()
                r = rng.random()
                if r < 0.35:
                    # Relocate 1-2 points to flip many triangle orientations
                    k = int(rng.integers(1, 3))
                    idx = rng.choice(n, size=k, replace=False)
                    start[idx] = random_start()[:k]
                elif r < 0.6:
                    # Perturb only a few points more strongly
                    k = int(rng.integers(2, 5))
                    idx = rng.choice(n, size=k, replace=False)
                    start[idx] += rng.normal(0.0, 0.05, (k, 2))
                else:
                    start += rng.normal(0.0, rng.choice([0.003, 0.01, 0.025]), start.shape)
                P = optimize_from(project_inside(start))
                if not np.all(np.isfinite(P)):
                    continue
                v = min_area(P)
                if v > cur_v - tol:
                    cur_P, cur_v = P, v
                if v > best_val:
                    best_val, best_P = v, P
            except Exception:
                continue

    # 3a: fresh basins that Phase 2 never explored
    for v0, P0 in pool[5:11]:
        hop(P0, v0, 30, 1e-4)
    # 3b: deeper search around the global best, with periodic resets to it
    hop(best_P, best_val, 150, 2e-4, reset_every=30)

    # Polish the best candidate with a couple of extra SLSQP passes.
    for _ in range(2):
        try:
            P = optimize_from(best_P)
            v = min_area(P)
            if np.all(np.isfinite(P)) and v > best_val:
                best_val, best_P = v, P
        except Exception:
            break

    return best_P


# EVOLVE-BLOCK-END
