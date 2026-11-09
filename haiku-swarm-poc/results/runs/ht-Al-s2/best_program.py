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
    tri_area = s3 / 4.0
    idx = np.array(list(combinations(range(n), 3)))
    I, J, K = idx[:, 0], idx[:, 1], idx[:, 2]
    m = len(idx)
    rows = np.arange(m)

    def areas(p):
        x, y = p[:, 0], p[:, 1]
        cr = (x[J] - x[I]) * (y[K] - y[I]) - (x[K] - x[I]) * (y[J] - y[I])
        return 0.5 * cr / tri_area

    def min_area(p):
        return np.min(np.abs(areas(p)))

    def inside(p, tol=1e-9):
        return bool(
            np.all(p[:, 1] >= -tol)
            and np.all(s3 * p[:, 0] - p[:, 1] >= -tol)
            and np.all(s3 * (1 - p[:, 0]) - p[:, 1] >= -tol)
        )

    A_ = np.array([0.0, 0.0])
    B_ = np.array([1.0, 0.0])
    C_ = np.array([0.5, s3 / 2])
    edges = [(A_, B_), (B_, C_), (C_, A_)]

    def sample(rng, boundary=False):
        pts = []
        while len(pts) < n:
            u, v = rng.random(2)
            if u + v > 1:
                u, v = 1 - u, 1 - v
            pts.append([u + 0.5 * v, v * s3 / 2])
        pts = np.array(pts)
        if boundary:
            k = rng.integers(3, 8)
            for i in rng.choice(n, size=k, replace=False):
                a, b = edges[rng.integers(3)]
                t = rng.random()
                pts[i] = a + t * (b - a)
        return pts

    # linear containment constraints: G z + h >= 0
    G = np.zeros((3 * n, 2 * n + 1))
    for i in range(n):
        G[3 * i, 2 * i + 1] = 1.0
        G[3 * i + 1, 2 * i] = s3
        G[3 * i + 1, 2 * i + 1] = -1.0
        G[3 * i + 2, 2 * i] = -s3
        G[3 * i + 2, 2 * i + 1] = -1.0
    h = np.zeros(3 * n)
    h[2::3] = s3
    obj_jac = np.concatenate([np.zeros(2 * n), [-1.0]])

    def solve(p0):
        sg = np.sign(areas(p0))
        sg[sg == 0] = 1.0
        f = 0.5 / tri_area

        def cons(z):
            return sg * areas(z[:-1].reshape(n, 2)) - z[-1]

        def cjac(z):
            p = z[:-1].reshape(n, 2)
            x, y = p[:, 0], p[:, 1]
            Jm = np.zeros((m, 2 * n + 1))
            Jm[rows, 2 * I] = (y[J] - y[K]) * f
            Jm[rows, 2 * I + 1] = (x[K] - x[J]) * f
            Jm[rows, 2 * J] = (y[K] - y[I]) * f
            Jm[rows, 2 * J + 1] = (x[I] - x[K]) * f
            Jm[rows, 2 * K] = (y[I] - y[J]) * f
            Jm[rows, 2 * K + 1] = (x[J] - x[I]) * f
            Jm *= sg[:, None]
            Jm[:, -1] = -1.0
            return Jm

        z0 = np.concatenate([p0.ravel(), [np.min(np.abs(areas(p0)))]])
        res = minimize(
            lambda z: -z[-1], z0,
            jac=lambda z: obj_jac,
            constraints=[
                {"type": "ineq", "fun": cons, "jac": cjac},
                {"type": "ineq", "fun": lambda z: G @ z + h, "jac": lambda z: G},
            ],
            method="SLSQP",
            options={"maxiter": 400, "ftol": 1e-14},
        )
        p = res.x[:-1].reshape(n, 2).copy()
        p[:, 1] = np.clip(p[:, 1], 0.0, s3 / 2)
        return p

    rng = np.random.default_rng(0)
    c = np.array([0.5, s3 / 6])

    def project(q):
        for i in range(n):
            for _ in range(80):
                if inside(q[i:i + 1], 0.0):
                    break
                q[i] = c + (q[i] - c) * 0.95
        return q

    cands = []
    for it in range(700):
        try:
            p = solve(sample(rng, boundary=(it % 2 == 0)))
        except Exception:
            continue
        if not inside(p):
            continue
        cands.append((min_area(p), p))

    if not cands:
        return project(sample(rng))

    cands.sort(key=lambda t: -t[0])
    best_val, best = cands[0]

    # polish the top local optima with refreshed sign patterns
    for val, p in cands[:8]:
        q = p
        for _ in range(4):
            try:
                q2 = solve(q)
            except Exception:
                break
            if inside(q2) and min_area(q2) > min_area(q):
                q = q2
            else:
                break
        v = min_area(q)
        if v > best_val:
            best_val, best = v, q

    # basin hopping with multi-scale perturbations
    sigmas = (0.003, 0.008, 0.015, 0.03, 0.06)
    for hop in range(1200):
        sigma = sigmas[hop % len(sigmas)]
        q = best + rng.normal(0.0, sigma, best.shape)
        q = project(q)
        try:
            q2 = solve(q)
        except Exception:
            continue
        if inside(q2) and min_area(q2) > best_val:
            best_val, best = min_area(q2), q2

    # tiny shrink toward centroid to guarantee containment
    best = c + (best - c) * (1 - 1e-10)
    return best


# EVOLVE-BLOCK-END
