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
    import time

    n = 11
    s3 = np.sqrt(3.0)
    tri = np.array(list(combinations(range(n), 3)))
    I, J, K = tri[:, 0], tri[:, 1], tri[:, 2]
    m = len(tri)
    rows = np.arange(m)

    def areas(p):
        x, y = p[:, 0], p[:, 1]
        return 0.5 * ((x[J] - x[I]) * (y[K] - y[I]) - (x[K] - x[I]) * (y[J] - y[I]))

    def min_area(p):
        return float(np.min(np.abs(areas(p))))

    def random_points(rng):
        u = rng.random((n, 2))
        flip = u.sum(axis=1) > 1
        u[flip] = 1 - u[flip]
        return np.column_stack([u[:, 0] + 0.5 * u[:, 1], (s3 / 2) * u[:, 1]])

    # Linear containment constraints: A @ z >= 0 for z = [x(0..n-1), y(0..n-1), t]
    A = np.zeros((3 * n, 2 * n + 1))
    for i in range(n):
        A[3 * i, n + i] = 1.0                      # y >= 0
        A[3 * i + 1, i] = s3; A[3 * i + 1, n + i] = -1.0   # s3*x - y >= 0
        A[3 * i + 2, i] = -s3; A[3 * i + 2, n + i] = -1.0  # s3*(1-x) - y >= 0
    # Third constraint is s3*(1-x) - y >= 0  ->  -s3*x - y >= -s3
    b_off = np.zeros(3 * n)
    b_off[2::3] = -s3

    rng = np.random.default_rng(42)
    best_pts, best_val = None, -1.0
    start = time.time()
    time_budget = 27.0
    max_starts = 1500

    def project(q):
        # Map points into the triangle via (v, u) coordinates
        v = np.clip(q[:, 1] / (s3 / 2), 0.0, 1.0)
        u = np.clip(q[:, 0] - 0.5 * v, 0.0, 1.0 - v)
        return np.column_stack([u + 0.5 * v, (s3 / 2) * v])

    def sym_points(rng):
        # 5 mirrored pairs (x -> 1-x) plus one point on the axis x = 0.5
        q = random_points(rng)
        h = n // 2
        q[h:2 * h, 0] = 1.0 - q[:h, 0]
        q[h:2 * h, 1] = q[:h, 1]
        t = rng.random()
        q[n - 1] = [0.5, t * (s3 / 2)]
        return q

    sigmas = [0.003, 0.006, 0.012, 0.025, 0.05]
    for it in range(max_starts):
        if time.time() - start > time_budget and best_pts is not None:
            break
        r = rng.random()
        if best_pts is None or it < 40 or r < 0.08:
            p0 = sym_points(rng) if rng.random() < 0.5 else random_points(rng)
        elif r < 0.50:
            # Relocate 1-3 random points, keep the rest of the best arrangement
            p0 = best_pts.copy()
            k = int(rng.integers(1, 4))
            idx = rng.choice(n, size=k, replace=False)
            p0[idx] = random_points(rng)[idx]
        else:
            sig = sigmas[int(rng.integers(len(sigmas)))]
            p0 = project(best_pts + sig * rng.normal(size=(n, 2)))
        a0 = areas(p0)
        sg = np.where(a0 >= 0, 1.0, -1.0)
        z0 = np.concatenate([p0[:, 0], p0[:, 1], [np.min(np.abs(a0))]])

        def cons_f(z):
            p = np.column_stack([z[:n], z[n:2 * n]])
            return sg * areas(p) - z[-1]

        def cons_j(z):
            x, y = z[:n], z[n:2 * n]
            Jm = np.zeros((m, 2 * n + 1))
            Jm[rows, I] = 0.5 * (y[J] - y[K])
            Jm[rows, J] = 0.5 * (y[K] - y[I])
            Jm[rows, K] = 0.5 * (y[I] - y[J])
            Jm[rows, n + I] = 0.5 * (x[K] - x[J])
            Jm[rows, n + J] = 0.5 * (x[I] - x[K])
            Jm[rows, n + K] = 0.5 * (x[J] - x[I])
            Jm *= sg[:, None]
            Jm[:, -1] = -1.0
            return Jm

        obj_g = np.zeros(2 * n + 1)
        obj_g[-1] = -1.0
        z = z0
        try:
            # Two rounds: re-derive orientation signs from the solution and re-optimize
            for _round in range(2):
                res = minimize(
                    lambda z: -z[-1], z, jac=lambda z: obj_g, method="SLSQP",
                    constraints=[
                        {"type": "ineq", "fun": cons_f, "jac": cons_j},
                        {"type": "ineq", "fun": lambda z: A @ z - b_off,
                         "jac": lambda z: A},
                    ],
                    options={"maxiter": 300, "ftol": 1e-13},
                )
                z = np.array(res.x)
                pp = np.column_stack([z[:n], z[n:2 * n]])
                aa = areas(pp)
                sg = np.where(aa >= 0, 1.0, -1.0)
                z[-1] = np.min(np.abs(aa))
        except Exception:
            continue
        p = np.column_stack([z[:n], z[n:2 * n]])
        # Project back into the triangle for safety
        p[:, 1] = np.clip(p[:, 1], 0.0, None)
        viol = np.maximum(0, p[:, 1] - s3 * p[:, 0])
        p[:, 0] += viol / (2 * s3) * 0 + viol / s3 * 0.5
        viol2 = np.maximum(0, p[:, 1] - s3 * (1 - p[:, 0]))
        p[:, 0] -= viol2 / s3 * 0.5
        inside = np.all(p[:, 1] >= -1e-12) and np.all(s3 * p[:, 0] - p[:, 1] >= -1e-9) \
            and np.all(s3 * (1 - p[:, 0]) - p[:, 1] >= -1e-9)
        if not inside:
            continue
        v = min_area(p)
        if v > best_val:
            best_val, best_pts = v, p.copy()

    if best_pts is not None:
        # Final high-precision polish of the best arrangement
        def make_cons(sgn):
            def f(z):
                return sgn * areas(np.column_stack([z[:n], z[n:2 * n]])) - z[-1]

            def jac(z):
                x, y = z[:n], z[n:2 * n]
                Jm = np.zeros((m, 2 * n + 1))
                Jm[rows, I] = 0.5 * (y[J] - y[K])
                Jm[rows, J] = 0.5 * (y[K] - y[I])
                Jm[rows, K] = 0.5 * (y[I] - y[J])
                Jm[rows, n + I] = 0.5 * (x[K] - x[J])
                Jm[rows, n + J] = 0.5 * (x[I] - x[K])
                Jm[rows, n + K] = 0.5 * (x[J] - x[I])
                Jm *= sgn[:, None]
                Jm[:, -1] = -1.0
                return Jm
            return f, jac

        try:
            zc = np.concatenate([best_pts[:, 0], best_pts[:, 1], [best_val]])
            for _ in range(3):
                pc = np.column_stack([zc[:n], zc[n:2 * n]])
                sgn = np.where(areas(pc) >= 0, 1.0, -1.0)
                cf, cj = make_cons(sgn)
                zc[-1] = min_area(pc)
                res = minimize(
                    lambda z: -z[-1], zc, jac=lambda z: obj_g, method="SLSQP",
                    constraints=[
                        {"type": "ineq", "fun": cf, "jac": cj},
                        {"type": "ineq", "fun": lambda z: A @ z - b_off,
                         "jac": lambda z: A},
                    ],
                    options={"maxiter": 500, "ftol": 1e-16},
                )
                zc = np.array(res.x)
                pc = np.column_stack([zc[:n], zc[n:2 * n]])
                ok = (np.all(pc[:, 1] >= 0) and np.all(s3 * pc[:, 0] - pc[:, 1] >= 0)
                      and np.all(s3 * (1 - pc[:, 0]) - pc[:, 1] >= 0))
                v = min_area(pc)
                if ok and v > best_val:
                    best_val, best_pts = v, pc.copy()
                zc[-1] = v
        except Exception:
            pass

    if best_pts is None:
        best_pts = random_points(rng)
    return best_pts


# EVOLVE-BLOCK-END
