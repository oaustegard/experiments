# EVOLVE-BLOCK-START
import numpy as np


def heilbronn_triangle11() -> np.ndarray:
    """
    Construct an arrangement of n points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 11.

    Returns:
        points: np.ndarray of shape (11,2) with the x,y coordinates of the points.
    """
    import itertools
    import time
    from scipy.optimize import minimize

    n = 11
    s3 = np.sqrt(3.0)
    rng = np.random.default_rng(12345)
    tri = np.array(list(itertools.combinations(range(n), 3)))
    I, J, K = tri[:, 0], tri[:, 1], tri[:, 2]
    m = len(tri)
    rows = np.arange(m)

    def signed_areas(p):
        x, y = p[:, 0], p[:, 1]
        return 0.5 * ((x[J] - x[I]) * (y[K] - y[I]) - (x[K] - x[I]) * (y[J] - y[I]))

    def min_area(p):
        return np.abs(signed_areas(p)).min()

    def sample():
        u = rng.random((n, 2))
        flip = u.sum(axis=1) > 1
        u[flip] = 1 - u[flip]
        return np.column_stack([u[:, 0] + 0.5 * u[:, 1], s3 / 2 * u[:, 1]])

    # Linear containment constraints: A z + b >= 0
    A = np.zeros((3 * n, 2 * n + 1))
    b = np.zeros(3 * n)
    for i in range(n):
        A[3 * i, 2 * i + 1] = 1.0                      # y >= 0
        A[3 * i + 1, 2 * i] = s3                       # s3*x - y >= 0
        A[3 * i + 1, 2 * i + 1] = -1.0
        A[3 * i + 2, 2 * i] = -s3                      # s3*(1-x) - y >= 0
        A[3 * i + 2, 2 * i + 1] = -1.0
        b[3 * i + 2] = s3

    def optimize(p0):
        sg = np.sign(signed_areas(p0))
        sg[sg == 0] = 1.0
        z0 = np.concatenate([p0.ravel(), [min_area(p0)]])

        def cons(z):
            return sg * signed_areas(z[:-1].reshape(n, 2)) - z[-1]

        def cons_jac(z):
            p = z[:-1].reshape(n, 2)
            x, y = p[:, 0], p[:, 1]
            M = np.zeros((m, 2 * n + 1))
            M[rows, 2 * I] = 0.5 * (y[J] - y[K])
            M[rows, 2 * J] = 0.5 * (y[K] - y[I])
            M[rows, 2 * K] = -0.5 * (y[J] - y[I])
            M[rows, 2 * I + 1] = 0.5 * (x[K] - x[J])
            M[rows, 2 * J + 1] = 0.5 * (x[I] - x[K])
            M[rows, 2 * K + 1] = 0.5 * (x[J] - x[I])
            M *= sg[:, None]
            M[:, -1] = -1.0
            return M

        obj_grad = np.zeros(2 * n + 1)
        obj_grad[-1] = -1.0
        res = minimize(
            lambda z: -z[-1], z0, jac=lambda z: obj_grad, method="SLSQP",
            constraints=[
                {"type": "ineq", "fun": cons, "jac": cons_jac},
                {"type": "ineq", "fun": lambda z: A @ z + b, "jac": lambda z: A},
            ],
            options={"maxiter": 500, "ftol": 1e-15},
        )
        return res.x[:-1].reshape(n, 2)

    def inside_fix(p):
        p = p.copy()
        # Pull slightly toward the centroid to remove tiny numerical violations
        c = np.array([0.5, s3 / 6])
        for _ in range(60):
            x, y = p[:, 0], p[:, 1]
            viol = max((-y).max(), (y - s3 * x).max(), (y - s3 * (1 - x)).max())
            if viol <= 0:
                break
            p = c + (p - c) * (1 - 1e-12 - 10 * viol)
        return p

    best_p, best_v = None, -1.0
    start = time.time()
    for it in range(100000):
        if time.time() - start > 40 and best_p is not None:
            break
        if it < 40 or best_p is None or rng.random() < 0.25:
            if rng.random() < 0.4:
                # Mirror-symmetric start about x = 0.5: 5 pairs + 1 axis point
                h = sample()[:5]
                axis = np.array([[0.5, rng.random() * s3 / 2 * 0.9 + 0.01]])
                mir = np.column_stack([1.0 - h[:, 0], h[:, 1]])
                p0 = np.vstack([h, mir, axis])
                p0 = inside_fix(p0 + rng.normal(0.0, 1e-3, size=p0.shape))
            else:
                p0 = sample()
        else:
            # Basin hopping: perturb the incumbent with a random scale
            sigma = 10 ** rng.uniform(-2.3, -1.1)
            p0 = inside_fix(best_p + rng.normal(0.0, sigma, size=best_p.shape))
        if min_area(p0) > best_v:
            best_p, best_v = p0, min_area(p0)
        try:
            p = inside_fix(optimize(p0))
            v = min_area(p)
            # Re-polish promising candidates: the sign pattern is recomputed
            # from the optimized point, so the second pass is a fresh,
            # well-conditioned SLSQP solve from a near-optimal vertex.
            if v > 0.8 * best_v:
                p2 = inside_fix(optimize(p))
                v2 = min_area(p2)
                if np.all(np.isfinite(p2)) and v2 > v:
                    p, v = p2, v2
            if np.all(np.isfinite(p)) and v > best_v:
                best_p, best_v = p, v
        except Exception:
            continue

    return best_p


# EVOLVE-BLOCK-END
