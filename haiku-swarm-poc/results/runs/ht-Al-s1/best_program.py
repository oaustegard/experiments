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
    rng = np.random.default_rng(12345)
    idx = np.array(list(combinations(range(n), 3)))
    I, J, K = idx[:, 0], idx[:, 1], idx[:, 2]
    m = len(idx)
    rows = np.arange(m)

    def signed_areas(p):
        x, y = p[:, 0], p[:, 1]
        return 0.5 * ((x[J] - x[I]) * (y[K] - y[I]) - (x[K] - x[I]) * (y[J] - y[I]))

    def min_area(p):
        return float(np.min(np.abs(signed_areas(p))))

    def sample():
        u = rng.random((n, 2))
        flip = u.sum(axis=1) > 1
        u[flip] = 1 - u[flip]
        # map barycentric-like coords to the triangle
        return np.column_stack([u[:, 0] + 0.5 * u[:, 1], (s3 / 2) * u[:, 1]])

    # Linear containment constraints: y>=0, s3*x - y>=0, s3*(1-x) - y>=0
    A_lin = np.zeros((3 * n, 2 * n + 1))
    b_lin = np.zeros(3 * n)
    for i in range(n):
        A_lin[3 * i, 2 * i + 1] = 1.0
        A_lin[3 * i + 1, 2 * i] = s3
        A_lin[3 * i + 1, 2 * i + 1] = -1.0
        A_lin[3 * i + 2, 2 * i] = -s3
        A_lin[3 * i + 2, 2 * i + 1] = -1.0
        b_lin[3 * i + 2] = -s3
    # constraint: A_lin @ z + b_lin >= 0
    b_lin = -b_lin * 0 + np.array([0.0, 0.0, s3] * n)

    best_p, best_v = None, -1.0
    elite = []  # list of (value, points), top distinct local optima
    n_starts = 1800
    snap = []
    for it in range(n_starts + 7):
        polish = (it >= n_starts) and best_p is not None
        if it == n_starts:
            snap = [e[1].copy() for e in elite]
            if not snap:
                snap = [best_p.copy()]
        if polish:
            # final intensive polish of every elite solution (no RNG used)
            if it - n_starts >= len(snap):
                break
            p0 = snap[it - n_starts].copy()
        elif elite and it % 5 >= 2:
            # basin hopping around a randomly chosen elite solution
            p0 = elite[int(rng.integers(0, len(elite)))][1].copy()
            mv = rng.integers(0, 3)
            if mv == 0:
                p0 = p0 + rng.normal(0.0, rng.choice([0.01, 0.03, 0.08]), p0.shape)
            elif mv == 1:
                # resample 1-3 random points entirely
                k = rng.integers(1, 4)
                ids = rng.choice(n, size=k, replace=False)
                p0[ids] = sample()[:k]
            else:
                # move 1-3 points onto the boundary at random positions
                k = rng.integers(1, 4)
                ids = rng.choice(n, size=k, replace=False)
                for i_ in ids:
                    t = rng.random()
                    e = rng.integers(0, 3)
                    A_ = np.array([[0, 0], [1, 0], [0.5, s3 / 2]])
                    a, b = A_[e], A_[(e + 1) % 3]
                    p0[i_] = a + t * (b - a)
            p0[:, 1] = np.clip(p0[:, 1], 0.0, s3 / 2)
            p0[:, 0] = np.clip(p0[:, 0], p0[:, 1] / s3, 1.0 - p0[:, 1] / s3)
        else:
            p0 = sample()
            if it % 2 == 0:
                # mirror-symmetric start about x=0.5: 5 pairs + 1 axis point
                p0[5:10, 0] = 1.0 - p0[:5, 0]
                p0[5:10, 1] = p0[:5, 1]
                p0[10, 0] = 0.5
                p0 = p0 + rng.normal(0.0, 0.01, p0.shape)
                p0[:, 1] = np.clip(p0[:, 1], 0.0, s3 / 2)
                p0[:, 0] = np.clip(p0[:, 0], p0[:, 1] / s3, 1.0 - p0[:, 1] / s3)
        sg = np.sign(signed_areas(p0))
        sg[sg == 0] = 1.0
        t0 = min_area(p0)
        z0 = np.concatenate([p0.ravel(), [t0]])

        def cons_f(z, sg=sg):
            p = z[:-1].reshape(n, 2)
            return sg * signed_areas(p) - z[-1]

        def cons_j(z, sg=sg):
            x = z[:-1].reshape(n, 2)[:, 0]
            y = z[:-1].reshape(n, 2)[:, 1]
            Jm = np.zeros((m, 2 * n + 1))
            Jm[rows, 2 * I] = 0.5 * (y[J] - y[K]) * sg
            Jm[rows, 2 * J] = 0.5 * (y[K] - y[I]) * sg
            Jm[rows, 2 * K] = 0.5 * (y[I] - y[J]) * sg
            Jm[rows, 2 * I + 1] = 0.5 * (x[K] - x[J]) * sg
            Jm[rows, 2 * J + 1] = 0.5 * (x[I] - x[K]) * sg
            Jm[rows, 2 * K + 1] = 0.5 * (x[J] - x[I]) * sg
            Jm[:, -1] = -1.0
            return Jm

        obj_g = np.zeros(2 * n + 1)
        obj_g[-1] = -1.0
        try:
            res = minimize(
                lambda z: -z[-1], z0, jac=lambda z: obj_g, method="SLSQP",
                constraints=[
                    {"type": "ineq", "fun": cons_f, "jac": cons_j},
                    {"type": "ineq", "fun": lambda z: A_lin @ z + b_lin,
                     "jac": lambda z: A_lin},
                ],
                options={"maxiter": 1000 if polish else 150, "ftol": 1e-13},
            )
            p = res.x[:-1].reshape(n, 2)
        except Exception:
            continue
        # project into triangle for strict feasibility
        p[:, 1] = np.clip(p[:, 1], 0.0, None)
        p[:, 0] = np.clip(p[:, 0], p[:, 1] / s3, 1.0 - p[:, 1] / s3)
        v = min_area(p)
        if v > best_v:
            best_v, best_p = v, p.copy()
        if v > 0.9 * best_v and all(abs(v - e[0]) > 1e-7 for e in elite):
            elite.append((v, p.copy()))
            elite.sort(key=lambda e: -e[0])
            elite = elite[:6]

    if best_p is None:
        best_p = sample()
    return best_p


# EVOLVE-BLOCK-END
