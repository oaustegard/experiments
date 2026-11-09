# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations
from scipy.optimize import minimize

S3 = np.sqrt(3.0)


def _areas(p, I, J, K):
    return 0.5 * ((p[J, 0] - p[I, 0]) * (p[K, 1] - p[I, 1])
                  - (p[K, 0] - p[I, 0]) * (p[J, 1] - p[I, 1]))


def _random_in_triangle(rng, n):
    u = rng.random((n, 2))
    flip = u.sum(axis=1) > 1
    u[flip] = 1 - u[flip]
    return u[:, 0:1] * np.array([1.0, 0.0]) + u[:, 1:2] * np.array([0.5, S3 / 2])


def _project(p):
    p = p.copy()
    p[:, 1] = np.clip(p[:, 1], 0, S3 / 2)
    for _ in range(3):
        p[:, 1] = np.maximum(p[:, 1], 0)
        # left edge: y <= S3*x ; right edge: y <= S3*(1-x)
        for a in range(len(p)):
            x, y = p[a]
            if y > S3 * x:
                # project onto line S3*x - y = 0
                d = (S3 * x - y) / 4.0
                x, y = x - S3 * d, y + d
            if y > S3 * (1 - x):
                d = (S3 * (1 - x) - y) / 4.0
                x, y = x + S3 * d, y + d
            p[a] = (x, max(y, 0.0))
    return p


def _optimize(p0, I, J, K, n):
    sgn = np.sign(_areas(p0, I, J, K))
    sgn[sgn == 0] = 1.0
    m = len(I)
    rows = np.arange(m)

    def cons_tri(z):
        p = z[:-1].reshape(n, 2)
        return sgn * _areas(p, I, J, K) - z[-1]

    def jac_tri(z):
        p = z[:-1].reshape(n, 2)
        Jm = np.zeros((m, 2 * n + 1))
        xi, yi = p[I, 0], p[I, 1]
        xj, yj = p[J, 0], p[J, 1]
        xk, yk = p[K, 0], p[K, 1]
        Jm[rows, 2 * I] = 0.5 * (yj - yk)
        Jm[rows, 2 * J] = 0.5 * (yk - yi)
        Jm[rows, 2 * K] = 0.5 * (yi - yj)
        Jm[rows, 2 * I + 1] = 0.5 * (xk - xj)
        Jm[rows, 2 * J + 1] = 0.5 * (xi - xk)
        Jm[rows, 2 * K + 1] = 0.5 * (xj - xi)
        Jm *= sgn[:, None]
        Jm[:, -1] = -1.0
        return Jm

    def cons_box(z):
        p = z[:-1].reshape(n, 2)
        return np.concatenate([p[:, 1],
                               S3 * p[:, 0] - p[:, 1],
                               S3 * (1 - p[:, 0]) - p[:, 1]])

    Bj = np.zeros((3 * n, 2 * n + 1))
    for a in range(n):
        Bj[a, 2 * a + 1] = 1
        Bj[n + a, 2 * a] = S3
        Bj[n + a, 2 * a + 1] = -1
        Bj[2 * n + a, 2 * a] = -S3
        Bj[2 * n + a, 2 * a + 1] = -1

    z0 = np.concatenate([p0.ravel(), [0.0]])
    z0[-1] = np.min(sgn * _areas(p0, I, J, K))
    res = minimize(lambda z: -z[-1], z0, jac=lambda z: np.eye(2 * n + 1)[-1] * -1.0,
                   constraints=[{'type': 'ineq', 'fun': cons_tri, 'jac': jac_tri},
                                {'type': 'ineq', 'fun': cons_box, 'jac': lambda z: Bj}],
                   method='SLSQP', options={'maxiter': 300, 'ftol': 1e-13})
    return res.x[:-1].reshape(n, 2)


def heilbronn_triangle11() -> np.ndarray:
    """
    Construct an arrangement of n points on or inside the unit equilateral triangle in order to
    maximize the area of the smallest triangle formed by these points. Here n = 11.

    Uses multi-start SLSQP (max-min formulation) with a fixed seed.

    Returns:
        points: np.ndarray of shape (11,2) with the x,y coordinates of the points.
    """
    n = 11
    rng = np.random.default_rng(12345)
    combos = np.array(list(combinations(range(n), 3)))
    I, J, K = combos[:, 0], combos[:, 1], combos[:, 2]

    best, best_val = None, -1.0
    for _ in range(400):
        try:
            p0 = _random_in_triangle(rng, n)
            p = _project(_optimize(p0, I, J, K, n))
            val = np.min(np.abs(_areas(p, I, J, K)))
            # polish: re-linearize with signs taken from the current solution
            if np.isfinite(val) and val > 0.8 * best_val:
                for _ in range(2):
                    q = _project(_optimize(p, I, J, K, n))
                    qv = np.min(np.abs(_areas(q, I, J, K)))
                    if qv > val:
                        p, val = q, qv
                    else:
                        break
            if np.isfinite(val) and val > best_val:
                best, best_val = p, val
        except Exception:
            continue
    if best is None:
        best = _random_in_triangle(rng, n)
        best_val = -1.0

    # Basin hopping: perturb/relocate points so triangle orientations flip,
    # then re-optimize in the new sign chamber.
    scales = [0.01, 0.03, 0.06]
    cur, cur_val = best, best_val
    for it in range(400):
        try:
            if it % 40 == 39:
                cur, cur_val = best, best_val  # restart from global best
            if it % 4 == 3:
                p0 = cur.copy()
                k = 1 + int(rng.integers(0, 2))
                idx = rng.choice(n, size=k, replace=False)
                p0[idx] = _random_in_triangle(rng, k)
            else:
                s = scales[it % len(scales)]
                p0 = _project(cur + rng.normal(0.0, s, cur.shape))
            p = _project(_optimize(p0, I, J, K, n))
            val = np.min(np.abs(_areas(p, I, J, K)))
            if not np.isfinite(val):
                continue
            if val > best_val:
                best, best_val = p, val
            if val > 0.97 * cur_val:
                cur, cur_val = p, val
        except Exception:
            continue
    return best


# EVOLVE-BLOCK-END
