# EVOLVE-BLOCK-START
import numpy as np


def heilbronn_triangle11() -> np.ndarray:
    """
    Construct an arrangement of n points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 11.

    Returns:
        points: np.ndarray of shape (11,2) with the x,y coordinates of the points.
    """
    n = 11
    s3 = np.sqrt(3.0)
    rng = np.random.default_rng(12345)
    tri = np.array([(i, j, k) for i in range(n) for j in range(i + 1, n)
                    for k in range(j + 1, n)])
    I, J, K = tri[:, 0], tri[:, 1], tri[:, 2]

    def signed_areas(p):
        x, y = p[:, 0], p[:, 1]
        return 0.5 * ((x[J] - x[I]) * (y[K] - y[I]) - (x[K] - x[I]) * (y[J] - y[I]))

    def min_area(p):
        return np.min(np.abs(signed_areas(p)))

    def sample(m):
        # uniform points in the triangle (sqrt barycentric trick)
        u = rng.random((m, 2))
        r1 = np.sqrt(u[:, 0])
        r2 = u[:, 1]
        b, c = r1 * (1 - r2), r1 * r2
        x = b * 1.0 + c * 0.5
        y = c * (s3 / 2)
        # Mix in boundary points: good configurations use the edges heavily.
        onb = rng.random(m) < 0.5
        e = rng.integers(0, 3, m)
        t = rng.random(m)
        bx = np.where(e == 0, t, np.where(e == 1, 1.0 - 0.5 * t, 0.5 * t))
        by = np.where(e == 0, 0.0, (s3 / 2) * t)
        x = np.where(onb, bx, x)
        y = np.where(onb, by, y)
        return np.stack([x, y], axis=1)

    def sample_sym():
        # mirror-symmetric about x = 0.5: 5 pairs + 1 point on the axis
        h = sample(5)
        h[:, 0] = np.where(h[:, 0] > 0.5, 1.0 - h[:, 0], h[:, 0])
        mir = h.copy()
        mir[:, 0] = 1.0 - mir[:, 0]
        ax = np.array([[0.5, rng.random() * s3 / 2]])
        return np.vstack([h, mir, ax])

    # Stage 1: random search for good starting configurations
    cands = []
    for q in range(10000):
        p = sample_sym() if q % 2 == 0 else sample(n)
        cands.append((min_area(p), p))
    cands.sort(key=lambda t: -t[0])
    starts = [c[1] for c in cands[:300]]
    best_p, best_v = cands[0][1].copy(), cands[0][0]

    # Stage 2: SLSQP refinement (maximize t s.t. sign*area >= t)
    try:
        from scipy.optimize import minimize

        rows = np.arange(len(tri))

        # inside-triangle linear constraints: y>=0, s3*x-y>=0, s3*(1-x)-y>=0
        G = np.zeros((3 * n, 2 * n + 1))
        h = np.zeros(3 * n)
        for i in range(n):
            G[3 * i, 2 * i + 1] = 1.0
            G[3 * i + 1, 2 * i] = s3
            G[3 * i + 1, 2 * i + 1] = -1.0
            G[3 * i + 2, 2 * i] = -s3
            G[3 * i + 2, 2 * i + 1] = -1.0
            h[3 * i + 2] = s3
        obj_grad = np.zeros(2 * n + 1)
        obj_grad[-1] = -1.0

        def refine(p0):
            sg = np.sign(signed_areas(p0))
            sg[sg == 0] = 1.0
            z0 = np.concatenate([p0.ravel(), [min_area(p0)]])

            def cons_area(z):
                p = z[:-1].reshape(n, 2)
                return sg * signed_areas(p) - z[-1]

            def jac_area(z):
                p = z[:-1].reshape(n, 2)
                x, y = p[:, 0], p[:, 1]
                Jm = np.zeros((len(tri), 2 * n + 1))
                Jm[rows, 2 * I] = 0.5 * (y[J] - y[K])
                Jm[rows, 2 * I + 1] = 0.5 * (x[K] - x[J])
                Jm[rows, 2 * J] = 0.5 * (y[K] - y[I])
                Jm[rows, 2 * J + 1] = -0.5 * (x[K] - x[I])
                Jm[rows, 2 * K] = -0.5 * (y[J] - y[I])
                Jm[rows, 2 * K + 1] = 0.5 * (x[J] - x[I])
                Jm *= sg[:, None]
                Jm[:, -1] = -1.0
                return Jm

            cons = [
                {"type": "ineq", "fun": cons_area, "jac": jac_area},
                {"type": "ineq", "fun": lambda z: G @ z + h, "jac": lambda z: G},
            ]
            res = minimize(lambda z: -z[-1], z0, jac=lambda z: obj_grad,
                           constraints=cons, method="SLSQP",
                           options={"maxiter": 300, "ftol": 1e-13})
            return res.x[:-1].reshape(n, 2)

        cen = np.array([0.5, s3 / 6])
        elites = []  # (value, points) pool of good local optima

        def consider(p0):
            nonlocal best_p, best_v
            try:
                p = refine(p0)
                if not np.all(np.isfinite(p)):
                    return
                # polish: re-linearize with updated orientation signs
                for _ in range(3):
                    p2 = refine(p)
                    if np.all(np.isfinite(p2)) and min_area(p2) > min_area(p):
                        p = p2
                    else:
                        break
                # enforce containment with the smallest feasible shrink
                q = None
                for eps in (0.0, 1e-15, 1e-14, 1e-13, 1e-12, 1e-11):
                    qq = cen + (p - cen) * (1 - eps)
                    viol = min(np.min(qq[:, 1]),
                               np.min(s3 * qq[:, 0] - qq[:, 1]),
                               np.min(s3 * (1 - qq[:, 0]) - qq[:, 1]))
                    if viol >= 0:
                        q = qq
                        break
                if q is None:
                    return
                v = min_area(q)
                elites.append((v, q))
                elites.sort(key=lambda t: -t[0])
                del elites[8:]
                if v > best_v:
                    best_v, best_p = v, q
            except Exception:
                return

        for p0 in starts:
            consider(p0)

        # Stage 3: basin hopping around incumbent best and elite pool
        for it in range(800):
            sigma = (0.01, 0.02, 0.04, 0.08)[it % 4]
            base = best_p if (it % 3 == 0 or not elites) else \
                elites[rng.integers(len(elites))][1]
            p0 = base.copy()
            k = int(rng.integers(1, 4))
            idx = rng.choice(n, size=k, replace=False)
            if it % 5 == 4:
                p0[idx] = sample(k)
            else:
                p0[idx] += rng.normal(0.0, sigma * 2.0, (k, 2))
            p0 = cen + (p0 - cen) * 0.98  # pull back toward the interior
            consider(p0)
    except Exception:
        pass

    return np.asarray(best_p, dtype=float)


# EVOLVE-BLOCK-END
