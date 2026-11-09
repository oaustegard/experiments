# EVOLVE-BLOCK-START
import numpy as np


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    n = 11
    s3 = np.sqrt(3.0)
    # Honeycomb lattice (touching hexagons, vertex on +x axis, angle 0)
    pts = [(1.5 * a, s3 * (a / 2.0 + b)) for a in range(-3, 4) for b in range(-3, 4)]
    pts.sort(key=lambda p: (round(p[0] ** 2 + p[1] ** 2, 6), p[1], p[0]))
    # Tiny spread so touching neighbours are strictly disjoint (avoids float-touch overlap)
    centers = np.array(pts[:n]) * (1 + 1e-6)
    inner_hex_data = np.column_stack([centers, np.zeros(n)])

    # All inner vertices
    k = np.arange(6) * np.pi / 3
    verts = (centers[:, None, :] + np.stack([np.cos(k), np.sin(k)], 1)[None]).reshape(-1, 2)
    # Outer hexagon (angle 0) edge normals at 30 + 60*j degrees, apothem = R*sqrt(3)/2
    ang = np.radians(30 + 60 * np.arange(6))
    normals = np.stack([np.cos(ang), np.sin(ang)], 1)

    def side_for(c):
        return np.max((verts - c) @ normals.T) / (s3 / 2.0)

    # Search for the outer center minimizing the side length (coarse-to-fine)
    best_c, best_r, step = np.zeros(2), side_for(np.zeros(2)), 0.5
    for _ in range(40):
        improved = False
        for dx in (-step, 0, step):
            for dy in (-step, 0, step):
                c = best_c + np.array([dx, dy])
                r = side_for(c)
                if r < best_r - 1e-12:
                    best_c, best_r, improved = c, r, True
        if not improved:
            step /= 2.0

    outer_hex_data = np.array([best_c[0], best_c[1], 0])
    outer_hex_side_length = float(best_r) * (1 + 1e-9)  # tiny tolerance

    # ---- Try tilted hexagons via penalty optimisation (safe fallback to lattice) ----
    try:
        import time
        from scipy.optimize import minimize

        t0 = time.time()
        kk = np.arange(6) * np.pi / 3
        iu, ju = np.triu_indices(n, 1)

        def hexv(P):
            a = P[:, 2, None] + kk
            return P[:, None, :2] + np.stack([np.cos(a), np.sin(a)], -1)

        def min_gap(P):
            # SAT: positive max gap along any edge normal => hexagons disjoint
            V = hexv(P)
            al = P[:, 2, None] + np.pi / 6 + np.arange(3) * np.pi / 3
            U = np.stack([np.cos(al), np.sin(al)], -1)
            ax = np.concatenate([U[iu], U[ju]], 1)
            pa = np.einsum('mvd,mad->mav', V[iu], ax)
            pb = np.einsum('mvd,mad->mav', V[ju], ax)
            g = np.maximum(pb.min(-1) - pa.max(-1), pa.min(-1) - pb.max(-1))
            return g.max(1)

        def cont_R(P):
            return np.max(hexv(P) @ normals.T) / (s3 / 2.0)

        def cost(x, w):
            P = x[:33].reshape(n, 3)
            g = np.minimum(min_gap(P), 0.0)
            c = np.maximum(hexv(P) @ normals.T - x[33] * s3 / 2.0, 0.0)
            return x[33] + w * (np.sum(g * g) + np.sum(c * c))

        def finalize(P):
            # Scale centres outward until strictly disjoint, then measure R
            def ok(s):
                Q = P.copy()
                Q[:, :2] *= s
                return min_gap(Q).min() >= 1e-6
            if not ok(1.6):
                return None
            lo, hi = 1.0, 1.6
            for _ in range(40):
                mid = (lo + hi) / 2
                if ok(mid):
                    hi = mid
                else:
                    lo = mid
            Q = P.copy()
            Q[:, :2] *= hi
            return Q, cont_R(Q) * (1 + 1e-9)

        def polish(x):
            # Constrained refinement: minimise R s.t. exact SAT gaps and containment
            cons = [
                {"type": "ineq",
                 "fun": lambda z: min_gap_all(z[:33].reshape(n, 3)) - 1e-6},
                {"type": "ineq",
                 "fun": lambda z: (z[33] * s3 / 2.0 - (hexv(
                     z[:33].reshape(n, 3)) @ normals.T).max(1)).ravel()},
            ]
            g = np.zeros(34)
            g[33] = 1.0
            r = minimize(lambda z: z[33], x, jac=lambda z: g,
                         constraints=cons, method="SLSQP",
                         options={"maxiter": 100, "ftol": 1e-10})
            return r.x

        def min_gap_all(P):
            # Per-pair max separating gap (vector over all pairs)
            V = hexv(P)
            al = P[:, 2, None] + np.pi / 6 + np.arange(3) * np.pi / 3
            U = np.stack([np.cos(al), np.sin(al)], -1)
            ax = np.concatenate([U[iu], U[ju]], 1)
            pa = np.einsum('mvd,mad->mav', V[iu], ax)
            pb = np.einsum('mvd,mad->mav', V[ju], ax)
            gg = np.maximum(pb.min(-1) - pa.max(-1), pa.min(-1) - pb.max(-1))
            return gg.max(1)

        rng = np.random.default_rng(0)
        base = centers / (1 + 1e-6)
        best_total, best_sol = outer_hex_side_length, None
        it = 0
        while time.time() - t0 < 12:
            it += 1
            P0 = np.zeros((n, 3))
            if it % 2:
                P0[:, :2] = base * 1.05 + rng.normal(0, 0.2, (n, 2))
                P0[:, 2] = rng.uniform(-0.5, 0.5, n)
            else:
                rr = 2.6 * np.sqrt(rng.uniform(0, 1, n))
                tt = rng.uniform(0, 2 * np.pi, n)
                P0[:, 0] = rr * np.cos(tt)
                P0[:, 1] = rr * np.sin(tt)
                P0[:, 2] = rng.uniform(0, np.pi / 3, n)
            x = np.concatenate([P0.ravel(), [4.4]])
            for w in (10.0, 100.0, 1000.0, 10000.0):
                x = minimize(cost, x, args=(w,), method="L-BFGS-B",
                             options={"maxiter": 100}).x
            if x[33] < 4.15:
                try:
                    x = polish(x)
                except Exception:
                    pass
            res = finalize(x[:33].reshape(n, 3))
            if res is not None and res[1] < best_total - 1e-6:
                best_total, best_sol = res[1], res[0]
        if best_sol is not None:
            inner_hex_data = np.column_stack(
                [best_sol[:, :2], np.degrees(best_sol[:, 2])])
            outer_hex_data = np.array([0.0, 0.0, 0.0])
            outer_hex_side_length = float(best_total)
    except Exception:
        pass

    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
