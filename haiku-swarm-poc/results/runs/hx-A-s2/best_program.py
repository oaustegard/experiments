# EVOLVE-BLOCK-START
import numpy as np


def _verts(P):
    """Vertices (n,6,2) of unit hexagons; P rows are (x, y, angle_deg)."""
    a = np.radians(P[:, 2])[:, None] + np.arange(6) * np.pi / 3
    return np.stack([P[:, 0:1] + np.cos(a), P[:, 1:2] + np.sin(a)], -1)


def _gaps(P, ii, jj):
    """Max separating-axis gap for each pair (>0 separated, <0 overlapping)."""
    t = np.radians(P[:, 2])
    d = P[jj, :2] - P[ii, :2]
    k = np.arange(6) * np.pi / 3
    best = np.full(len(ii), -np.inf)
    for src in (ii, jj):
        for q in range(3):
            phi = t[src] + np.pi / 6 + q * np.pi / 3
            u = np.stack([np.cos(phi), np.sin(phi)], -1)
            h = sum(np.max(np.cos(t[m][:, None] + k - phi[:, None]), axis=1) for m in (ii, jj))
            best = np.maximum(best, np.abs((d * u).sum(-1)) - h)
    return best


def _need_R(P):
    """Smallest flat-top outer side length (centered at origin) containing all vertices."""
    ang = np.pi / 6 + np.arange(3) * np.pi / 3
    N = np.stack([np.cos(ang), np.sin(ang)], -1)
    return np.abs(_verts(P) @ N.T).max() * 2 / np.sqrt(3)


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
    k = 1.0 + 1e-6  # tiny safety margin so touching edges are not flagged as overlaps
    a = np.array([s3, 0.0]) * k  # lattice basis (pointy-top hexagons share edges)
    b = np.array([s3 / 2, 1.5]) * k
    off = 4.0 / 3.0  # lattice-coordinate offset of the outer hexagon's center

    # 12 lattice points fit in the tight hexagonal window; keep 11 of them
    pts = [(i, j) for i in range(4) for j in range(4) if 1 <= i + j <= 4][:n]
    inner_hex_data = np.array(
        [list((i - off) * a + (j - off) * b) + [30.0] for i, j in pts]
    )

    outer_hex_data = np.array([0.0, 0.0, 0.0])  # flat-top outer hexagon at origin
    # containment: c.n <= R*sqrt(3)/2 - 1 with max c.n = 2.5*k  ->  R = 2*(1+2.5k)/sqrt(3)
    outer_hex_side_length = 2.0 * (1.0 + 2.5 * k) / s3

    # Numerical refinement (SLSQP) from the lattice start and seeded random starts.
    # A candidate is accepted only if verified separated and strictly smaller R.
    try:
        from scipy.optimize import minimize
        import time

        ii, jj = np.triu_indices(n, 1)
        a6 = np.pi / 6 + np.arange(6) * np.pi / 3
        N6 = np.stack([np.cos(a6), np.sin(a6)], -1)

        def cons(z):
            P = z[:-1].reshape(n, 3)
            # 2e-6 safety margin on pair gaps so solutions pass 1e-6 verification
            return np.concatenate([z[-1] * s3 / 2 - (_verts(P) @ N6.T).ravel(), _gaps(P, ii, jj) - 2e-6])

        rng = np.random.default_rng(0)
        starts = [inner_hex_data.copy()]
        for _ in range(40):
            r = 2.8 * np.sqrt(rng.random(n))
            ph = 2 * np.pi * rng.random(n)
            starts.append(np.stack([r * np.cos(ph), r * np.sin(ph), rng.uniform(0, 60, n)], -1))

        deadline = time.time() + 75.0  # wall-clock budget for the whole refinement

        def attempt(P0, maxiter):
            """SLSQP from P0, repair, verify. Returns (R, P) or None."""
            z0 = np.append(P0.ravel(), _need_R(P0))
            res = minimize(
                lambda z: z[-1], z0,
                jac=lambda z: np.eye(1, len(z), len(z) - 1)[0],
                constraints=[{"type": "ineq", "fun": cons}],
                method="SLSQP", options={"maxiter": maxiter, "ftol": 1e-12},
            )
            P = res.x[:-1].reshape(n, 3).copy()
            for t in range(40):  # repair tiny overlaps by expanding centers (tiny geometric steps)
                if _gaps(P, ii, jj).min() >= 1e-6:
                    break
                P[:, :2] *= 1.0 + 2e-6 * 1.5 ** t
            if _gaps(P, ii, jj).min() < 1e-6:
                return None
            return _need_R(P) + 1e-7, P

        best_R, best_P = outer_hex_side_length, inner_hex_data.copy()

        def consider(out):
            nonlocal best_R, best_P
            if out is not None and out[0] < best_R:
                best_R, best_P = out

        # Stage 1: global exploration from lattice + random starts
        stage1_end = time.time() + 30.0
        pool = []
        for P0 in starts:
            if time.time() > stage1_end:
                break
            out = attempt(P0, 200)
            if out is not None:
                pool.append(out)
            consider(out)
        pool.sort(key=lambda o: o[0])

        # Stage 2: long polish of the top few distinct basins (SLSQP often stops short)
        for out in pool[:3]:
            if time.time() > deadline:
                break
            consider(attempt(out[1].copy(), 600))

        # Stage 3: basin hopping around the incumbent with cycling noise scales
        for r in range(10000):  # runs until the wall-clock deadline
            if time.time() > deadline:
                break
            sig = (0.01, 0.03, 0.08, 0.15)[r % 4]
            P0 = best_P.copy()
            P0[:, :2] += rng.normal(0.0, sig, (n, 2))
            P0[:, 2] += rng.normal(0.0, sig * 30.0, n)
            consider(attempt(P0, 300))

        inner_hex_data, outer_hex_side_length = best_P, best_R
    except Exception:
        pass  # keep the verified lattice packing

    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
