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
    gap = 1.0005  # tiny spacing so hexagons do not touch exactly
    # Lattice of touching unit hexagons (vertices at angle 0): neighbor
    # centers are sqrt(3) apart in directions 30 + 60k degrees.
    a = s3 * gap * np.array([np.cos(np.pi / 6), np.sin(np.pi / 6)])
    b = s3 * gap * np.array([np.cos(np.pi / 2), np.sin(np.pi / 2)])
    # Outer hexagon apothem directions
    U = np.array([[np.cos(np.pi / 6 + k * np.pi / 3), np.sin(np.pi / 6 + k * np.pi / 3)] for k in range(6)])

    cells = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            if abs(i + j) <= 2:
                c = i * a + j * b
                cells.append((np.max(U @ c), np.hypot(*c), c))
    cells.sort(key=lambda t: (round(t[0], 9), t[1]))
    centers = np.array([c for _, _, c in cells[:n]])

    inner_hex_data = np.column_stack([centers, np.zeros(n)])

    outer_hex_data = np.array([0, 0, 0])  # centered at origin
    # Inner vertex reaches sqrt(3)/2 beyond its center along an apothem direction
    max_proj = np.max(centers @ U.T)
    outer_hex_side_length = float(max_proj / (s3 / 2) + 1.0 + 1e-6)

    # ---- Local optimization with tilted hexagons (falls back to lattice) ----
    from scipy.optimize import minimize

    k6 = np.arange(6) * np.pi / 3
    iu = np.triu_indices(n, 1)

    def verts(z):
        t = z[:, 2][:, None] + k6[None, :]
        return np.stack([z[:, 0:1] + np.cos(t), z[:, 1:2] + np.sin(t)], axis=2)  # (n,6,2)

    def pair_pen(V, i, j):
        # SAT penetration depth between hexagons i, j (0 if separated)
        best = np.inf
        for h in (i, j):
            ang = zs[h, 2] + np.pi / 6 + np.arange(3) * np.pi / 3
            ax = np.stack([np.cos(ang), np.sin(ang)], axis=1)
            p = V[i] @ ax.T
            q = V[j] @ ax.T
            ov = np.minimum(p.max(0) - q.min(0), q.max(0) - p.min(0))
            best = min(best, ov.min())
            if best <= 0:
                return 0.0
        return best

    def penalties(z, R):
        global_z = z
        V = verts(global_z)
        cont = np.maximum(V @ U.T - R * s3 / 2, 0.0)  # (n,6,6)
        c = float(np.sum(cont**2))
        return V, c

    def total_pen(z, R, with_pairs=True):
        global zs
        zs = z
        V, c = penalties(z, R)
        o = 0.0
        for i, j in zip(*iu):
            if np.hypot(*(z[i, :2] - z[j, :2])) < 2.0:
                o += pair_pen(V, i, j) ** 2
        return c + o, o, V

    def max_overlap(z):
        return np.sqrt(total_pen(z, 1e9)[1])

    def req_R(z):
        V = verts(z)
        return float(np.max(V @ U.T) / (s3 / 2))

    rng = np.random.default_rng(0)
    base = np.column_stack([centers / gap, np.zeros(n)])
    best_R, best_z = outer_hex_side_length, inner_hex_data.copy()
    for trial in range(6):
        z0 = base.copy()
        z0[:, :2] *= 0.93 + 0.02 * trial
        z0[:, :2] += rng.normal(0, 0.05, (n, 2))
        z0[:, 2] = rng.uniform(-0.3, 0.3, n)
        x = np.append(z0.ravel(), req_R(z0))
        for mu in (10.0, 100.0, 1000.0):
            f = lambda v: v[-1] + mu * total_pen(v[:-1].reshape(n, 3), v[-1])[0]
            x = minimize(f, x, method="L-BFGS-B", options={"maxiter": 150}).x
        z = x[:-1].reshape(n, 3)
        # Remove residual overlap by gently expanding centers, then fit R exactly
        for _ in range(60):
            if max_overlap(z) <= 1e-12:
                break
            z = z.copy()
            z[:, :2] *= 1.0005
        else:
            continue
        R = req_R(z) + 1e-6
        if R < best_R:
            best_R, best_z = R, z.copy()

    if best_R < outer_hex_side_length:
        inner_hex_data = np.column_stack([best_z[:, :2], np.degrees(best_z[:, 2])])
        outer_hex_side_length = float(best_R)

    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
