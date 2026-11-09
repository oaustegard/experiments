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
    # Tight honeycomb (flat-top) lattice: neighbor distance sqrt(3).
    s = np.sqrt(3.0)
    eps = 1.0001  # tiny gap to avoid touching/overlap numerical issues
    pts = [
        (0, 0),
        (0, s), (0, -s),
        (1.5, s / 2), (-1.5, s / 2), (1.5, -s / 2), (-1.5, -s / 2),
        (0, 2 * s), (0, -2 * s),
        (3, 0), (-3, 0),
    ]
    inner_hex_data = np.array([[x * eps, y * eps, 0.0] for x, y in pts])

    outer_hex_data = np.array([0, 0, 0])  # centered at origin
    # Centers must lie in a hexagon of side R-1; farthest is y = 2*sqrt(3)*eps
    outer_hex_side_length = 5.001

    try:
        res = _optimize(inner_hex_data, outer_hex_side_length)
        if res is not None and res[1] < outer_hex_side_length:
            inner_hex_data, outer_hex_side_length = res
    except Exception:
        pass  # fall back to the valid honeycomb baseline

    return inner_hex_data, outer_hex_data, outer_hex_side_length


def _verts(x, y, t):
    a = np.radians(t)[:, None] + np.radians(60.0) * np.arange(6)[None, :]
    return np.stack([x[:, None] + np.cos(a), y[:, None] + np.sin(a)], axis=-1)


def _valid(data, R, tol=1e-9):
    n = len(data)
    V = _verts(data[:, 0], data[:, 1], data[:, 2])
    nb = np.radians(30.0 + 60.0 * np.arange(6))
    N = np.stack([np.cos(nb), np.sin(nb)], axis=1)
    if np.max(V @ N.T) > R * np.sqrt(3) / 2 - tol:
        return False
    for i in range(n):
        for j in range(i + 1, n):
            ax = []
            for P in (V[i], V[j]):
                e = np.roll(P, -1, axis=0) - P
                ax.append(np.stack([-e[:, 1], e[:, 0]], axis=1))
            A = np.concatenate(ax)
            A /= np.linalg.norm(A, axis=1, keepdims=True)
            pi, pj = V[i] @ A.T, V[j] @ A.T
            gap = np.maximum(pj.min(0) - pi.max(0), pi.min(0) - pj.max(0))
            if gap.max() < tol:
                return False
    return True


def _optimize(init, R0, restarts=10):
    from scipy.optimize import minimize

    n = len(init)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    I = np.array([p[0] for p in pairs])
    J = np.array([p[1] for p in pairs])
    nb = np.radians(30.0 + 60.0 * np.arange(6))
    N = np.stack([np.cos(nb), np.sin(nb)], axis=1)
    m = len(pairs)

    def unpack(z):
        return z[:n], z[n:2 * n], z[2 * n:3 * n], z[3 * n], z[3 * n + 1:3 * n + 1 + m], z[3 * n + 1 + m:]

    def cons(z):
        x, y, t, R, phi, d = unpack(z)
        V = _verts(x, y, t)
        c1 = (R * np.sqrt(3) / 2 - V @ N.T).ravel()
        nv = np.stack([np.cos(phi), np.sin(phi)], axis=1)
        pi = np.einsum("pkc,pc->pk", V[I], nv)
        pj = np.einsum("pkc,pc->pk", V[J], nv)
        return np.concatenate([c1, (d[:, None] - pi).ravel(), (pj - d[:, None]).ravel()])

    rng = np.random.default_rng(0)
    best = None
    for r in range(restarts):
        Rs = R0
        if r < 6 or best is None:
            s = 0.0 if r == 0 else 0.15
            x = init[:, 0] * 0.98 + rng.normal(0, s, n)
            y = init[:, 1] * 0.98 + rng.normal(0, s, n)
            t = init[:, 2] + rng.normal(0, 20 * s, n)
        else:
            # basin hopping: small perturbation of the incumbent best packing
            s = [0.02, 0.04, 0.06][r % 3]
            b = best[0]
            x = b[:, 0] + rng.normal(0, s, n)
            y = b[:, 1] + rng.normal(0, s, n)
            t = b[:, 2] + rng.normal(0, 60 * s, n)
            Rs = best[1] * 1.02
        phi = np.arctan2(y[J] - y[I], x[J] - x[I])
        d = 0.5 * ((x[I] + x[J]) * np.cos(phi) + (y[I] + y[J]) * np.sin(phi))
        z0 = np.concatenate([x, y, t, [Rs], phi, d])
        out = minimize(lambda z: z[3 * n], z0, jac=lambda z: np.eye(len(z))[3 * n],
                       constraints=[{"type": "ineq", "fun": cons}],
                       method="SLSQP", options={"maxiter": 300, "ftol": 1e-12})
        x, y, t, R, _, _ = unpack(out.x)
        k = 1 + 1e-5
        data = np.stack([x * k, y * k, t], axis=1)
        Rk = R * k
        if _valid(data, Rk) and (best is None or Rk < best[1]):
            best = (data, Rk)
    return best


# EVOLVE-BLOCK-END
