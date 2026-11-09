# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

N = 11
S3 = np.sqrt(3.0)
_I, _J = np.triu_indices(N, 1)
_K6 = np.arange(6) * np.pi / 3.0
_OUTER_N = np.stack([np.cos(np.pi / 6 + _K6), np.sin(np.pi / 6 + _K6)], axis=1)  # (6,2)
_TIME_BUDGET = 15.0


def _geom(P):
    """P: (N,3) rows (x, y, theta_rad) -> vertices (N,6,2), edge normals (N,6,2)."""
    t = P[:, 2:3] + _K6[None, :]
    V = np.stack([P[:, 0:1] + np.cos(t), P[:, 1:2] + np.sin(t)], axis=2)
    NN = np.stack([np.cos(t + np.pi / 6), np.sin(t + np.pi / 6)], axis=2)
    return V, NN


def _overlap_min(P):
    """For each pair, the minimum (over separating axes) overlap depth."""
    V, NN = _geom(P)
    A = np.concatenate([NN[_I], NN[_J]], axis=1)  # (55,12,2)
    PI = np.einsum("pvd,pad->pav", V[_I], A)
    PJ = np.einsum("pvd,pad->pav", V[_J], A)
    ov = np.minimum(PI.max(2), PJ.max(2)) - np.maximum(PI.min(2), PJ.min(2))
    return ov.min(axis=1)


def _penalties(P, R):
    """Penetration-squared overlap penalty (SAT) and containment penalty."""
    pen = _overlap_min(P)
    pov = float(np.sum(np.maximum(pen, 0.0) ** 2))
    V, _ = _geom(P)
    m = (V @ _OUTER_N.T).max(axis=2)  # (N,6)
    pc = float(np.sum(np.maximum(m - R * S3 / 2.0, 0.0) ** 2))
    return pov, pc


def _required_R(P):
    """Exact minimal outer side (outer hexagon centered at origin, angle 0)."""
    V, _ = _geom(P)
    m = (V @ _OUTER_N.T).max()
    return m / (S3 / 2.0)


def _max_pen_ok(P):
    return bool(np.all(_overlap_min(P) <= 0.0))


def _repair(P):
    """Fine, geometrically growing outward scaling of centers until strictly separated."""
    P = P.copy()
    eps = 1e-8
    for _ in range(120):
        if _max_pen_ok(P):
            return P
        P[:, :2] *= 1.0 + eps
        eps *= 1.5
    return None


def _lattice_fallback():
    u1 = S3 * np.array([np.cos(np.pi / 6), np.sin(np.pi / 6)])
    u2 = S3 * np.array([np.cos(np.pi / 2), np.sin(np.pi / 2)])
    pts = []
    for m in range(-4, 5):
        for n in range(-4, 5):
            p = m * u1 + n * u2
            pts.append((np.hypot(*p), p))
    pts.sort(key=lambda t: t[0])
    P = np.zeros((N, 3))
    for i in range(N):
        P[i, :2] = pts[i][1]
    return P


def _relax(P0, R0, mus, maxiter):
    """Penalty continuation with L-BFGS over (centers, angles, R)."""
    z = np.concatenate([P0.ravel(), [R0]])

    for mu in mus:
        def f(zv):
            P = zv[:-1].reshape(N, 3)
            Rv = zv[-1]
            pov, pc = _penalties(P, Rv)
            return Rv + mu * (pov + pc)

        res = minimize(f, z, method="L-BFGS-B", options={"maxiter": maxiter})
        z = res.x

    P = z[:-1].reshape(N, 3).copy()
    P[:, 2] = np.mod(P[:, 2], np.pi / 3)
    return P


def _finalize(P):
    """Repair to a strictly valid packing and return (P, exact R) or (None, inf)."""
    if P is None:
        return None, np.inf
    Q = _repair(P)
    if Q is None:
        return None, np.inf
    return Q, _required_R(Q)


def _random_start(rng):
    P0 = np.zeros((N, 3))
    for i in range(N):
        r = rng.uniform(0, 3.0)
        a = rng.uniform(0, 2 * np.pi)
        P0[i, 0] = r * np.cos(a)
        P0[i, 1] = r * np.sin(a)
        P0[i, 2] = rng.choice([0.0, np.pi / 6, rng.uniform(0, np.pi / 3)])
    return P0


def _perturb(P, R, rng):
    """Relocate 1-2 hexagons to random spots inside the outer hexagon; re-randomize one angle."""
    Q = P.copy()
    k = int(rng.integers(1, 3))
    idx = rng.choice(N, size=k, replace=False)
    rad = 0.75 * R
    for i in idx:
        r = rad * np.sqrt(rng.uniform())
        a = rng.uniform(0, 2 * np.pi)
        Q[i, 0] = r * np.cos(a)
        Q[i, 1] = r * np.sin(a)
        Q[i, 2] = rng.uniform(0, np.pi / 3)
    j = int(rng.integers(0, N))
    Q[j, 2] = rng.uniform(0, np.pi / 3)
    return Q


def _optimize(rng, deadline):
    best_P = _lattice_fallback()
    best_R = _required_R(best_P)

    # Initial random restarts with the full penalty schedule.
    for _ in range(2):
        if time.time() > deadline:
            break
        P0 = _random_start(rng)
        P = _relax(P0, 5.0, [10.0, 1e2, 1e3, 1e4], 150)
        Q, R = _finalize(P)
        if Q is not None and R < best_R:
            best_P, best_R = Q, R

    # Basin hopping from the best packing found so far.
    while time.time() < deadline:
        cand = _perturb(best_P, best_R, rng)
        P = _relax(cand, best_R * 1.02, [1e2, 1e3, 1e4, 1e5], 80)
        Q, R = _finalize(P)
        if Q is not None and R < best_R - 1e-9:
            best_P, best_R = Q, R

    return best_P, best_R


def hexagon_packing_11():
    """
    Returns
        inner_hex_data: (11,3) array of (x, y, angle_degrees)
        outer_hex_data: (3,) array (x, y, angle_degrees)
        outer_hex_side_length: float
    """
    start = time.time()
    deadline = start + _TIME_BUDGET

    rng = np.random.default_rng(12345)
    best_P, best_R = _optimize(rng, deadline)

    inner_hex_data = np.zeros((N, 3))
    inner_hex_data[:, 0] = best_P[:, 0]
    inner_hex_data[:, 1] = best_P[:, 1]
    inner_hex_data[:, 2] = np.degrees(best_P[:, 2])
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    outer_hex_side_length = float(best_R)

    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
