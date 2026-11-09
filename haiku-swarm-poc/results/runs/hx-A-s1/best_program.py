# EVOLVE-BLOCK-START
import time
import numpy as np

_N = 11
_ANG = np.radians(60 * np.arange(6))
_a = np.radians(30 + 60 * np.arange(6))
_NRM = np.stack([np.cos(_a), np.sin(_a)], 1)  # outer-edge normals (6,2)
_N3 = np.radians(30 + 60 * np.arange(3))  # edge-normal directions of a hexagon
_I, _J = np.triu_indices(_N, 1)  # 55 pairs
_NP = len(_I)
_H = np.sqrt(3) / 2


def _verts(P):
    a = P[:, 2, None] + _ANG[None]
    return np.stack([P[:, 0, None] + np.cos(a), P[:, 1, None] + np.sin(a)], -1)


def _hw(ang, th):
    """Half-width of a unit hexagon (rotation th) along axis angles ang (m,k)."""
    return np.cos(ang[:, :, None] - th[:, None, None] - _ANG[None, None, :]).max(2)


def _sat(P):
    """Per pair: overlap along each of the 6 candidate axes (neg = separated)."""
    d = P[_J, :2] - P[_I, :2]
    ang = np.concatenate(
        [P[_I, 2, None] + _N3[None], P[_J, 2, None] + _N3[None]], 1
    )  # (55,6)
    proj = np.cos(ang) * d[:, 0, None] + np.sin(ang) * d[:, 1, None]
    h = _hw(ang, P[_I, 2]) + _hw(ang, P[_J, 2])
    return ang, proj, h


def _pen(x, R):
    """Penalty: squared containment violation + squared SAT overlap depth."""
    P = x.reshape(_N, 3)
    viol = np.maximum(_verts(P) @ _NRM.T - R * _H, 0.0)
    _, proj, h = _sat(P)
    ov = np.maximum((h - np.abs(proj)).min(1), 0.0)
    return (viol**2).sum() + (ov**2).sum()


def _cons(z):
    """All entries must be >= 0 for a valid packing."""
    P = z[:33].reshape(_N, 3)
    R = z[33]
    phi = z[34 : 34 + _NP]
    c = z[34 + _NP : 34 + 2 * _NP]
    V = _verts(P)  # (11,6,2)
    cont = (R * _H - V @ _NRM.T).ravel()
    n = np.stack([np.cos(phi), np.sin(phi)], 1)  # (55,2)
    pi = np.einsum("pvk,pk->pv", V[_I], n)
    pj = np.einsum("pvk,pk->pv", V[_J], n)
    return np.concatenate([cont, (c[:, None] - pi).ravel(), (pj - c[:, None]).ravel()])


def _init_sep(P):
    """Initial separating lines (phi, c) from the best SAT axis of each pair."""
    ang, proj, h = _sat(P)
    k = np.argmin(h - np.abs(proj), 1)
    r = np.arange(_NP)
    a = ang[r, k]
    flip = proj[r, k] < 0
    a = np.where(flip, a + np.pi, a)
    n = np.stack([np.cos(a), np.sin(a)], 1)
    sep = np.abs(proj[r, k]) - h[r, k]  # may be negative if still overlapping
    c = np.einsum("pk,pk->p", n, P[_I, :2]) + _hw(a[:, None], P[_I, 2])[:, 0] + sep / 2
    return a, c


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    # Honeycomb packing: unit hexagons (vertices at 0, 60, ... deg) tile the
    # plane with centers on a triangular lattice of spacing sqrt(3).
    # Scale lattice by s slightly > 1 so neighbours don't touch exactly
    # (avoids floating-point overlap flags).
    s = 1.00005
    d = np.sqrt(3) * s
    centers = [(0.0, 0.0)]
    # 6 nearest neighbours (distance sqrt(3), directions 30, 90, ...)
    for k in range(6):
        a = np.radians(30 + 60 * k)
        centers.append((d * np.cos(a), d * np.sin(a)))
    # 4 of the 6 lattice points at distance 3 (directions 0, 60, 180, 240)
    for k in (0, 1, 3, 4):
        a = np.radians(60 * k)
        centers.append((3 * s * np.cos(a), 3 * s * np.sin(a)))

    inner_hex_data = np.array([[x, y, 0.0] for x, y in centers])

    outer_hex_data = np.array([0.0, 0.0, 0.0])  # same orientation, at origin
    # With aligned hexagons, centers must lie in a concentric hexagon of
    # side R-1. The farthest centers are at distance 3*s along vertex
    # directions, so R >= 1 + 3*s. Add a tiny margin.
    outer_hex_side_length = 1 + 3 * s + 0.0001  # ~4.00025

    # ---- Global search (random multistart) then basin-hopping polish.
    # Only accepted if strictly feasible and better than the honeycomb above.
    try:
        from scipy.optimize import minimize

        rng = np.random.default_rng(1)
        base = inner_hex_data[:, :2]
        best_R, best_P = outer_hex_side_length, None
        best_z = None
        e = np.zeros(34 + 2 * _NP)
        e[33] = 1.0
        budget = 70.0
        t0 = time.time()
        trial = 0
        while time.time() - t0 < budget:
            elapsed = time.time() - t0
            if trial == 0:  # honeycomb seed
                P0 = np.column_stack([base * 1.03, np.zeros(_N)])
                R0 = 4.3
            elif best_P is None or elapsed < 0.4 * budget:
                # exploration: scattered centres, random tilts
                r = 2.9 * np.sqrt(rng.random(_N))
                t = rng.random(_N) * 2 * np.pi
                P0 = np.column_stack(
                    [r * np.cos(t), r * np.sin(t), rng.random(_N) * np.pi / 3]
                )
                R0 = 4.3
            else:
                # exploitation: perturb the incumbent (basin hopping)
                sc = rng.choice([0.005, 0.015, 0.03, 0.08, 0.15])
                P0 = best_P + rng.normal(0, 1, best_P.shape) * np.array(
                    [sc, sc, 2 * sc]
                )
                R0 = best_R * 1.03
            trial += 1
            # Phase 1: cheap penalty relaxation at a generous R.
            rel = minimize(
                _pen,
                P0.ravel(),
                args=(R0,),
                method="L-BFGS-B",
                options={"maxiter": 150},
            )
            P1 = rel.x.reshape(_N, 3)
            phi0, c0 = _init_sep(P1)
            z0 = np.concatenate([P1.ravel(), [R0], phi0, c0])
            # Phase 2: exact SAT-line formulation, minimise R.
            res = minimize(
                lambda z: z[33],
                z0,
                jac=lambda z: e,
                constraints=[{"type": "ineq", "fun": _cons}],
                method="SLSQP",
                options={"maxiter": 300, "ftol": 1e-12},
            )
            z = res.x
            if _cons(z).min() > -1e-9 and z[33] < best_R:
                best_R, best_P = z[33], z[:33].reshape(_N, 3).copy()
                best_z = z.copy()

        # Final polish: long SLSQP run from the exact best state (keeps its
        # separating lines), accepted only if still strictly feasible and better.
        if best_z is not None:
            res = minimize(
                lambda z: z[33],
                best_z,
                jac=lambda z: e,
                constraints=[{"type": "ineq", "fun": _cons}],
                method="SLSQP",
                options={"maxiter": 600, "ftol": 1e-14},
            )
            z = res.x
            if _cons(z).min() > -1e-9 and z[33] < best_R:
                best_R, best_P = z[33], z[:33].reshape(_N, 3).copy()

        if best_P is not None:
            # Inflate centres (keeps separating lines valid) and R for float safety.
            P = best_P.copy()
            P[:, :2] *= 1 + 2e-5
            P[:, 2] = np.degrees(P[:, 2])
            R = best_R * (1 + 3e-5)
            if R < outer_hex_side_length:
                inner_hex_data, outer_hex_side_length = P, R
    except Exception:
        pass

    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
