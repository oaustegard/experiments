# EVOLVE-BLOCK-START
import time

import numpy as np
from scipy.optimize import minimize

_H = np.sqrt(3) / 2  # apothem of a unit hexagon
_N = np.array([[np.cos(np.radians(30 + 60 * k)), np.sin(np.radians(30 + 60 * k))] for k in range(6)])
_V60 = np.radians(60 * np.arange(6))


def _verts(P):
    """P: (n,3) with angle in radians -> vertices (n,6,2)."""
    a = P[:, 2, None] + _V60[None]
    return P[:, None, :2] + np.stack([np.cos(a), np.sin(a)], -1)


def _pairs(P, cut):
    """For each pair closer than `cut`, pick the best separating edge axis.
    Returns list of (separation, owner, other, edge_index)."""
    V = _verts(P)
    n = len(P)
    out = []
    for a in range(n):
        for b in range(a + 1, n):
            if np.hypot(*(P[a, :2] - P[b, :2])) > cut:
                continue
            best = (-1e9, 0, 0, 0)
            for o, t in ((a, b), (b, a)):
                th = P[o, 2] + np.radians(30 + 60 * np.arange(6))
                nv = np.stack([np.cos(th), np.sin(th)], 1)
                sep = (nv @ (V[t] - P[o, :2]).T).min(axis=1) - _H
                k = int(sep.argmax())
                if sep[k] > best[0]:
                    best = (sep[k], o, t, k)
            out.append(best)
    return out


def _solve(P0, R0, rounds=4, maxiter=60):
    """SLSQP with separating axes frozen per round and re-picked between rounds."""
    n = len(P0)
    x = np.append(P0.ravel(), R0)
    g = np.zeros(len(x))
    g[-1] = 1.0
    bnds = [(None, None)] * (len(x) - 1) + [(3.3, None)]
    for _ in range(rounds):
        pr = _pairs(x[:-1].reshape(n, 3), 3.2)
        O = np.array([q[1] for q in pr], dtype=int)
        T = np.array([q[2] for q in pr], dtype=int)
        K = np.array([q[3] for q in pr], dtype=int)
        offs = np.radians(30 + 60 * K)

        def cons(x, O=O, T=T, offs=offs):
            P = x[:-1].reshape(n, 3)
            V = _verts(P)
            c1 = x[-1] * _H - np.einsum("kd,nvd->nkv", _N, V)
            th = P[O, 2] + offs
            nv = np.stack([np.cos(th), np.sin(th)], 1)
            c2 = np.einsum("md,mvd->mv", nv, V[T] - P[O, None, :2]) - _H
            return np.concatenate([c1.ravel(), c2.ravel()])

        res = minimize(
            lambda x: x[-1], x, jac=lambda x: g, bounds=bnds,
            constraints=[{"type": "ineq", "fun": cons}],
            method="SLSQP", options={"maxiter": maxiter, "ftol": 1e-12},
        )
        x = res.x
    return x[:-1].reshape(n, 3), x[-1]


def _finalize(P, margin=1e-6):
    """Spread centers slightly, recompute exact R, verify disjointness."""
    P = P.copy()
    P[:, :2] *= 1.0 + margin
    R = np.max(np.einsum("kd,nvd->nkv", _N, _verts(P))) / _H + 1e-9
    if min(q[0] for q in _pairs(P, np.inf)) <= 1e-9:
        return None
    return P, R


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    # Same-orientation unit hexagons tile a triangular lattice whose neighbour
    # vectors have length sqrt(3) at angles 30, 90, 150, ... degrees.
    eps = 1e-6  # tiny gap so touching hexagons never count as overlapping
    s = 1.0 + eps
    ang = np.radians(30 + 60 * np.arange(6))
    ring1 = np.sqrt(3) * np.stack([np.cos(ang), np.sin(ang)], axis=1)
    # Ring-2 edge midpoints (v_k + v_{k+1}) fit exactly inside a side-4 hexagon.
    ring2 = ring1 + np.roll(ring1, -1, axis=0)

    # Center + 6 ring-1 + 4 of the 6 ring-2 midpoints = 11 hexagons
    centers = np.vstack([[0.0, 0.0], ring1, ring2[:4]]) * s
    inner_hex_data = np.column_stack([centers, np.zeros(len(centers))])

    outer_hex_data = np.array([0, 0, 0])  # centered at origin, same orientation
    # Required side: 1 + 3*s (ring-2 centers at lattice offset 3*s*sqrt(3)/2 along normals)
    outer_hex_side_length = 1.0 + 3.0 * s + eps

    # Numerical search for tilted packings; keep the lattice packing as fallback.
    try:
        rng = np.random.default_rng(0)
        n = len(centers)
        t0 = time.time()
        it = 0
        best_P, best_R = None, np.inf  # raw (unspread) best solution for basin hopping
        while time.time() - t0 < 20.0:
            it += 1
            hop = best_P is not None and time.time() - t0 > 9.0 and it % 5 != 0
            if hop:  # basin hopping: perturb the best packing found so far
                sig = rng.choice([0.01, 0.03, 0.08])
                P0 = best_P + rng.normal(0, sig, best_P.shape)
                R0 = best_R + 0.2
                rounds, mi = 5, 100
            elif it % 2:  # jittered lattice start
                P0 = np.column_stack([centers + rng.normal(0, 0.25, centers.shape),
                                      rng.normal(0, 0.2, n)])
                R0 = 4.3
                rounds, mi = 4, 60
            else:  # random tilted start
                rad = 3.2 * np.sqrt(rng.uniform(0, 1, n))
                phi = rng.uniform(0, 2 * np.pi, n)
                P0 = np.column_stack([rad * np.cos(phi), rad * np.sin(phi),
                                      rng.uniform(0, np.pi / 3, n)])
                R0 = 6.0
                rounds, mi = 4, 60
            try:
                P, R = _solve(P0, R0, rounds=rounds, maxiter=mi)
                fin = _finalize(P)
            except Exception:
                continue
            if fin is not None and fin[1] < best_R - 1e-9:
                best_P, best_R = P.copy(), fin[1]
            if fin is not None and fin[1] < outer_hex_side_length - 1e-9:
                Pf, Rf = fin
                inner_hex_data = np.column_stack([Pf[:, :2], np.degrees(Pf[:, 2])])
                outer_hex_side_length = float(Rf)
    except Exception:
        pass

    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
