# EVOLVE-BLOCK-START
import time

import numpy as np
from scipy.optimize import minimize

_S3 = np.sqrt(3.0)
_N = 11
_VT = np.arange(6) * (np.pi / 3)                      # vertex angle offsets
_ON = np.column_stack((np.cos(np.pi / 6 + _VT),       # outer edge normals
                       np.sin(np.pi / 6 + _VT)))      # shape (6,2)
_NK = np.arange(3) * (np.pi / 3) + np.pi / 6          # inner edge normal offsets
_I, _J = np.triu_indices(_N, 1)
_DELTA = 1e-4                                         # clearance during optimisation


def _verts(P):
    t = P[:, 2:3] + _VT[None, :]
    return np.stack((P[:, 0:1] + np.cos(t), P[:, 1:2] + np.sin(t)), axis=-1)  # (n,6,2)


def _axes(P):
    t = P[:, 2:3] + _NK[None, :]
    return np.stack((np.cos(t), np.sin(t)), axis=-1)  # (n,3,2)


def _penalty(z, R):
    P = z.reshape(_N, 3)
    V = _verts(P)
    h = R * _S3 / 2.0
    dots = V @ _ON.T                                   # (n,6,6)
    viol = np.maximum(dots - (h - _DELTA), 0.0)
    pc = np.sum(viol ** 2)
    A = _axes(P)
    ax = np.concatenate((A[_I], A[_J]), axis=1)        # (m,6,2)
    pi = np.einsum("mvd,mad->mav", V[_I], ax)
    pj = np.einsum("mvd,mad->mav", V[_J], ax)
    ov = np.minimum(pi.max(-1) - pj.min(-1), pj.max(-1) - pi.min(-1))  # (m,6)
    pen = ov.min(-1) + _DELTA
    po = np.sum(np.maximum(pen, 0.0) ** 2)
    return pc + po


def _obj(w, mu):
    R = w[-1]
    return R + mu * _penalty(w[:-1], R)


def _exact(P):
    """Return (valid, outer_side). Exact separating-axis and containment checks."""
    if not np.all(np.isfinite(P)):
        return False, np.inf
    V = _verts(P)
    R_req = float((V @ _ON.T).max() / (_S3 / 2.0))
    A = _axes(P)
    ax = np.concatenate((A[_I], A[_J]), axis=1)
    pi = np.einsum("mvd,mad->mav", V[_I], ax)
    pj = np.einsum("mvd,mad->mav", V[_J], ax)
    ov = np.minimum(pi.max(-1) - pj.min(-1), pj.max(-1) - pi.min(-1))
    if np.max(ov.min(-1)) >= -1e-9:
        return False, np.inf
    return True, R_req * (1.0 + 1e-9)


def _req_R(P):
    """Outer side needed to contain the hexagons (ignores overlaps)."""
    V = _verts(P)
    return float((V @ _ON.T).max() / (_S3 / 2.0))


def _lattice_seed(rng, theta_deg, s):
    """11 hexes from a triangular lattice of spacing s, all at angle theta_deg
    (lattice directions chosen so that hexes share edges)."""
    th = np.radians(theta_deg)
    phi = th + np.pi / 6
    a = s * np.array([np.cos(phi), np.sin(phi)])
    b = s * np.array([np.cos(phi + np.pi / 3), np.sin(phi + np.pi / 3)])
    u, v = rng.random(2)
    ii, jj = np.meshgrid(np.arange(-5, 6), np.arange(-5, 6))
    L = ii.ravel()[:, None] * a + jj.ravel()[:, None] * b
    C = L - (u * a + v * b)                      # random lattice offset
    order = np.argsort(np.sum(C ** 2, axis=1))
    C = C[order[:_N]]
    C = C - C.mean(axis=0)                      # recentre cluster
    C = C + rng.normal(scale=0.03, size=C.shape)
    ang = th + rng.normal(scale=0.02, size=_N)
    return np.column_stack((C, ang))


def _ramp(w, mus, t_end):
    for mu in mus:
        if time.time() > t_end:
            break
        res = minimize(_obj, w, args=(mu,), method="L-BFGS-B",
                       options={"maxiter": 150})
        w = res.x
    return w


def _perturb(P, rng, sc):
    Q = P.copy()
    Q[:, :2] += rng.normal(scale=sc, size=(_N, 2))
    Q[:, 2] += rng.normal(scale=sc, size=_N)
    return Q


def _relocate(P, rng, R):
    """Reinsert one random hexagon at a random interior position and angle."""
    Q = P.copy()
    k = rng.integers(_N)
    r = max(R - 1.0, 0.1) * 0.9 * np.sqrt(rng.random())
    th = rng.random() * 2.0 * np.pi
    Q[k, 0] = r * np.cos(th)
    Q[k, 1] = r * np.sin(th)
    Q[k, 2] = rng.uniform(0.0, np.pi / 3)
    return Q


def hexagon_packing_11():
    """
    Returns
        inner_hex_data: np.ndarray (11,3) of (x, y, angle_degrees)
        outer_hex_data: np.ndarray (3,) of (x, y, angle_degrees)
        outer_hex_side_length: float
    """
    t0 = time.time()
    budget = 28.0
    search_end = t0 + budget
    rng = np.random.default_rng(2024)
    mus = (1.0, 10.0, 100.0, 1e3, 1e4, 1e5)

    best_P = None
    best_R = np.inf

    def consider(P):
        nonlocal best_P, best_R
        ok, R = _exact(P)
        if ok and R < best_R:
            best_R = R
            best_P = P.copy()

    # Guaranteed-valid fallback: sparse aligned lattice (spacing 2.4 > 2).
    fallback = _lattice_seed(rng, 0.0, 2.4)
    consider(fallback)

    sc = 0.08
    run = 0
    while time.time() < search_end:
        run += 1
        if best_P is not None and run % 2 == 0:
            # Basin hop from the best packing.
            if rng.random() < 0.4:
                P0 = _relocate(_perturb(best_P, rng, sc * 0.5), rng, best_R)
            else:
                P0 = _perturb(best_P, rng, sc)
            R0 = best_R * 1.02
            mus_run = mus[2:]
        elif rng.random() < 0.6:
            # Structured start: aligned hex-lattice subset (edges shared).
            if rng.random() < 0.7:
                theta = 0.0 if rng.random() < 0.5 else 30.0
            else:
                theta = rng.uniform(0.0, 60.0)
            s = rng.uniform(1.74, 2.0)
            P0 = _lattice_seed(rng, theta, s)
            R0 = max(_req_R(P0), 3.0) * 1.1
            mus_run = mus
        else:
            # Random start.
            r = 3.0 * np.sqrt(rng.random(_N))
            th = rng.random(_N) * 2 * np.pi
            P0 = np.column_stack((r * np.cos(th), r * np.sin(th),
                                  rng.uniform(0, np.pi / 3, _N)))
            R0 = 4.6
            mus_run = mus

        w = np.concatenate((P0.ravel(), [R0]))
        w = _ramp(w, mus_run, search_end)
        before = best_R
        consider(w[:-1].reshape(_N, 3))
        if best_R >= before:
            sc = min(0.2, sc * 1.05)

    # Final polish and exact shrink-repair: only verified packings are accepted.
    if best_P is not None:
        final_end = t0 + budget + 4.0
        for f in (0.998, 0.9985, 0.999, 0.9995):
            if time.time() > final_end:
                break
            w = np.concatenate((best_P.ravel(), [best_R * f]))
            w = _ramp(w, (1e3, 1e4, 1e5), final_end)
            consider(w[:-1].reshape(_N, 3))

    if best_P is None:
        best_P = fallback
        best_R = _req_R(fallback) * (1.0 + 1e-9)

    inner = best_P.copy()
    inner[:, 2] = np.degrees(inner[:, 2])
    outer = np.array([0.0, 0.0, 0.0])
    return inner, outer, float(best_R)


# EVOLVE-BLOCK-END
