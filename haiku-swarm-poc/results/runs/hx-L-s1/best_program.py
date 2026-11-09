import time
import numpy as np
from scipy.optimize import minimize

S3 = np.sqrt(3.0)
OUT_N = np.array(
    [[np.cos(np.radians(30 + 60 * k)), np.sin(np.radians(30 + 60 * k))] for k in range(6)]
)
IU, JU = np.triu_indices(11, 1)


def _verts(c, th):
    ang = th[:, None] + np.arange(6) * (np.pi / 3)
    return c[:, None, :] + np.stack([np.cos(ang), np.sin(ang)], -1)


def _axes(th):
    a = th[:, None] + np.radians(30) + np.arange(3) * (np.pi / 3)
    return np.stack([np.cos(a), np.sin(a)], -1)


def _pen_overlap(c, th):
    V = _verts(c, th)
    A = _axes(th)
    Ax = np.concatenate([A[IU], A[JU]], axis=1)
    pI = np.einsum("pvd,pad->pav", V[IU], Ax)
    pJ = np.einsum("pvd,pad->pav", V[JU], Ax)
    ov = np.minimum(pI.max(-1) - pJ.min(-1), pJ.max(-1) - pI.min(-1)).min(-1)
    return np.maximum(ov, 0.0)


def _cont_viol(c, th, R):
    V = _verts(c, th)
    proj = np.einsum("ivd,kd->ivk", V, OUT_N) - R * S3 / 2
    return np.maximum(proj, 0.0).ravel()


def _R_needed(c, th):
    V = _verts(c, th)
    return np.einsum("ivd,kd->ivk", V, OUT_N).max() / (S3 / 2)


def _obj(z, mu):
    R = z[33]
    c = z[:22].reshape(11, 2)
    th = z[22:33]
    cv = _cont_viol(c, th, R)
    pv = _pen_overlap(c, th)
    return R + mu * (np.sum(cv ** 2) + np.sum(pv ** 2))


def _solve(z):
    for mu in (10.0, 1e2, 1e3, 1e4, 1e5, 1e6):
        try:
            z = minimize(_obj, z, args=(mu,), method="L-BFGS-B",
                         options={"maxiter": 400}).x
        except Exception:
            break
    return z


def _finalize(z):
    c = z[:22].reshape(11, 2).copy()
    th = z[22:33].copy()
    for s in [1.0, 1 + 1e-8, 1 + 1e-7, 1 + 1e-6, 1 + 1e-5, 1 + 1e-4, 1 + 1e-3]:
        cc = c * s
        if _pen_overlap(cc, th).max() <= 1e-10:
            R = _R_needed(cc, th) * (1 + 1e-9) + 1e-9
            if _pen_overlap(cc, th).max() <= 1e-10 and np.all(_cont_viol(cc, th, R) <= 1e-12):
                return cc, th, R
    return None


def _hex_blame(z):
    R = z[33]
    c = z[:22].reshape(11, 2)
    th = z[22:33]
    pv = _pen_overlap(c, th)
    blame = np.zeros(11)
    np.add.at(blame, IU, pv)
    np.add.at(blame, JU, pv)
    cv = _cont_viol(c, th, R).reshape(11, -1).sum(1)
    return blame + cv


def _reinsert(z, rng, idx):
    z = z.copy()
    R = z[33]
    rmax = max(R * S3 / 2 - 1.0, 0.1)
    r = rmax * np.sqrt(rng.random())
    a = rng.uniform(0, 2 * np.pi)
    z[2 * idx] = r * np.cos(a)
    z[2 * idx + 1] = r * np.sin(a)
    z[22 + idx] = rng.uniform(0, np.pi / 3)
    return z


def _jitter(z, rng, s=0.03):
    z = z.copy()
    z[:33] += rng.normal(0, s, 33)
    return z


def _from_cand(cand):
    c, th, R = cand
    return np.concatenate([c.ravel(), th, [R]])


def _lattice_seed(rng):
    # honeycomb-lattice points for unit hexes (neighbor distance sqrt(3))
    e1 = S3 * np.array([np.cos(np.pi / 6), np.sin(np.pi / 6)])
    e2 = S3 * np.array([0.0, 1.0])
    pts = np.array([a * e1 + b * e2 for a in range(-4, 5) for b in range(-4, 5)])
    off = rng.uniform(-S3, S3, 2)
    q = pts + off
    idx = np.argsort(np.linalg.norm(q, axis=1))[:11]
    c = q[idx] * 0.95
    base = rng.choice([0.0, np.pi / 6])
    th = base + rng.normal(0, 0.02, 11)
    return np.concatenate([c.ravel(), th, [4.8]])


def _cluster_seed(rng):
    # 7-hex cluster at lattice spacing plus 4 tilted rim hexes
    e = [S3 * np.array([np.cos(np.radians(30 + 60 * k)), np.sin(np.radians(30 + 60 * k))])
         for k in range(6)]
    centers = [np.zeros(2)] + e
    ring = [2 * ei for ei in e] + [e[k] + e[(k + 1) % 6] for k in range(6)]
    pick = rng.choice(len(ring), 4, replace=False)
    centers += [ring[i] for i in pick]
    c = np.array(centers) * 1.0
    th = np.zeros(11)
    th[7:] = np.pi / 6 + rng.normal(0, 0.05, 4)
    th[:7] += rng.normal(0, 0.02, 7)
    return np.concatenate([c.ravel(), th, [4.8]])


def _random_seed(rng):
    c = rng.uniform(-2.8, 2.8, (11, 2))
    th = rng.uniform(0, np.pi / 3, 11)
    return np.concatenate([c.ravel(), th, [4.8]])


def _seed(rng, k):
    if k == 0:
        return _lattice_seed(rng)
    if k == 1:
        return _cluster_seed(rng)
    return _random_seed(rng)


def _fallback():
    inner = np.array(
        [
            [0, 0, 0],
            [-2.5, 0, 0],
            [2.5, 0, 0],
            [-1.25, 2.17, 0],
            [1.25, 2.17, 0],
            [-1.25, -2.17, 0],
            [1.25, -2.17, 0],
            [-3.75, 2.17, 0],
            [3.75, 2.17, 0],
            [-3.75, -2.17, 0],
            [3.75, -2.17, 0],
        ],
        dtype=float,
    )
    c = inner[:, :2].copy()
    th = np.zeros(11)
    R = _R_needed(c, th) * (1 + 1e-9) + 1e-9
    return inner, R


def hexagon_packing_11():
    """
    Packs 11 unit hexagons in a regular hexagon (outer centered at origin,
    vertices at R*(cos 60k, sin 60k)). Returns (inner (11,3) array of
    x, y, angle_deg; outer (x, y, angle_deg); outer side length R).
    """
    rng = np.random.default_rng(2024)
    t0 = time.time()
    budget = 12.0

    best = None
    bestR = np.inf
    cur_z = None
    curR = np.inf

    # Phase 1: mixed structured / random multistart
    trial = 0
    while time.time() - t0 < 4.5 or best is None:
        trial += 1
        if trial > 400 and best is not None:
            break
        z = _solve(_seed(rng, trial % 3))
        cand = _finalize(z)
        if cand is not None and cand[2] < bestR:
            bestR = cand[2]
            best = cand
            cur_z = _from_cand(cand)
            curR = bestR
        if time.time() - t0 > budget:
            break

    # Phase 2: annealed basin hopping with stall-triggered restarts
    if best is not None:
        p2_start = time.time()
        p2_len = max(budget - (p2_start - t0), 1e-3)
        T0, T1 = 2e-3, 2e-5
        stall = 0
        while time.time() - t0 < budget:
            frac = min((time.time() - p2_start) / p2_len, 1.0)
            T = T0 * (T1 / T0) ** frac

            if stall > 60:
                # restart from a fresh seed, keep global best
                z = _solve(_seed(rng, rng.integers(3)))
                cand = _finalize(z)
                stall = 0
                if cand is not None:
                    cur_z = _from_cand(cand)
                    curR = cand[2]
                    if cand[2] < bestR - 1e-9:
                        bestR = cand[2]
                        best = cand
                continue

            z = cur_z.copy()
            mode = rng.random()
            if mode < 0.5:
                blame = _hex_blame(z)
                if blame.max() > 1e-6:
                    p = blame + 1e-9
                    idx = rng.choice(11, p=p / p.sum())
                else:
                    idx = rng.integers(11)
                z = _reinsert(z, rng, idx)
            elif mode < 0.7:
                idx = rng.integers(11)
                z[22 + idx] = (z[22 + idx] + np.radians(30)) % (np.pi / 3)
            else:
                pass
            z = _jitter(z, rng, s=0.03 if mode < 0.85 else 0.1)
            z = _solve(z)
            cand = _finalize(z)
            if cand is None:
                stall += 1
                continue

            Rn = cand[2]
            if Rn < bestR - 1e-9:
                bestR = Rn
                best = cand
                stall = 0
            else:
                stall += 1

            if Rn < curR or rng.random() < np.exp(-(Rn - curR) / T):
                cur_z = _from_cand(cand)
                curR = Rn

    if best is None:
        inner, R = _fallback()
        return inner, np.array([0.0, 0.0, 0.0]), float(R)

    c, th, R = best
    inner = np.column_stack([c, np.degrees(th)])
    outer = np.array([0.0, 0.0, 0.0])
    return inner, outer, float(R)
