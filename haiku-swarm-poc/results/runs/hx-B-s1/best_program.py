import time
import numpy as np
from scipy.optimize import minimize

N = 11
APO = np.sqrt(3.0) / 2.0
BASE = np.arange(6) * (np.pi / 3.0)
NORMALS = np.stack([np.cos(np.pi / 6 + BASE), np.sin(np.pi / 6 + BASE)], axis=1)  # (6,2)
AX_OFF = np.pi / 6 + np.arange(3) * (np.pi / 3.0)
_IU = np.triu_indices(N, 1)
SEP_MARGIN = 2e-6
PAIR_CUTOFF = 2.6
MUS_FULL = (1e1, 1e2, 1e3, 1e4, 1e5, 1e6)
MUS_SHORT = (1e2, 1e3, 1e4, 1e5, 1e6)
BOUNDS = [(None, None)] * (3 * N) + [(1.0, 20.0)]
T_TOTAL = 15.0
F_MULTI = 0.30   # end of multistart (fraction of T_TOTAL)
F_BH = 0.68      # end of basin hopping
F_POLISH = 0.88  # end of SLSQP polish of pool
POOL_SIZE = 8


def _verts(C, A):
    ang = A[:, None] + BASE[None, :]
    x = C[:, 0:1] + np.cos(ang)
    y = C[:, 1:2] + np.sin(ang)
    return np.stack([x, y], axis=2)  # (n,6,2)


def _pair_depth(C, A):
    """Separating-axis penetration matrix; entry (i,j) <= 0 means i and j are separated."""
    V = _verts(C, A)
    ang = A[:, None] + AX_OFF[None, :]
    U = np.stack([np.cos(ang), np.sin(ang)], axis=2).reshape(-1, 2)
    P = np.einsum('ivd,md->imv', V, U)
    mn = P.min(axis=2)
    mx = P.max(axis=2)
    D = np.minimum(mx[:, None, :], mx[None, :, :]) - np.maximum(mn[:, None, :], mn[None, :, :])
    return D.min(axis=2)


def _contain(C, A):
    V = _verts(C, A)
    return np.einsum('ivd,kd->ivk', V, NORMALS)  # (n,6,6)


def _objective(z, mu):
    C = z[:2 * N].reshape(N, 2)
    A = z[2 * N:3 * N]
    R = z[3 * N]
    P = _contain(C, A)
    viol = np.maximum(P - R * APO, 0.0)
    pp = np.maximum(_pair_depth(C, A)[_IU], 0.0)
    return R + mu * (np.sum(viol ** 2) + np.sum(pp ** 2))


def _finalize(C, A):
    """Repair pass: minimal scale about the origin that strictly separates all pairs,
    then R is read off the exact containment. Returns (R, C, A) or None."""
    if not (np.all(np.isfinite(C)) and np.all(np.isfinite(A))):
        return None
    grid = (1.0, 1.0 + 1e-9, 1.0 + 1e-8, 1.0 + 1e-7, 1.0 + 1e-6, 1.0 + 1e-5,
            1.0 + 1e-4, 1.0 + 1e-3, 1.0 + 1e-2)

    def sep(s):
        return np.max(_pair_depth(C * s, A)[_IU]) <= -1e-9

    k = None
    for idx, s in enumerate(grid):
        if sep(s):
            k = idx
            break
    if k is None:
        return None
    hi = grid[k]
    lo = grid[k - 1] if k > 0 else None
    if lo is not None:
        for _ in range(30):  # bisect toward the minimal feasible scale
            mid = 0.5 * (lo + hi)
            if sep(mid):
                hi = mid
            else:
                lo = mid
    C2 = C * hi
    R = float(_contain(C2, A).max() / APO)
    if np.isfinite(R) and R > 0:
        return R, C2, A.copy()
    return None


def _relax(C, A, R0, mus):
    z = np.concatenate([C.ravel(), A, [R0]])
    for mu in mus:
        try:
            r = minimize(_objective, z, args=(mu,), method='L-BFGS-B',
                         bounds=BOUNDS, options={'maxiter': 150})
            if not np.all(np.isfinite(r.x)):
                break
            z = r.x
        except Exception:
            break
    return z[:2 * N].reshape(N, 2), z[2 * N:3 * N], float(z[3 * N])


def _polish(C, A, R, maxiter=500):
    """SLSQP on exact containment and per-pair separating-line constraints, minimizing R."""
    d = np.linalg.norm(C[:, None, :] - C[None, :, :], axis=2)
    keep = [(int(i), int(j)) for i, j in zip(*_IU) if d[i, j] < PAIR_CUTOFF]
    m = len(keep)
    if m == 0:
        return None
    I = np.array([p[0] for p in keep], dtype=int)
    J = np.array([p[1] for p in keep], dtype=int)
    v = C[J] - C[I]
    ph0 = np.arctan2(v[:, 1], v[:, 0])
    U0 = np.stack([np.cos(ph0), np.sin(ph0)], axis=1)
    b0 = 0.5 * np.sum((C[I] + C[J]) * U0, axis=1)
    ar = np.arange(m)

    def unpack(z):
        C_ = z[:2 * N].reshape(N, 2)
        A_ = z[2 * N:3 * N]
        R_ = z[3 * N]
        ph = z[3 * N + 1:3 * N + 1 + m]
        bb = z[3 * N + 1 + m:3 * N + 1 + 2 * m]
        return C_, A_, R_, ph, bb

    def cons(z):
        C_, A_, R_, ph, bb = unpack(z)
        V = _verts(C_, A_)
        cont = (R_ * APO - np.einsum('nkd,ld->nkl', V, NORMALS)).ravel()
        U = np.stack([np.cos(ph), np.sin(ph)], axis=1)
        P = np.einsum('nkd,md->mnk', V, U)  # (m,N,6)
        c1 = bb[:, None] - P[ar, I] - SEP_MARGIN
        c2 = P[ar, J] - bb[:, None] - SEP_MARGIN
        return np.concatenate([cont, c1.ravel(), c2.ravel()])

    def obj(z):
        return z[3 * N]

    bounds = [(None, None)] * (3 * N) + [(1.0, 20.0)] + [(None, None)] * (2 * m)
    z0 = np.concatenate([C.ravel(), A, [R], ph0, b0])
    try:
        res = minimize(obj, z0, method='SLSQP', bounds=bounds,
                       constraints=[{'type': 'ineq', 'fun': cons}],
                       options={'maxiter': maxiter, 'ftol': 1e-12})
        z = res.x
    except Exception:
        return None
    if not np.all(np.isfinite(z)):
        return None
    C_, A_, R_, _, _ = unpack(z)
    return C_, A_, float(R_)


def _lattice_seed(rng):
    """Honeycomb cluster of 11 cells, tilt 0 or 30 degrees (or random), mixed 0/30 deg cells."""
    a1 = np.sqrt(3.0) * np.array([np.cos(np.pi / 6), np.sin(np.pi / 6)])
    a2 = np.sqrt(3.0) * np.array([0.0, 1.0])
    ii, jj = np.meshgrid(np.arange(-3, 4), np.arange(-3, 4))
    L = np.stack([ii.ravel(), jj.ravel()], axis=1).astype(float)
    pts = L @ np.array([a1, a2])
    u = rng.random()
    if u < 0.3:
        tilt = 0.0
    elif u < 0.6:
        tilt = np.pi / 6
    elif u < 0.75:
        tilt = rng.uniform(0.0, np.pi / 3)
    else:
        tilt = 0.0
    c, s = np.cos(tilt), np.sin(tilt)
    pts = pts @ np.array([[c, -s], [s, c]]).T
    shift = rng.uniform(-0.5, 0.5, 2)
    q = pts + shift
    order = np.argsort(np.linalg.norm(q, axis=1) + 0.05 * rng.random(len(q)))
    C = q[order[:N]] * 1.03 + 0.03 * rng.standard_normal((N, 2))
    base = np.where(rng.random(N) < 0.5, 0.0, np.pi / 6)
    A = tilt + base + 0.04 * rng.standard_normal(N)
    return C, A


def _random_seed(rng):
    r = 2.8 * np.sqrt(rng.random(N))
    th = 2.0 * np.pi * rng.random(N)
    C = np.stack([r * np.cos(th), r * np.sin(th)], axis=1)
    A = rng.uniform(0.0, np.pi / 3.0, N)
    return C, A


def _relocate(C, A, R, rng, k):
    C = C.copy()
    A = A.copy()
    rmax = max(R * APO - 1.0, 0.2)
    r = rmax * np.sqrt(rng.random())
    th = 2.0 * np.pi * rng.random()
    C[k] = [r * np.cos(th), r * np.sin(th)]
    A[k] = (0.0 if rng.random() < 0.5 else np.pi / 6) + 0.1 * rng.standard_normal()
    return C, A


def _targeted_move(C, A, R, rng):
    """Reinsert the most overlapped / out-of-bounds hexagon at the most open candidate spot."""
    C = C.copy()
    A = A.copy()
    pd = _pair_depth(C, A)
    np.fill_diagonal(pd, -np.inf)
    overlap = np.maximum(pd.max(axis=1), 0.0)
    cv = np.maximum(_contain(C, A).max(axis=2) - R * APO, 0.0).max(axis=1)
    score = overlap + cv
    if score.max() <= 1e-9:
        k = int(rng.integers(N))
    else:
        k = int(np.argmax(score + 1e-3 * rng.random(N)))
    others = np.delete(np.arange(N), k)
    half = max(R * APO - 0.5, 0.2)
    M = 400
    rr = half * np.sqrt(rng.random(M))
    th = 2.0 * np.pi * rng.random(M)
    cand = np.stack([rr * np.cos(th), rr * np.sin(th)], axis=1)
    dmin = np.linalg.norm(cand[:, None, :] - C[others][None, :, :], axis=2).min(axis=1)
    bnd = R * APO - (cand @ NORMALS.T).max(axis=1)
    sc = np.minimum(dmin, bnd + 1.0) + 0.05 * rng.random(M)
    C[k] = cand[int(np.argmax(sc))]
    A[k] = rng.uniform(0.0, np.pi / 3.0)
    return C, A


def _perturb(Cb, Ab, Rb, rng):
    u = rng.random()
    if u < 0.35:
        return _targeted_move(Cb, Ab, Rb, rng)
    if u < 0.6:
        C, A = Cb.copy(), Ab.copy()
        for k in rng.choice(N, size=int(rng.integers(1, 3)), replace=False):
            C, A = _relocate(C, A, Rb, rng, int(k))
        return C, A
    if u < 0.8:
        A = Ab.copy()
        k = int(rng.integers(N))
        A[k] = (0.0 if rng.random() < 0.5 else np.pi / 6) + 0.05 * rng.standard_normal()
        return Cb + 0.02 * rng.standard_normal(Cb.shape), A
    scale = float(rng.choice([0.02, 0.05, 0.1]))
    return Cb + scale * rng.standard_normal(Cb.shape), Ab + 0.5 * scale * rng.standard_normal(N)


def hexagon_packing_11():
    rng = np.random.default_rng(2024)
    t0 = time.time()

    def elapsed():
        return time.time() - t0

    pool = []  # sorted list of (R, C, A), distinct valid packings, best first

    def consider(C, A):
        out = _finalize(C, A)
        if out is None:
            return None
        R, C2, A2 = out
        if any(abs(R - p[0]) < 1e-8 for p in pool):
            return out
        pool.append((R, C2, A2))
        pool.sort(key=lambda t: t[0])
        del pool[POOL_SIZE:]
        return out

    # Phase 1: multistart from lattice and random seeds.
    trial = 0
    while elapsed() < F_MULTI * T_TOTAL and trial < 200:
        trial += 1
        if trial % 3 == 0:
            C, A = _random_seed(rng)
        else:
            C, A = _lattice_seed(rng)
        C, A, _ = _relax(C, A, 4.5, MUS_FULL)
        consider(C, A)

    # Phase 2: basin hopping with Metropolis acceptance.
    cur = pool[0] if pool else None  # (R, C, A)
    t_bh0 = elapsed()
    span = max(F_BH * T_TOTAL - t_bh0, 1e-9)
    while elapsed() < F_BH * T_TOTAL:
        frac = min(max((elapsed() - t_bh0) / span, 0.0), 1.0)
        T = 0.01 * (1.0 - frac) + 1e-5
        if cur is None:
            C, A = _random_seed(rng)
            C, A, Rn = _relax(C, A, 4.5, MUS_FULL)
            consider(C, A)
            if np.isfinite(Rn):
                cur = (Rn, C.copy(), A.copy())
            continue
        if rng.random() < 0.75 or len(pool) == 1:
            Rb, Cb, Ab = cur
        else:
            Rb, Cb, Ab = pool[int(rng.integers(len(pool)))]
        C, A = _perturb(Cb, Ab, Rb, rng)
        C, A, Rn = _relax(C, A, Rb * 1.02, MUS_SHORT)
        consider(C, A)
        if not np.isfinite(Rn):
            continue
        if Rn < cur[0] or rng.random() < np.exp(-(Rn - cur[0]) / T):
            cur = (Rn, C.copy(), A.copy())

    # Phase 3: exact SLSQP polish with repair, on the top pool entries.
    def polish_chain(R0, C0, A0, rounds=3):
        Rc, Cc, Ac = R0, C0, A0
        for _ in range(rounds):
            if elapsed() > F_POLISH * T_TOTAL:
                break
            out = _polish(Cc, Ac, Rc, maxiter=500)
            if out is None:
                break
            C1, A1, R1 = out
            consider(C1, A1)
            if R1 < Rc - 1e-10:
                Rc, Cc, Ac = R1, C1, A1
            else:
                break
        return Rc, Cc, Ac

    for R0, C0, A0 in list(pool[:POOL_SIZE]):
        if elapsed() > F_POLISH * T_TOTAL:
            break
        polish_chain(R0, C0, A0)

    # Phase 4: small perturbed restarts of the best polished packing.
    while pool and elapsed() < T_TOTAL:
        Rb, Cb, Ab = pool[0]
        C = Cb + 0.01 * rng.standard_normal(Cb.shape)
        A = Ab + 0.01 * rng.standard_normal(N)
        if elapsed() > F_POLISH * T_TOTAL:
            break
        polish_chain(Rb * 1.01, C, A, rounds=2)

    if not pool:
        # Fallback: axis-aligned honeycomb cluster, validated through the same repair pass.
        fb_C = np.array([
            [0, 0], [-2.5, 0], [2.5, 0],
            [-1.25, 2.17], [1.25, 2.17],
            [-1.25, -2.17], [1.25, -2.17],
            [-3.75, 2.17], [3.75, 2.17],
            [-3.75, -2.17], [3.75, -2.17],
        ], dtype=float)
        fb_A = np.zeros(N)
        out = _finalize(fb_C, fb_A)
        if out is None:
            R_fb, fb_C2, fb_A2 = 8.0, fb_C, fb_A
        else:
            R_fb, fb_C2, fb_A2 = out
        inner = np.column_stack([fb_C2, np.rad2deg(fb_A2)])
        return inner, np.array([0.0, 0.0, 0.0]), float(R_fb * (1 + 1e-10) + 1e-10)

    R, C, A = pool[0]
    inner_hex_data = np.column_stack([C, np.rad2deg(A)]).astype(float)
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    return inner_hex_data, outer_hex_data, float(R * (1 + 1e-10) + 1e-10)
