# EVOLVE-BLOCK-START
"""Row / hex / graded-boundary / random multistart -> penalty relaxation -> SLSQP -> exact LP radii,
then time-bounded threshold basin hopping over an elite pool with smallest-circle reinsertion,
and a final multi-tolerance active-contact Newton snap followed by exact LP radii (n=26)."""
import time
import numpy as np
from scipy.optimize import minimize, linprog

N = 26
PI, PJ = np.triu_indices(N, 1)
P = len(PI)
TIME_BUDGET = 12.0
ELITE_K = 4
STALL_LIMIT = 25
THRESH0 = 0.002
ACTIVE_TOLS = (1e-9, 1e-7, 1e-6)
MARGIN = 1e-12

ROW_PATTERNS = [
    [5, 6, 5, 5, 5],
    [6, 5, 6, 5, 4],
    [5, 6, 5, 6, 4],
    [5, 5, 6, 5, 5],
    [4, 5, 6, 5, 6],
]


def row_seed(rows):
    ys = np.linspace(0.1, 0.9, len(rows))
    pts = []
    for k, y in zip(rows, ys):
        lo = 0.1 - 0.02 * (k == 6)
        hi = 1.0 - lo
        for x in np.linspace(lo, hi, k):
            pts.append((x, y))
    return np.clip(np.array(pts, dtype=float), 0.02, 0.98)


def hex_seed(rows):
    ys = np.linspace(0.09, 0.91, len(rows))
    pts = []
    for idx, (k, y) in enumerate(zip(rows, ys)):
        shift = 0.5 * (idx % 2) / max(k, 1)
        xs = np.linspace(0.08, 0.92, k) + shift * (0.84 / max(k - 1, 1))
        for x in np.clip(xs, 0.05, 0.95):
            pts.append((x, y))
    return np.clip(np.array(pts, dtype=float), 0.02, 0.98)


def graded_seed(rng):
    """Large corner/edge circles around a smaller, jittered interior grid."""
    corners = np.array([[0.08, 0.08], [0.92, 0.08], [0.08, 0.92], [0.92, 0.92]])
    n_edge = int(rng.integers(8, 11))
    ang = np.linspace(0, 2 * np.pi, n_edge, endpoint=False) + rng.uniform(0, 0.4)
    edge = np.stack([0.5 + 0.43 * np.cos(ang), 0.5 + 0.43 * np.sin(ang)], axis=1)
    rest = N - len(corners) - len(edge)
    g = int(np.ceil(np.sqrt(rest)))
    gx = np.linspace(0.25, 0.75, g)
    grid = np.array([(x, y) for x in gx for y in gx])[:rest]
    pts = np.vstack([corners, edge, grid]) + rng.normal(0.0, 0.01, (N, 2))
    return np.clip(pts, 0.03, 0.97)


def random_seed(rng):
    return rng.uniform(0.08, 0.92, (N, 2))


def greedy_radii(centers):
    x, y = centers[:, 0], centers[:, 1]
    radii = np.clip(np.minimum.reduce([x, y, 1 - x, 1 - y]), 0.0, None)
    for i in range(N):
        for j in range(i + 1, N):
            d = np.hypot(*(centers[i] - centers[j]))
            s = radii[i] + radii[j]
            if s > d and s > 0:
                f = d / s
                radii[i] *= f
                radii[j] *= f
    return radii


def repair(centers, radii):
    c = np.clip(centers, 0.0, 1.0)
    r = np.clip(radii, 0.0, None)
    x, y = c[:, 0], c[:, 1]
    r = np.clip(np.minimum.reduce([r, x, y, 1 - x, 1 - y]), 0.0, None)
    for i in range(N):
        for j in range(i + 1, N):
            d = np.hypot(*(c[i] - c[j]))
            s = r[i] + r[j]
            if s > d:
                f = d / s if s > 0 else 0.0
                r[i] *= f
                r[j] *= f
    return c, r


def is_valid(centers, radii, tol=1e-10):
    c = np.asarray(centers, dtype=float)
    r = np.asarray(radii, dtype=float)
    if not (np.all(np.isfinite(c)) and np.all(np.isfinite(r))):
        return False
    x, y = c[:, 0], c[:, 1]
    if np.any(r < -tol):
        return False
    if np.any(x - r < -tol) or np.any(x + r > 1 + tol):
        return False
    if np.any(y - r < -tol) or np.any(y + r > 1 + tol):
        return False
    d = np.sqrt(((c[PI] - c[PJ]) ** 2).sum(axis=1))
    return bool(np.all(d >= r[PI] + r[PJ] - tol))


def lp_radii(centers):
    """Exact LP: maximise sum(r) for fixed centres."""
    c = np.asarray(centers, dtype=float)
    x, y = c[:, 0], c[:, 1]
    bnd = np.stack([x, y, 1.0 - x, 1.0 - y], axis=1).ravel()
    W = np.zeros((4 * N, N))
    W[np.arange(4 * N), np.repeat(np.arange(N), 4)] = 1.0
    d = np.sqrt(((c[PI] - c[PJ]) ** 2).sum(axis=1))
    Q = np.zeros((P, N))
    Q[np.arange(P), PI] = 1.0
    Q[np.arange(P), PJ] = 1.0
    A = np.vstack([W, Q])
    b = np.concatenate([np.maximum(bnd - MARGIN, 0.0), np.maximum(d - MARGIN, 0.0)])
    try:
        res = linprog(-np.ones(N), A_ub=A, b_ub=b, bounds=[(0.0, None)] * N, method="highs")
    except Exception:
        return None
    if not res.success:
        return None
    return np.maximum(res.x, 0.0)


def lp_polish(c, r):
    rl = lp_radii(c)
    if rl is not None:
        c2, rl2 = repair(c, rl)
        if rl2.sum() > r.sum() + 1e-12 and is_valid(c2, rl2):
            return c2, rl2
    return c, r


def _pen_obj(z, w):
    x, y, r = z[:N], z[N:2 * N], z[2 * N:]
    dx = x[PI] - x[PJ]
    dy = y[PI] - y[PJ]
    d = np.sqrt(dx * dx + dy * dy) + 1e-12
    ov = np.maximum(0.0, r[PI] + r[PJ] - d)
    a = np.maximum(0.0, r - x)
    b = np.maximum(0.0, x + r - 1.0)
    c_ = np.maximum(0.0, r - y)
    e = np.maximum(0.0, y + r - 1.0)
    E = np.sum(ov * ov) + np.sum(a * a) + np.sum(b * b) + np.sum(c_ * c_) + np.sum(e * e)
    f = -np.sum(r) + w * E
    coef = -2.0 * w * ov / d
    gx = np.bincount(PI, coef * dx, N) - np.bincount(PJ, coef * dx, N)
    gy = np.bincount(PI, coef * dy, N) - np.bincount(PJ, coef * dy, N)
    gr = np.bincount(PI, 2 * w * ov, N) + np.bincount(PJ, 2 * w * ov, N)
    gx += -2 * w * a + 2 * w * b
    gy += -2 * w * c_ + 2 * w * e
    gr += 2 * w * (a + b + c_ + e) - 1.0
    return f, np.concatenate([gx, gy, gr])


BOUNDS = [(0, 1)] * (2 * N) + [(0, 0.5)] * N


def relax(centers, radii):
    z = np.concatenate([centers[:, 0], centers[:, 1], radii])
    try:
        for w in (10.0, 100.0, 1000.0, 1e4):
            res = minimize(_pen_obj, z, args=(w,), jac=True, method="L-BFGS-B",
                           bounds=BOUNDS, options={"maxiter": 200})
            z = res.x
    except Exception:
        pass
    return np.stack([z[:N], z[N:2 * N]], axis=1), z[2 * N:]


def _wall_fun(z):
    x, y, r = z[:N], z[N:2 * N], z[2 * N:]
    return np.concatenate([x - r, 1 - x - r, y - r, 1 - y - r])


def _wall_jac(z):
    J = np.zeros((4 * N, 3 * N))
    k = np.arange(N)
    J[k, k] = 1.0
    J[k, 2 * N + k] = -1.0
    J[N + k, k] = -1.0
    J[N + k, 2 * N + k] = -1.0
    J[2 * N + k, N + k] = 1.0
    J[2 * N + k, 2 * N + k] = -1.0
    J[3 * N + k, N + k] = -1.0
    J[3 * N + k, 2 * N + k] = -1.0
    return J


def _pair_fun(z):
    x, y, r = z[:N], z[N:2 * N], z[2 * N:]
    dx = x[PI] - x[PJ]
    dy = y[PI] - y[PJ]
    s = r[PI] + r[PJ]
    return dx * dx + dy * dy - s * s


def _pair_jac(z):
    x, y, r = z[:N], z[N:2 * N], z[2 * N:]
    dx = x[PI] - x[PJ]
    dy = y[PI] - y[PJ]
    s = r[PI] + r[PJ]
    J = np.zeros((P, 3 * N))
    rows = np.arange(P)
    J[rows, PI] = 2 * dx
    J[rows, PJ] = -2 * dx
    J[rows, N + PI] = 2 * dy
    J[rows, N + PJ] = -2 * dy
    J[rows, 2 * N + PI] = -2 * s
    J[rows, 2 * N + PJ] = -2 * s
    return J


def optimise(centers, radii, maxiter=300, ftol=1e-12):
    z0 = np.concatenate([centers[:, 0], centers[:, 1], radii])
    cons = [
        {"type": "ineq", "fun": _wall_fun, "jac": _wall_jac},
        {"type": "ineq", "fun": _pair_fun, "jac": _pair_jac},
    ]

    def obj(z):
        return -np.sum(z[2 * N:])

    def obj_jac(z):
        g = np.zeros(3 * N)
        g[2 * N:] = -1.0
        return g

    try:
        res = minimize(obj, z0, jac=obj_jac, bounds=BOUNDS, constraints=cons,
                       method="SLSQP", options={"maxiter": maxiter, "ftol": ftol})
        z = res.x
        if not np.all(np.isfinite(z)):
            z = z0
    except Exception:
        z = z0
    return np.stack([z[:N], z[N:2 * N]], axis=1), z[2 * N:]


def polish(centers, radii, maxiter=300, use_relax=True):
    """(Optional relax) -> SLSQP -> repair -> exact LP radii; returns a validated triple."""
    c, r = relax(centers, radii) if use_relax else (centers, radii)
    c, r = optimise(c, r, maxiter=maxiter)
    c, r = repair(c, r)
    c, r = lp_polish(c, r)
    if not is_valid(c, r):
        c, r = repair(centers, greedy_radii(centers))
    return c, r, float(np.sum(r))


# ---------- active-contact Newton snap ----------
def _contact_system(z, pairs, walls):
    x, y, rr = z[:N], z[N:2 * N], z[2 * N:]
    npair = len(pairs)
    m = npair + len(walls)
    F = np.zeros(m)
    J = np.zeros((m, 3 * N))
    if npair:
        i = PI[pairs]
        j = PJ[pairs]
        k = np.arange(npair)
        dx = x[i] - x[j]
        dy = y[i] - y[j]
        s = rr[i] + rr[j]
        F[k] = dx * dx + dy * dy - s * s
        J[k, i] = 2.0 * dx
        J[k, j] = -2.0 * dx
        J[k, N + i] = 2.0 * dy
        J[k, N + j] = -2.0 * dy
        J[k, 2 * N + i] = -2.0 * s
        J[k, 2 * N + j] = -2.0 * s
    if walls:
        wi = np.array([w[0] for w in walls], dtype=int)
        wt = np.array([w[1] for w in walls], dtype=int)
        k = npair + np.arange(len(walls))
        xi, yi, ri = x[wi], y[wi], rr[wi]
        F[k] = np.select([wt == 0, wt == 1, wt == 2],
                         [xi - ri, 1.0 - xi - ri, yi - ri],
                         default=1.0 - yi - ri)
        col = np.where(wt <= 1, wi, N + wi)
        sgn = np.where((wt == 0) | (wt == 2), 1.0, -1.0)
        J[k, col] = sgn
        J[k, 2 * N + wi] = -1.0
    return F, J


def newton_refine(c, r, tol, iters=60):
    """Min-norm Gauss-Newton on contacts with gap < tol. Returns centres; radii come from LP."""
    gaps = np.hypot(*(c[PI] - c[PJ]).T) - (r[PI] + r[PJ])
    pairs = np.where(gaps < tol)[0]
    walls = []
    for i in range(N):
        x, y = c[i]
        for t, g in enumerate((x - r[i], 1.0 - x - r[i], y - r[i], 1.0 - y - r[i])):
            if g < tol:
                walls.append((i, t))
    if len(pairs) + len(walls) == 0:
        return None
    z = np.concatenate([c[:, 0], c[:, 1], r])
    for _ in range(iters):
        F, J = _contact_system(z, pairs, walls)
        if np.max(np.abs(F)) < 1e-14:
            break
        step = np.linalg.lstsq(J, -F, rcond=None)[0]
        z = z + step
    F, _ = _contact_system(z, pairs, walls)
    if not np.all(np.isfinite(z)) or np.max(np.abs(F)) > 1e-10:
        return None
    return np.stack([z[:N], z[N:2 * N]], axis=1)


def refine_exact(c, r, best_s):
    """Snap contacts at several tolerances, re-solve radii by LP; accept only validated gains."""
    for tol in ACTIVE_TOLS:
        cn = newton_refine(c, r, tol)
        if cn is None:
            continue
        rl = lp_radii(cn)
        if rl is None:
            continue
        c2, r2 = repair(cn, rl)
        if is_valid(c2, r2) and r2.sum() > best_s + 1e-12:
            c, r, best_s = c2, r2, float(r2.sum())
    return c, r, best_s


# ---------- structured moves ----------
_G = np.linspace(0.02, 0.98, 60)
_GX, _GY = np.meshgrid(_G, _G)
_GRID = np.stack([_GX.ravel(), _GY.ravel()], axis=1)
_GWALL = np.minimum.reduce([_GRID[:, 0], _GRID[:, 1], 1 - _GRID[:, 0], 1 - _GRID[:, 1]])


def reinsert(centers, radii, k):
    """Move the k smallest circles into the grid cells with largest clearance."""
    c = centers.copy()
    r = radii.copy()
    for i in np.argsort(r)[:k]:
        others = np.delete(np.arange(N), i)
        dd = np.sqrt(((_GRID[:, None, :] - c[others][None, :, :]) ** 2).sum(-1))
        dd = dd - r[others][None, :]
        score = np.minimum(_GWALL, dd.min(axis=1))
        j = int(np.argmax(score))
        c[i] = _GRID[j]
        r[i] = max(score[j], 0.0)
    return c, r


def kick_smallest(rng, centers, radii, k):
    c = centers.copy()
    idx = np.argsort(radii)[:k]
    c[idx] = np.clip(c[idx] + rng.normal(0.0, 0.05, size=(k, 2)), 0.03, 0.97)
    return c, greedy_radii(c)


def global_kick(rng, centers):
    c = np.clip(centers + rng.normal(0.0, 0.01, centers.shape), 0.02, 0.98)
    return c, greedy_radii(c)


def propose(rng, pool, c, r):
    u = rng.random()
    if u < 0.35:
        e = pool[int(rng.integers(len(pool)))]
        return reinsert(e[0], e[1], int(rng.choice([1, 2])))
    if u < 0.7:
        return kick_smallest(rng, c, r, int(rng.choice([2, 3])))
    return global_kick(rng, c)


def pool_insert(pool, c, r, s):
    key = np.round(np.sort(r), 4)
    for _, _, _, k in pool:
        if np.allclose(key, k, atol=1e-4):
            return pool
    pool.append((c, r, s, key))
    pool.sort(key=lambda t: -t[2])
    del pool[ELITE_K:]
    return pool


def construct_packing():
    t0 = time.perf_counter()
    rng = np.random.default_rng(2024)

    seeds = [row_seed(rows) for rows in ROW_PATTERNS]
    seeds += [hex_seed(rows) for rows in ROW_PATTERNS[:3]]
    seeds += [graded_seed(rng) for _ in range(3)]
    seeds += [random_seed(rng) for _ in range(2)]

    pool = []
    for s in seeds:
        if time.perf_counter() - t0 > TIME_BUDGET * 0.35 and pool:
            break
        c, r, tot = polish(s, greedy_radii(s))
        if is_valid(c, r):
            pool = pool_insert(pool, c, r, tot)

    if not pool:
        s = row_seed(ROW_PATTERNS[0])
        c, r = repair(s, greedy_radii(s))
        pool = [(c, r, float(np.sum(r)), np.round(np.sort(r), 4))]

    best_c, best_r, best_s = pool[0][0], pool[0][1], pool[0][2]

    # Threshold basin hopping with elite reheats, time-bounded.
    cur_c, cur_r, cur_s = best_c, best_r, best_s
    stall = 0
    t_bh = time.perf_counter()
    bh_end = t0 + TIME_BUDGET * 0.85
    span = max(bh_end - t_bh, 1e-6)
    while time.perf_counter() < bh_end:
        frac = (time.perf_counter() - t_bh) / span
        thr = THRESH0 * (1.0 - frac) + 1e-6
        if stall >= STALL_LIMIT:
            e = pool[int(rng.integers(len(pool)))]
            cur_c, cur_r, cur_s = e[0], e[1], e[2]
            stall = 0
        pc, pr = propose(rng, pool, cur_c, cur_r)
        c, r, tot = polish(pc, pr, use_relax=False)
        if not is_valid(c, r):
            stall += 1
            continue
        if tot > cur_s - thr:
            cur_c, cur_r, cur_s = c, r, tot
        pool = pool_insert(pool, c, r, tot)
        if tot > best_s + 1e-9:
            best_c, best_r, best_s = c, r, tot
            stall = 0
        else:
            stall += 1

    # Focused perturbation of the smallest circles of the incumbent.
    while time.perf_counter() - t0 < TIME_BUDGET * 0.93:
        pc, pr = reinsert(best_c, best_r, int(rng.choice([3, 4])))
        pc = np.clip(pc + rng.normal(0.0, 0.003, pc.shape), 0.02, 0.98)
        c, r, tot = polish(pc, pr, use_relax=False)
        if is_valid(c, r) and tot > best_s:
            best_c, best_r, best_s = c, r, tot

    # Final: SLSQP tightening, LP radii, multi-tolerance contact snap.
    c, r = optimise(best_c, best_r, maxiter=800, ftol=1e-15)
    c, r = repair(c, r)
    c, r = lp_polish(c, r)
    if is_valid(c, r) and np.sum(r) > best_s:
        best_c, best_r, best_s = c, r, float(np.sum(r))
    best_c, best_r, best_s = refine_exact(best_c, best_r, best_s)

    if not is_valid(best_c, best_r):
        best_c, best_r = repair(best_c, greedy_radii(best_c))
    return best_c, best_r, float(np.sum(best_r))

# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_packing():
    """Run the circle packing constructor for n=26"""
    centers, radii, sum_radii = construct_packing()
    return centers, radii, sum_radii


if __name__ == "__main__":
    centers, radii, sum_radii = run_packing()
    print(f"Valid: {is_valid(centers, radii)}")
    print(f"Sum of radii: {sum_radii}")
