# EVOLVE-BLOCK-START
"""Row seeds + LP radii + joint SLSQP polish + pooled hole-reinsertion / kick
basin hopping over the top distinct packings, time-bounded, n=26."""
import time
import numpy as np
from scipy.optimize import linprog, minimize

N = 26
ROW_PATTERNS = [
    [5, 6, 5, 6, 4],
    [6, 5, 6, 5, 4],
    [5, 5, 6, 5, 5],
    [4, 6, 5, 6, 5],
    [5, 6, 4, 6, 5],
    [6, 5, 5, 5, 5],
    [5, 5, 5, 5, 6],
    [5, 4, 5, 4, 5, 3],
]
N_RANDOM_STARTS = 6
POOL_SIZE = 5
TIME_BUDGET = 40.0
SIGMAS = (0.03, 0.015, 0.008)

_IU, _JU = np.triu_indices(N, 1)
_M = len(_IU)
_ROWS = np.arange(_M)
_IDX = np.arange(N)


def make_layout(counts, margin=0.08):
    """Staggered rows: row k has counts[k] circles spread across the width."""
    ys = np.linspace(margin, 1 - margin, len(counts))
    pts = []
    for k, m in enumerate(counts):
        if m == 1:
            xs = np.array([0.5])
        else:
            xs = np.linspace(margin, 1 - margin, m)
            if k % 2 == 1:
                shift = 0.5 * (xs[1] - xs[0])
                xs = np.clip(xs + shift, margin, 1 - margin)
        for x in xs:
            pts.append([x, ys[k]])
    return np.array(pts, dtype=float)


def make_graded_layout(counts, margin=0.07):
    """Row layout with outer circles nudged toward the walls for mixed sizes."""
    c = make_layout(counts, margin=margin)
    centre = np.array([0.5, 0.5])
    d = np.linalg.norm(c - centre, axis=1)
    scale = 1.0 + 0.04 * (d / (d.max() + 1e-9))
    c = centre + (c - centre) * scale[:, None]
    return np.clip(c, margin, 1 - margin)


def wall_of(c):
    return np.minimum.reduce([c[:, 0], c[:, 1], 1 - c[:, 0], 1 - c[:, 1]])


def make_feasible(c, r):
    """Uniformly shrink radii only as much as needed for strict feasibility."""
    r = np.clip(np.asarray(r, dtype=float), 0.0, None)
    wall = np.maximum(wall_of(c), 0.0)
    factor = 1.0
    pos = r > 0
    if np.any(pos):
        factor = min(factor, float(np.min(wall[pos] / r[pos])))
    d = np.linalg.norm(c[_IU] - c[_JU], axis=1)
    s = r[_IU] + r[_JU]
    m = s > 0
    if np.any(m):
        factor = min(factor, float(np.min(d[m] / s[m])))
    factor = max(0.0, min(1.0, factor * (1 - 1e-12)))
    return r * factor


def lp_radii(c):
    """LP-optimal radii for fixed centers, then made strictly feasible."""
    wall = np.maximum(wall_of(c), 0.0)
    d = np.linalg.norm(c[_IU] - c[_JU], axis=1)
    A = np.zeros((_M, N))
    A[_ROWS, _IU] = 1.0
    A[_ROWS, _JU] = 1.0
    bounds = [(0.0, float(w)) for w in wall]
    try:
        res = linprog(-np.ones(N), A_ub=A, b_ub=d, bounds=bounds, method="highs")
    except Exception:
        return None
    if not res.success:
        return None
    return make_feasible(c, np.maximum(res.x, 0.0))


def total(c):
    r = lp_radii(c)
    if r is None:
        return -1.0, None
    return float(np.sum(r)), r


def joint_slsqp(c0, r0, rounds=2, maxiter=400):
    """Jointly optimize (x, y, r) with SLSQP and analytic Jacobians."""

    def obj(z):
        return -np.sum(z[2 * N:])

    def obj_grad(z):
        g = np.zeros_like(z)
        g[2 * N:] = -1.0
        return g

    def cons(z):
        x, y, r = z[:N], z[N:2 * N], z[2 * N:]
        dx = x[_IU] - x[_JU]
        dy = y[_IU] - y[_JU]
        s = r[_IU] + r[_JU]
        pair = dx * dx + dy * dy - s * s
        wall = np.concatenate([x - r, 1 - x - r, y - r, 1 - y - r])
        return np.concatenate([pair, wall])

    def cons_jac(z):
        x, y, r = z[:N], z[N:2 * N], z[2 * N:]
        dx = x[_IU] - x[_JU]
        dy = y[_IU] - y[_JU]
        s = r[_IU] + r[_JU]
        J = np.zeros((_M + 4 * N, 3 * N))
        J[_ROWS, _IU] = 2 * dx
        J[_ROWS, _JU] = -2 * dx
        J[_ROWS, N + _IU] = 2 * dy
        J[_ROWS, N + _JU] = -2 * dy
        J[_ROWS, 2 * N + _IU] = -2 * s
        J[_ROWS, 2 * N + _JU] = -2 * s
        b = _M
        J[b + _IDX, _IDX] = 1.0
        J[b + _IDX, 2 * N + _IDX] = -1.0
        b = _M + N
        J[b + _IDX, _IDX] = -1.0
        J[b + _IDX, 2 * N + _IDX] = -1.0
        b = _M + 2 * N
        J[b + _IDX, N + _IDX] = 1.0
        J[b + _IDX, 2 * N + _IDX] = -1.0
        b = _M + 3 * N
        J[b + _IDX, N + _IDX] = -1.0
        J[b + _IDX, 2 * N + _IDX] = -1.0
        return J

    z = np.concatenate([c0[:, 0], c0[:, 1], r0])
    bounds = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
    for _ in range(rounds):
        try:
            res = minimize(obj, z, jac=obj_grad, method="SLSQP", bounds=bounds,
                           constraints=[{"type": "ineq", "fun": cons, "jac": cons_jac}],
                           options={"maxiter": maxiter, "ftol": 1e-12})
            z = res.x
        except Exception:
            break
    c = np.clip(np.column_stack([z[:N], z[N:2 * N]]), 1e-6, 1 - 1e-6)
    return c


def polish(c0, r0, rounds=2, maxiter=400):
    """SLSQP polish, then certify centers by LP; returns (sum, centers, radii) or None."""
    try:
        c = joint_slsqp(c0, r0, rounds=rounds, maxiter=maxiter)
    except Exception:
        return None
    s, r = total(c)
    if r is None:
        return None
    return s, c, r


def hole_point(others, other_r, rng, samples=3000):
    """Point with the largest clearance from other circles and the walls."""
    pts = rng.uniform(0.02, 0.98, size=(samples, 2))
    wall = np.minimum.reduce([pts[:, 0], pts[:, 1], 1 - pts[:, 0], 1 - pts[:, 1]])
    d = np.linalg.norm(pts[:, None, :] - others[None, :, :], axis=2) - other_r[None, :]
    clear = np.minimum(wall, d.min(axis=1))
    return pts[int(np.argmax(clear))]


def pool_offer(pool, res, k=POOL_SIZE):
    """Keep the top-k distinct packings (distinct by score to 1e-7)."""
    if res is None or res[0] <= 0:
        return
    for p in pool:
        if abs(p[0] - res[0]) < 1e-7:
            return
    pool.append(res)
    pool.sort(key=lambda t: -t[0])
    del pool[k:]


def construct_packing():
    t_start = time.perf_counter()
    rng = np.random.default_rng(0)
    pool = []

    # 1) Row seeds (uniform and graded), scored by the LP.
    cands = []
    for counts in ROW_PATTERNS:
        for make in (make_layout, make_graded_layout):
            c = make(counts)
            if len(c) != N:
                continue
            s, r = total(c)
            if r is not None:
                cands.append((s, c, r))
    cands.sort(key=lambda t: -t[0])
    for res in cands:
        pool_offer(pool, res)

    # 2) Random starts polished by joint SLSQP.
    for _ in range(N_RANDOM_STARTS):
        if time.perf_counter() - t_start > 0.3 * TIME_BUDGET:
            break
        c0 = rng.uniform(0.08, 0.92, size=(N, 2))
        pool_offer(pool, polish(c0, np.full(N, 0.06)))

    # 3) Joint SLSQP polish from the top seeds.
    for _, c0, r0 in cands[:3]:
        pool_offer(pool, polish(c0, r0))

    if not pool:
        grid = np.array([[0.1 + 0.8 * (i % 6) / 5, 0.1 + 0.8 * (i // 6) / 4]
                         for i in range(N)])
        pool.append((0.0, grid, np.zeros(N)))

    # 4) Basin hopping spread across the pool: hole reinsertion and kicks.
    it = 0
    while time.perf_counter() - t_start < TIME_BUDGET:
        _, base_c, base_r = pool[it % len(pool)]
        if it % 2 == 0:
            k = int(np.argmin(base_r))
            keep = np.arange(N) != k
            cand = base_c.copy()
            cand[k] = hole_point(base_c[keep], base_r[keep], rng)
        else:
            sigma = SIGMAS[(it // 2) % len(SIGMAS)]
            cand = np.clip(base_c + rng.normal(0.0, sigma, size=base_c.shape),
                           0.02, 0.98)
        pool_offer(pool, polish(cand, base_r * 0.9, rounds=1, maxiter=300))
        it += 1

    # 5) Final certification: LP radii on the best centers, then strict feasibility.
    _, centers, best_r = pool[0]
    radii = lp_radii(centers)
    if radii is None:
        radii = make_feasible(centers, best_r)
    return centers, radii, float(np.sum(radii))

# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_packing():
    """Run the circle packing constructor for n=26"""
    centers, radii, sum_radii = construct_packing()
    return np.asarray(centers, dtype=float), np.asarray(radii, dtype=float), float(sum_radii)


if __name__ == "__main__":
    centers, radii, sum_radii = run_packing()
    print(f"Sum of radii: {sum_radii}")
