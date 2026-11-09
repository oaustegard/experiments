# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles.

Pipeline:
  1. Seeds: staggered row / hex-like layouts, each also settled by a soft
     repulsion energy (overlap penalty + wall penalty) to spread the centers,
     plus graded and ring layouts. Each seed is refined by alternating SLSQP
     (joint centers/radii) and LP (optimal radii for fixed centers) while the
     validated sum improves.
  2. Multi-start basin hop with adaptive move selection, screened by an LP-only
     score before refinement. The hop budget is weighted toward the leading start.
  3. Final polish: LP radii with the smallest validated shrink.
Every returned packing is validated before it is accepted.
"""
import time
import numpy as np
from scipy.optimize import linprog, minimize

N = 26
_IU, _JU = np.triu_indices(N, 1)
_P = len(_IU)
_HOP_BUDGET_S = 8.0
_START_SHARE = (0.5, 0.3, 0.2)
_TOP_K = 3
_REFINE_ROUNDS = 3
_SCREEN_MARGIN = 0.12
_REPEL_RHO = 0.085
_MOVES = ("jitter", "relocate_small", "swap_small_large", "relocate_big",
          "relocate_rattler")


def _row_layout(counts, lo=0.1, hi=0.9):
    """Staggered row layout: each row has counts[k] circles spread over [lo, hi]."""
    R = len(counts)
    ys = np.linspace(lo, hi, R) if R > 1 else np.array([0.5])
    pts = []
    for k, m in enumerate(counts):
        xs = np.linspace(lo, hi, m) if m > 1 else np.array([0.5])
        for x in xs:
            pts.append((x, ys[k]))
    return np.array(pts, dtype=float)


def _hex_layout(counts, lo=0.08, hi=0.92):
    """Hex-like layout: odd rows are offset by half a pitch relative to even rows."""
    R = len(counts)
    ys = np.linspace(lo, hi, R)
    pts = []
    for k, m in enumerate(counts):
        if k % 2 == 1 and m > 1:
            edges = np.linspace(lo, hi, m + 1)
            xs = 0.5 * (edges[:-1] + edges[1:])
        else:
            xs = np.linspace(lo, hi, m) if m > 1 else np.array([0.5])
        for x in xs:
            pts.append((x, ys[k]))
    return np.array(pts, dtype=float)


def _wall_dist(c):
    return np.maximum(
        np.minimum.reduce([c[:, 0], c[:, 1], 1 - c[:, 0], 1 - c[:, 1]]), 0.0
    )


def _greedy_radii(c):
    """Valid fallback: wall limits, then proportional pairwise shrinking."""
    r = _wall_dist(c).copy()
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(c[i] - c[j])
            if r[i] + r[j] > d and r[i] + r[j] > 0:
                s = d / (r[i] + r[j])
                r[i] *= s
                r[j] *= s
    return r


def _lp_radii(centers, shrink=1e-6):
    """Optimal radii for fixed centers: max sum(r) s.t. r_i <= wall_i, r_i + r_j <= d_ij."""
    c = np.clip(centers, 0.0, 1.0)
    wall = _wall_dist(c)
    d = np.linalg.norm(c[_IU] - c[_JU], axis=1)
    A = np.zeros((_P, N))
    rows = np.arange(_P)
    A[rows, _IU] = 1.0
    A[rows, _JU] = 1.0
    try:
        res = linprog(
            -np.ones(N),
            A_ub=A,
            b_ub=d,
            bounds=[(0.0, float(w)) for w in wall],
            method="highs",
        )
        if res.status != 0 or res.x is None:
            return _greedy_radii(c)
        return np.maximum(res.x * (1.0 - shrink), 0.0)
    except Exception:
        return _greedy_radii(c)


def _is_valid(c, r, tol=1e-12):
    if np.any(r < -tol):
        return False
    if np.any(r > _wall_dist(c) + tol):
        return False
    d = np.linalg.norm(c[_IU] - c[_JU], axis=1)
    return bool(np.all(r[_IU] + r[_JU] <= d + tol))


def _polish_radii(c):
    """Smallest shrink whose LP radii pass an exact (tol=0) validation."""
    for shrink in (0.0, 1e-12, 1e-10, 1e-9, 1e-7, 1e-6):
        r = _lp_radii(c, shrink=shrink)
        if _is_valid(c, r, tol=0.0):
            return r
    return _greedy_radii(np.clip(c, 0.0, 1.0))


def _rattlers(c, r, tol=1e-7):
    """Indices of circles touching no neighbour and no wall (free slack)."""
    d = np.linalg.norm(c[_IU] - c[_JU], axis=1)
    tight = (r[_IU] + r[_JU]) >= d - tol
    contact = np.zeros(N, dtype=bool)
    contact[_IU[tight]] = True
    contact[_JU[tight]] = True
    contact |= r >= _wall_dist(c) - tol
    return np.where(~contact)[0]


def _repel(c0, rho=_REPEL_RHO, iters=300):
    """Soft-repulsion relaxation: penalize pairs closer than 2*rho and centers
    within rho of a wall. Returns relaxed centers (falls back to input on error)."""
    c0 = np.clip(np.asarray(c0, dtype=float), 0.0, 1.0)

    def energy(z):
        p = z.reshape(N, 2)
        diff = p[_IU] - p[_JU]
        d = np.sqrt((diff * diff).sum(axis=1) + 1e-12)
        ov = np.maximum(0.0, 2.0 * rho - d)
        E = float(np.sum(ov * ov))
        coef = -2.0 * ov / d
        gp = coef[:, None] * diff
        g = np.zeros((N, 2))
        np.add.at(g, _IU, gp)
        np.add.at(g, _JU, -gp)
        lo = np.maximum(0.0, rho - p)
        hi = np.maximum(0.0, p - (1.0 - rho))
        E += float(np.sum(lo * lo) + np.sum(hi * hi))
        g += -2.0 * lo + 2.0 * hi
        return E, g.ravel()

    try:
        res = minimize(
            energy,
            c0.ravel(),
            jac=True,
            method="L-BFGS-B",
            bounds=[(0.0, 1.0)] * (2 * N),
            options={"maxiter": iters},
        )
        out = np.clip(res.x.reshape(N, 2), 0.0, 1.0)
        if np.all(np.isfinite(out)):
            return out
    except Exception:
        pass
    return c0


def _refine(centers, radii, ftol=1e-12, maxiter=400):
    """Jointly optimize centers and radii with SLSQP (squared-distance constraints)."""
    z0 = np.concatenate([centers[:, 0], centers[:, 1], radii])

    def unpack(z):
        return z[:N], z[N:2 * N], z[2 * N:]

    def cons_f(z):
        x, y, r = unpack(z)
        dx = x[_IU] - x[_JU]
        dy = y[_IU] - y[_JU]
        s = r[_IU] + r[_JU]
        pair = dx * dx + dy * dy - s * s
        bnd = np.concatenate([x - r, 1 - x - r, y - r, 1 - y - r])
        return np.concatenate([pair, bnd])

    def cons_j(z):
        x, y, r = unpack(z)
        dx = x[_IU] - x[_JU]
        dy = y[_IU] - y[_JU]
        s = r[_IU] + r[_JU]
        J = np.zeros((_P + 4 * N, 3 * N))
        rows = np.arange(_P)
        J[rows, _IU] = 2 * dx
        J[rows, _JU] = -2 * dx
        J[rows, N + _IU] = 2 * dy
        J[rows, N + _JU] = -2 * dy
        J[rows, 2 * N + _IU] = -2 * s
        J[rows, 2 * N + _JU] = -2 * s
        idx = np.arange(N)
        b = _P
        J[b + idx, idx] = 1.0
        J[b + idx, 2 * N + idx] = -1.0
        J[b + N + idx, idx] = -1.0
        J[b + N + idx, 2 * N + idx] = -1.0
        J[b + 2 * N + idx, N + idx] = 1.0
        J[b + 2 * N + idx, 2 * N + idx] = -1.0
        J[b + 3 * N + idx, N + idx] = -1.0
        J[b + 3 * N + idx, 2 * N + idx] = -1.0
        return J

    def obj(z):
        return -np.sum(z[2 * N:])

    def obj_j(z):
        g = np.zeros(3 * N)
        g[2 * N:] = -1.0
        return g

    bounds = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
    res = minimize(
        obj,
        z0,
        jac=obj_j,
        method="SLSQP",
        bounds=bounds,
        constraints=[{"type": "ineq", "fun": cons_f, "jac": cons_j}],
        options={"maxiter": maxiter, "ftol": ftol},
    )
    x, y, _ = unpack(res.x)
    return np.clip(np.column_stack([x, y]), 0.0, 1.0)


def _refine_loop(c0):
    """LP radii, then alternate SLSQP and LP while the validated sum improves.

    Returns (centers, radii, sum) if valid, else None.
    """
    c = np.clip(np.asarray(c0, dtype=float), 0.0, 1.0)
    r = _lp_radii(c)
    if not _is_valid(c, r):
        return None
    cur_c, cur_r, cur_s = c, r, float(r.sum())
    for _ in range(_REFINE_ROUNDS):
        try:
            c1 = _refine(cur_c, cur_r)
            r1 = _lp_radii(c1)
        except Exception:
            break
        if _is_valid(c1, r1) and float(r1.sum()) > cur_s + 1e-10:
            cur_c, cur_r, cur_s = c1, r1, float(r1.sum())
        else:
            break
    return cur_c, cur_r, cur_s


def _gap_point(others, rng, n=800):
    """Sample points; return the one with the most clearance from others and walls."""
    cand = rng.uniform(0.03, 0.97, (n, 2))
    wall = _wall_dist(cand)
    d = np.linalg.norm(cand[:, None, :] - others[None, :, :], axis=2).min(axis=1)
    clear = np.minimum(wall, d / 2.0)
    return cand[int(np.argmax(clear))]


def _graded_seed(rng):
    """Graded start: large circles at a random subset of corners/edge midpoints,
    a central anchor, and smaller circles filling the largest interior gaps."""
    anchors = np.array(
        [
            [0.08, 0.08], [0.92, 0.08], [0.08, 0.92], [0.92, 0.92],
            [0.5, 0.06], [0.5, 0.94], [0.06, 0.5], [0.94, 0.5],
            [0.5, 0.5],
        ],
        dtype=float,
    )
    k = int(rng.integers(5, len(anchors) + 1))
    pts = anchors[:k] + rng.normal(0.0, 0.01, (k, 2))
    pts = np.clip(pts, 0.03, 0.97)
    while len(pts) < N:
        pts = np.vstack([pts, _gap_point(pts, rng)])
    return pts


def _ring_seed(rng):
    """Ring of 8 around a hex-like core, with the remaining circles gap-filled."""
    ang = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False) + rng.uniform(0, 0.4)
    ring = np.column_stack([0.5 + 0.40 * np.cos(ang), 0.5 + 0.40 * np.sin(ang)])
    pts = np.vstack([[[0.5, 0.5]], ring])
    pts = np.clip(pts, 0.05, 0.95)
    while len(pts) < N:
        pts = np.vstack([pts, _gap_point(pts, rng)])
    return pts


def _seed_centers(rng):
    base = [
        _row_layout([5, 5, 5, 5, 6]),
        _row_layout([6, 5, 6, 5, 4]),
        _row_layout([5, 6, 5, 5, 5]),
        _row_layout([6, 5, 6, 5, 4], lo=0.12, hi=0.88),
        _row_layout([5, 5, 5, 5, 6], lo=0.07, hi=0.93),
        _hex_layout([6, 5, 6, 5, 4]),
        _hex_layout([5, 6, 5, 6, 4]),
        _hex_layout([5, 5, 6, 5, 5]),
    ]
    base += [_graded_seed(rng) for _ in range(4)]
    base += [_ring_seed(rng) for _ in range(2)]
    # Settle the lattice-like seeds with soft repulsion as additional starts.
    relaxed = [_repel(s) for s in base[:8]] + [_repel(s) for s in base[8:]]
    return base + relaxed


def _perturb(c, r, rng, sigma, move):
    """Apply one named move to the layout."""
    order = np.argsort(r)
    cand = c.copy()
    if move == "jitter":
        cand = cand + rng.normal(0.0, sigma, cand.shape)
    elif move == "relocate_small":
        i = int(rng.choice(order[:6]))
        mask = np.ones(N, dtype=bool)
        mask[i] = False
        cand[i] = _gap_point(cand[mask], rng)
        cand = cand + rng.normal(0.0, 0.2 * sigma, cand.shape)
    elif move == "swap_small_large":
        i = int(rng.choice(order[:8]))
        j = int(rng.choice(order[-8:]))
        cand[[i, j]] = cand[[j, i]]
        cand = cand + rng.normal(0.0, 0.2 * sigma, cand.shape)
    elif move == "relocate_rattler":
        ratt = _rattlers(c, r)
        pool = ratt if len(ratt) > 0 else order[:3]
        i = int(rng.choice(pool))
        mask = np.ones(N, dtype=bool)
        mask[i] = False
        cand[i] = _gap_point(cand[mask], rng)
        cand = cand + rng.normal(0.0, 0.2 * sigma, cand.shape)
    else:  # relocate_big
        i = int(rng.choice(order[-6:]))
        mask = np.ones(N, dtype=bool)
        mask[i] = False
        cand[i] = _gap_point(cand[mask], rng)
        cand = cand + rng.normal(0.0, 0.2 * sigma, cand.shape)
    return np.clip(cand, 0.02, 0.98)


def _hop(best_c, best_r, best_s, rng, budget_s):
    """Time-budgeted basin hop with adaptive move selection and LP screening."""
    cur_c, cur_r, cur_s = best_c, best_r, best_s
    tries = np.full(len(_MOVES), 2.0)
    succ = np.ones(len(_MOVES))
    t0 = time.time()
    while True:
        elapsed = time.time() - t0
        if elapsed >= budget_s:
            break
        frac = elapsed / budget_s
        rate = succ / tries
        w = rate / rate.sum()
        m = int(rng.choice(len(_MOVES), p=w))
        base_sigma = 0.02 * (1.0 - frac) + 0.002
        sigma = base_sigma * (0.5 + rate[m])
        tries[m] += 1.0
        try:
            cand = _perturb(cur_c, cur_r, rng, sigma, _MOVES[m])
            if float(_lp_radii(cand).sum()) < cur_s - _SCREEN_MARGIN:
                continue
            out = _refine_loop(cand)
        except Exception:
            continue
        if out is None:
            continue
        c1, r1, s1 = out
        if s1 > cur_s + 1e-9:
            succ[m] += 1.0
            cur_c, cur_r, cur_s = c1, r1, s1
        elif s1 > cur_s - 0.002 and rng.random() < 0.1:
            cur_c, cur_r, cur_s = c1, r1, s1
        if s1 > best_s + 1e-9:
            best_c, best_r, best_s = c1, r1, s1
    return best_c, best_r, best_s


def construct_packing():
    rng = np.random.default_rng(0)

    results = []
    for c0 in _seed_centers(rng):
        try:
            out = _refine_loop(c0)
        except Exception:
            out = None
        if out is not None and _is_valid(out[0], out[1]):
            results.append(out)
    results.sort(key=lambda t: -t[2])

    starts = []
    for c, r, s in results:
        if all(abs(s - s2) > 1e-6 for _, _, s2 in starts):
            starts.append((c, r, s))
        if len(starts) >= _TOP_K:
            break

    if not starts:
        c = _row_layout([5, 5, 5, 5, 6])
        r = _greedy_radii(c)
        starts = [(c, r, float(r.sum()))]

    # Weighted hop budget: the leading start gets the largest share.
    shares = list(_START_SHARE[:len(starts)])
    total = sum(shares)
    best_c, best_r, best_s = starts[0]
    for k, (c, r, s) in enumerate(starts):
        budget = _HOP_BUDGET_S * shares[k] / total
        hop_rng = np.random.default_rng(12345 + k)
        try:
            hc, hr, hs = _hop(c, r, s, hop_rng, budget)
        except Exception:
            continue
        if hs > best_s + 1e-9 and _is_valid(hc, hr):
            best_c, best_r, best_s = hc, hr, hs

    try:
        polished = _polish_radii(best_c)
        if _is_valid(best_c, polished, tol=0.0) and float(polished.sum()) >= best_s - 1e-12:
            best_r = polished
    except Exception:
        pass

    if not _is_valid(best_c, best_r):
        best_r = _greedy_radii(best_c)

    return best_c, best_r, float(np.sum(best_r))


def compute_max_radii(centers):
    """Maximum valid radii for fixed centers (LP-based)."""
    return _lp_radii(np.asarray(centers, dtype=float))


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_packing():
    """Run the circle packing constructor for n=26"""
    centers, radii, sum_radii = construct_packing()
    return centers, radii, sum_radii


if __name__ == "__main__":
    centers, radii, sum_radii = run_packing()
    print(f"Sum of radii: {sum_radii}")
