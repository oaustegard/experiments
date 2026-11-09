# EVOLVE-BLOCK-START
"""Annealed elite-anchored search for 26 circles in the unit square.

Pipeline:
  1. Seeds (staggered rows, mixed boundary/interior, hex patches, graded ring)
     and their square symmetries are scored by the exact LP; the best are
     refined by _solve().
  2. _solve(): exact LP baseline; the penalty surrogate is only used when the
     baseline is weak. Then SLSQP -> LP. Never worse than the LP of its input.
  3. Search: a walker perturbs itself or an elite, screens with the LP, and
     refines promising candidates with SLSQP. Acceptance is annealed (worse
     LP-valid packings are accepted with probability exp(delta / T)); the
     global best and an elite pool are tracked separately; gap restarts fire
     when stalled.
  4. Final basin hops run on several elites, using swap, jitter, and
     gap-relocation moves; only LP-verified gains are kept.
  5. Every returned packing is strictly valid (shrink ladder on final radii).
"""
import math
import time
import numpy as np
from scipy.optimize import linprog, minimize

N = 26
PI, PJ = np.triu_indices(N, 1)
M = len(PI)
IDX = np.arange(N)

ROW_PATTERNS = [
    [5, 6, 5, 5, 5],
    [6, 5, 6, 5, 4],
    [5, 5, 6, 5, 5],
    [4, 6, 5, 6, 5],
    [5, 5, 5, 5, 6],
    [4, 5, 4, 5, 4, 4],
    [5, 4, 5, 4, 4, 4],
    [4, 6, 4, 6, 6],
    [3, 5, 5, 5, 5, 3],
]

HEX_SPACINGS = (0.18, 0.19, 0.20, 0.21, 0.22, 0.23)
TIME_BUDGET = 12.0
ELITE_SIZE = 6
SCREEN_MARGIN = 0.02
SURROGATE_THRESHOLD = 2.4   # only run the penalty surrogate on weak baselines
SEED_TIME_FRAC = 0.35
SEARCH_END_FRAC = 0.62
HOP_END_FRAC = 0.95
HOP_ELITES = 3
ANNEAL_T_HI = 0.01          # initial acceptance temperature (sum units)
ANNEAL_T_LO = 1e-5          # final acceptance temperature
WANDER_LIMIT = 0.03         # walker may fall this far below the incumbent
SHRINK_LADDER = (1.0, 1.0 - 1e-12, 1.0 - 1e-10, 1.0 - 1e-8,
                 1.0 - 1e-6, 0.99999, 0.9999, 0.999)


def _layout(rows, rng=None, jitter=0.0):
    """Staggered rows: even rows span full width, odd rows are offset."""
    pts = []
    ys = np.linspace(0.1, 0.9, len(rows))
    for k, cnt in enumerate(rows):
        if k % 2 == 0:
            xs = np.linspace(0.1, 0.9, cnt)
        else:
            xs = np.linspace(0.1 + 0.4 / cnt, 0.9 - 0.4 / cnt, cnt)
        for x in xs:
            pts.append([x, ys[k]])
    c = np.array(pts, dtype=float)
    if rng is not None and jitter > 0:
        c += rng.normal(scale=jitter, size=c.shape)
    return np.clip(c, 0.05, 0.95)


def _mixed_seed(rng, jitter=0.0):
    """Eight boundary circles plus 18 interior circles."""
    pts = [[0.07, 0.07], [0.93, 0.07], [0.07, 0.93], [0.93, 0.93],
           [0.5, 0.05], [0.5, 0.95], [0.05, 0.5], [0.95, 0.5]]
    xs = np.linspace(0.2, 0.8, 6)
    for y in (0.3, 0.5, 0.7):
        for x in xs:
            pts.append([x, y])
    c = np.array(pts, dtype=float)
    if jitter > 0:
        c += rng.normal(scale=jitter, size=c.shape)
    return np.clip(c, 0.03, 0.97)


def _hex_seed(rng, spacing, jitter=0.01):
    """Hexagonal lattice clipped to the square, adjusted to N points."""
    h = spacing * np.sqrt(3.0) / 2.0
    pts = []
    y = 0.08
    row = 0
    while y <= 0.92:
        off = spacing / 2.0 if row % 2 else 0.0
        x = 0.08 + off
        while x <= 0.92:
            pts.append([x, y])
            x += spacing
        y += h
        row += 1
    pts = np.array(pts, dtype=float)
    if len(pts) > N:
        keep = rng.choice(len(pts), size=N, replace=False)
        pts = pts[keep]
    while len(pts) < N:
        pts = np.vstack([pts, rng.uniform(0.15, 0.85, size=(1, 2))])
    pts += rng.normal(scale=jitter, size=pts.shape)
    return np.clip(pts, 0.05, 0.95)


def _graded_seed(rng, jitter=0.008):
    """12 perimeter circles (inset ring) + 14 interior staggered circles."""
    s = 0.07
    pts = []
    for k in range(12):
        t = 4.0 * k / 12.0
        side = int(t)
        f = t - side
        if side == 0:
            p = (s + f * (1 - 2 * s), s)
        elif side == 1:
            p = (1 - s, s + f * (1 - 2 * s))
        elif side == 2:
            p = (1 - s - f * (1 - 2 * s), 1 - s)
        else:
            p = (s, 1 - s - f * (1 - 2 * s))
        pts.append(list(p))
    ys = np.linspace(0.3, 0.75, 4)
    counts = [4, 4, 3, 3]
    for k, (y, cnt) in enumerate(zip(ys, counts)):
        lo, hi = (0.28, 0.72)
        if k % 2 == 0:
            xs = np.linspace(lo, hi, cnt)
        else:
            xs = np.linspace(lo + 0.06, hi - 0.06, cnt)
        for x in xs:
            pts.append([x, y])
    c = np.array(pts, dtype=float)
    c += rng.normal(scale=jitter, size=c.shape)
    return np.clip(c, 0.04, 0.96)


def _symmetric_variants(c):
    """Square symmetries (transpose, x-mirror, 90-degree rotation)."""
    transpose = c[:, ::-1].copy()
    mirror = np.column_stack([1.0 - c[:, 0], c[:, 1]])
    rotate = np.column_stack([1.0 - c[:, 1], c[:, 0]])
    return [transpose, mirror, rotate]


def _lp_radii(c):
    """Exact max-sum radii for fixed centers (LP). None on failure."""
    try:
        d = np.sqrt(np.sum((c[PI] - c[PJ]) ** 2, axis=1))
        A_pair = np.zeros((M, N))
        A_pair[np.arange(M), PI] = 1.0
        A_pair[np.arange(M), PJ] = 1.0
        A_wall = np.zeros((4 * N, N))
        A_wall[IDX, IDX] = 1.0
        A_wall[N + IDX, IDX] = 1.0
        A_wall[2 * N + IDX, IDX] = 1.0
        A_wall[3 * N + IDX, IDX] = 1.0
        b_wall = np.concatenate([c[:, 0], 1.0 - c[:, 0],
                                 c[:, 1], 1.0 - c[:, 1]])
        A = np.vstack([A_pair, A_wall])
        b = np.maximum(np.concatenate([d, b_wall]), 0.0)
        res = linprog(-np.ones(N), A_ub=A, b_ub=b,
                      bounds=[(0.0, None)] * N, method="highs")
        if not res.success or res.x is None or not np.all(np.isfinite(res.x)):
            return None
        return np.maximum(res.x, 0.0)
    except Exception:
        return None


def _repair(c, r):
    """Shrink only violating radii so every wall and pair constraint holds."""
    r = np.maximum(np.asarray(r, dtype=float), 0.0)
    wall = np.minimum(np.minimum(c[:, 0], c[:, 1]),
                      np.minimum(1.0 - c[:, 0], 1.0 - c[:, 1]))
    r = np.minimum(r, np.maximum(wall, 0.0))
    d = np.sqrt(np.sum((c[PI] - c[PJ]) ** 2, axis=1))
    for _ in range(100):
        bad = np.where(r[PI] + r[PJ] > d)[0]
        if bad.size == 0:
            break
        for k in bad:
            i, j = PI[k], PJ[k]
            s = r[i] + r[j]
            if s > d[k]:
                f = d[k] / s * (1.0 - 1e-12)
                r[i] *= f
                r[j] *= f
    return np.maximum(r, 0.0)


def _is_valid(c, r, tol=0.0):
    if not np.all(np.isfinite(r)) or np.any(r < -tol):
        return False
    if np.any(c - r[:, None] < -tol) or np.any(c + r[:, None] > 1.0 + tol):
        return False
    d = np.sqrt(np.sum((c[PI] - c[PJ]) ** 2, axis=1))
    return bool(np.all(r[PI] + r[PJ] <= d + tol))


def _strict_valid(c, r):
    """Repair, then apply the smallest shrink from the ladder that is valid."""
    r = _repair(c, r)
    for f in SHRINK_LADDER:
        rr = r * f
        if _is_valid(c, rr):
            return rr
    return None


def _valid_lp(c):
    """(centers, radii, sum) with exact LP radii, strictly valid, or None."""
    r = _lp_radii(c)
    if r is None:
        return None
    r = _strict_valid(c, r)
    if r is None:
        return None
    return (c.copy(), r, float(r.sum()))


def _pen_fg(z, w):
    """Penalised objective -sum(r) + w * squared violations, with gradient."""
    c = z[:2 * N].reshape(N, 2)
    r = z[2 * N:]
    diff = c[PI] - c[PJ]
    d = np.sqrt(np.sum(diff ** 2, axis=1) + 1e-12)
    ov = np.maximum(0.0, r[PI] + r[PJ] - d)
    f = -np.sum(r) + w * np.sum(ov ** 2)
    coef = 2.0 * w * ov
    u = diff / d[:, None]
    gc = np.zeros((N, 2))
    gr = -np.ones(N)
    np.add.at(gc, PI, -coef[:, None] * u)
    np.add.at(gc, PJ, coef[:, None] * u)
    np.add.at(gr, PI, coef)
    np.add.at(gr, PJ, coef)
    x, y = c[:, 0], c[:, 1]
    v = np.maximum(0.0, r - x)
    f += w * np.sum(v ** 2)
    gc[:, 0] += -2.0 * w * v
    gr += 2.0 * w * v
    v = np.maximum(0.0, r - (1.0 - x))
    f += w * np.sum(v ** 2)
    gc[:, 0] += 2.0 * w * v
    gr += 2.0 * w * v
    v = np.maximum(0.0, r - y)
    f += w * np.sum(v ** 2)
    gc[:, 1] += -2.0 * w * v
    gr += 2.0 * w * v
    v = np.maximum(0.0, r - (1.0 - y))
    f += w * np.sum(v ** 2)
    gc[:, 1] += 2.0 * w * v
    gr += 2.0 * w * v
    return f, np.concatenate([gc.ravel(), gr])


def _surrogate(c0):
    """Penalty continuation with L-BFGS; returns candidate centers."""
    z = np.concatenate([c0.ravel(), np.full(N, 0.05)])
    bounds = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
    for w in (10.0, 100.0, 1e3, 1e4, 1e5):
        try:
            res = minimize(_pen_fg, z, args=(w,), jac=True, method="L-BFGS-B",
                           bounds=bounds, options={"maxiter": 250})
            if np.all(np.isfinite(res.x)):
                z = res.x
        except Exception:
            break
    return np.clip(z[:2 * N].reshape(N, 2), 0.0, 1.0)


def _ineq(z):
    c = z[:2 * N].reshape(N, 2)
    r = z[2 * N:]
    diff = c[PI] - c[PJ]
    pair = np.sum(diff ** 2, axis=1) - (r[PI] + r[PJ]) ** 2
    x, y = c[:, 0], c[:, 1]
    wall = np.concatenate([x - r, 1.0 - x - r, y - r, 1.0 - y - r])
    return np.concatenate([pair, wall])


def _ineq_jac(z):
    c = z[:2 * N].reshape(N, 2)
    r = z[2 * N:]
    diff = c[PI] - c[PJ]
    J = np.zeros((M + 4 * N, 3 * N))
    rows = np.arange(M)
    for k in range(2):
        J[rows, 2 * PI + k] = 2.0 * diff[:, k]
        J[rows, 2 * PJ + k] = -2.0 * diff[:, k]
    s = r[PI] + r[PJ]
    J[rows, 2 * N + PI] = -2.0 * s
    J[rows, 2 * N + PJ] = -2.0 * s
    base = M
    J[base + IDX, 2 * IDX] = 1.0
    J[base + IDX, 2 * N + IDX] = -1.0
    base += N
    J[base + IDX, 2 * IDX] = -1.0
    J[base + IDX, 2 * N + IDX] = -1.0
    base += N
    J[base + IDX, 2 * IDX + 1] = 1.0
    J[base + IDX, 2 * N + IDX] = -1.0
    base += N
    J[base + IDX, 2 * IDX + 1] = -1.0
    J[base + IDX, 2 * N + IDX] = -1.0
    return J


def _objective(z):
    return -np.sum(z[2 * N:])


def _obj_grad(z):
    g = np.zeros_like(z)
    g[2 * N:] = -1.0
    return g


def _smooth_optimize(c0, r0, maxiter=400):
    """Joint SLSQP over centers and radii; returns candidate centers."""
    z0 = np.concatenate([c0.ravel(), r0])
    bounds = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
    cons = [{"type": "ineq", "fun": _ineq, "jac": _ineq_jac}]
    z = z0
    try:
        res = minimize(_objective, z0, jac=_obj_grad, method="SLSQP",
                       bounds=bounds, constraints=cons,
                       options={"maxiter": maxiter, "ftol": 1e-14})
        if np.all(np.isfinite(res.x)):
            z = res.x
    except Exception:
        z = z0
    return np.clip(z[:2 * N].reshape(N, 2), 0.0, 1.0)


def _refine(c0, r0=None, rounds=3):
    """SLSQP -> LP loop. The LP of the input is the baseline; never worse."""
    if r0 is None:
        r0 = np.full(N, 0.08)
    best = _valid_lp(c0)
    c = c0
    for _ in range(rounds):
        c_new = _smooth_optimize(c, r0)
        cand = _valid_lp(c_new)
        if cand is None:
            break
        if best is None or cand[2] > best[2] + 1e-10:
            best = cand
            c = c_new
            r0 = cand[1]
        else:
            break
    return best


def _solve(c0, r0=None):
    """LP baseline; penalty surrogate only if the baseline is weak; SLSQP->LP."""
    base = _valid_lp(c0)
    cands = [base] if base is not None else []
    if base is None or base[2] < SURROGATE_THRESHOLD:
        try:
            cands.append(_valid_lp(_surrogate(c0)))
        except Exception:
            pass
    cands = [q for q in cands if q is not None]
    if not cands:
        return None
    start = max(cands, key=lambda q: q[2])
    try:
        ref = _refine(start[0], start[1] if r0 is None else r0)
    except Exception:
        ref = None
    if ref is not None and ref[2] > start[2]:
        return ref
    return start


def _solve_fast(c0, r0):
    """Search-loop variant: SLSQP->LP directly from perturbed centers."""
    try:
        return _refine(c0, r0)
    except Exception:
        return None


def _gap_point(c, r, rng, samples=3000):
    """Random point with the largest clearance to walls and other circles."""
    pts = rng.uniform(0.02, 0.98, size=(samples, 2))
    wall = np.minimum(np.minimum(pts[:, 0], 1.0 - pts[:, 0]),
                      np.minimum(pts[:, 1], 1.0 - pts[:, 1]))
    D = np.linalg.norm(pts[:, None, :] - c[None, :, :], axis=2) - r[None, :]
    clear = np.minimum(wall, D.min(axis=1))
    return pts[np.argmax(clear)]


def _perturb(c, r, rng):
    """Gap relocation, random swap, small-large swap, or global jitter."""
    cand = c.copy()
    u = rng.random()
    if u < 0.35:
        k = int(rng.integers(1, 3))
        order = np.argsort(r)[:6]
        movers = rng.choice(order, size=min(k, len(order)), replace=False)
        for i in movers:
            keep = np.delete(IDX, i)
            cand[i] = _gap_point(cand[keep], r[keep], rng)
        cand += rng.normal(scale=0.004, size=cand.shape)
    elif u < 0.6:
        i, j = rng.choice(N, size=2, replace=False)
        cand[[i, j]] = cand[[j, i]]
        cand += rng.normal(scale=0.003, size=cand.shape)
    elif u < 0.8:
        i = int(rng.choice(np.argsort(r)[:4]))
        j = int(rng.choice(np.argsort(-r)[:4]))
        cand[[i, j]] = cand[[j, i]]
        cand += rng.normal(scale=0.003, size=cand.shape)
    else:
        cand += rng.normal(scale=0.008, size=cand.shape)
    return np.clip(cand, 0.02, 0.98)


def _gap_restart(c, r, rng):
    """Move 3-4 of the smallest circles into the largest gaps."""
    cand = c.copy()
    k = int(rng.integers(3, 5))
    order = np.argsort(r)[:8]
    movers = rng.choice(order, size=min(k, len(order)), replace=False)
    for i in movers:
        keep = np.delete(IDX, i)
        cand[i] = _gap_point(cand[keep], r[keep], rng)
    cand += rng.normal(scale=0.01, size=cand.shape)
    return np.clip(cand, 0.02, 0.98)


def _hop_move(c, r, rng):
    """Hop move for the final stage: swap, jitter, or gap relocation."""
    small = np.argsort(r)[:5]
    u = rng.random()
    if u < 0.4:
        cand = c.copy()
        i, j = rng.choice(small, size=2, replace=False)
        cand[[i, j]] = cand[[j, i]]
        cand += rng.normal(scale=0.002, size=cand.shape)
    elif u < 0.7:
        cand = c.copy()
        i = int(rng.choice(small))
        cand[i] += rng.normal(scale=0.002, size=2)
    else:
        cand = _gap_restart(c, r, rng)
    return np.clip(cand, 0.02, 0.98)


def _final_hops(incumbent, rng, t_end):
    """Basin hops; only LP-verified gains are kept."""
    cur = incumbent
    while time.time() < t_end:
        c, r, _ = cur
        cand = _hop_move(c, r, rng)
        new = _solve_fast(cand, r)
        if new is None:
            new = _valid_lp(cand)
        if new is not None and new[2] > cur[2] + 1e-10:
            cur = new
    return cur


def construct_packing(time_budget=TIME_BUDGET, seed=7):
    t0 = time.time()
    rng = np.random.default_rng(seed)

    def elapsed():
        return time.time() - t0

    best = [None]    # (centers, radii, sum)
    elites = []      # diverse strong layouts

    def update(res):
        if res is None:
            return
        if best[0] is None or res[2] > best[0][2] + 1e-10:
            best[0] = res
        if all(abs(res[2] - e[2]) > 1e-6 for e in elites):
            elites.append(res)
            elites.sort(key=lambda e: -e[2])
            del elites[ELITE_SIZE:]

    def safe_solve(c0, r0=None):
        try:
            return _solve(c0, r0)
        except Exception:
            return None

    def fresh_seed():
        u = rng.random()
        if u < 0.25:
            return _mixed_seed(rng, 0.02)
        if u < 0.5:
            rows = ROW_PATTERNS[int(rng.integers(len(ROW_PATTERNS)))]
            return _layout(rows, rng, 0.02)
        if u < 0.75:
            return _graded_seed(rng)
        sp = HEX_SPACINGS[int(rng.integers(len(HEX_SPACINGS)))]
        return _hex_seed(rng, sp, 0.015)

    # 1. Seeds: row, mixed and hex layouts plus their square symmetries.
    base_seeds = [_layout(rows) for rows in ROW_PATTERNS]
    base_seeds.append(_mixed_seed(rng))
    base_seeds += [_hex_seed(rng, sp, 0.01) for sp in HEX_SPACINGS]
    base_seeds += [_graded_seed(rng) for _ in range(3)]
    base_seeds += [_layout(rows, rng, 0.015) for rows in ROW_PATTERNS[:4]]
    seeds = list(base_seeds)
    for s0 in base_seeds[:len(ROW_PATTERNS) + 1]:
        seeds.extend(_symmetric_variants(s0))

    for s0 in seeds:
        try:
            update(_valid_lp(s0))
        except Exception:
            pass
    for s0 in seeds:
        if elapsed() > SEED_TIME_FRAC * time_budget:
            break
        update(safe_solve(s0))

    while best[0] is None and elapsed() < 0.5 * time_budget:
        update(safe_solve(fresh_seed()))

    # 2. Annealed walker with elite-anchored moves and best-tracking.
    if best[0] is not None:
        stall = 0
        cur = best[0]
        search_start = elapsed()
        search_end = SEARCH_END_FRAC * time_budget
        span = max(1e-9, search_end - search_start)
        while elapsed() < search_end:
            frac = (elapsed() - search_start) / span
            temp = ANNEAL_T_HI * (1.0 - frac) + ANNEAL_T_LO

            use_elite = len(elites) > 1 and rng.random() < 0.3
            base = elites[int(rng.integers(len(elites)))] if use_elite else cur
            try:
                cand = _perturb(base[0], base[1], rng)
                screen = _valid_lp(cand)
                if screen is None:
                    stall += 1
                    continue
                if screen[2] >= base[2] - SCREEN_MARGIN:
                    new = _solve_fast(cand, base[1])
                    if new is None:
                        new = screen
                else:
                    new = screen
            except Exception:
                new = None
            if new is None:
                stall += 1
                continue

            update(new)
            if new[2] > cur[2] + 1e-10:
                stall = 0
            else:
                stall += 1

            # Annealed acceptance on the exact LP sum.
            if new[2] >= cur[2]:
                cur = new
            else:
                delta = new[2] - cur[2]
                if rng.random() < math.exp(delta / max(temp, 1e-12)):
                    cur = new
            # Do not wander too far below the incumbent.
            if cur[2] < best[0][2] - WANDER_LIMIT:
                cur = best[0]

            if stall > 60 and elapsed() < 0.5 * time_budget:
                stall = 0
                pick = elites[int(rng.integers(min(3, len(elites))))]
                if rng.random() < 0.6:
                    seed_c = _gap_restart(pick[0], pick[1], rng)
                else:
                    seed_c = fresh_seed()
                update(safe_solve(seed_c))
                cur = best[0]

        # 3. Final polish of the incumbent.
        final = best[0]
        polished = safe_solve(final[0], final[1])
        for cand in (final, polished):
            if cand is None:
                continue
            checked = _valid_lp(cand[0])
            if checked is not None and checked[2] > best[0][2]:
                best[0] = checked

        # 4. Final basin hops on the top elites, over the remaining time.
        try:
            pool = [best[0]]
            for e in elites:
                if len(pool) >= HOP_ELITES:
                    break
                if all(abs(e[2] - p[2]) > 1e-6 for p in pool):
                    pool.append(e)
            t_hop_end = t0 + HOP_END_FRAC * time_budget
            for k, e in enumerate(pool):
                now = time.time()
                if now >= t_hop_end:
                    break
                slice_end = now + (t_hop_end - now) / (len(pool) - k)
                hopped = _final_hops(e, rng, slice_end)
                update(hopped)
            if best[0] is not None:
                best[0] = max([best[0]] + [q for q in elites],
                              key=lambda q: q[2])
        except Exception:
            pass

    if best[0] is not None:
        c, r, _ = best[0]
        r_fin = _strict_valid(c, r)
        if r_fin is not None:
            return c, r_fin, float(np.sum(r_fin))

    # Fallback: trivially valid circles on a grid.
    g = np.linspace(0.1, 0.9, 6)
    pts = np.array([[x, y] for x in g for y in g])[:N]
    r0 = np.minimum(np.minimum(pts[:, 0], 1.0 - pts[:, 0]),
                    np.minimum(pts[:, 1], 1.0 - pts[:, 1]))
    radii = _repair(pts, r0)
    return pts, radii, float(np.sum(radii))


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_packing():
    """Run the circle packing constructor for n=26"""
    centers, radii, sum_radii = construct_packing()
    return centers, radii, sum_radii


def visualize(centers, radii):
    """
    Visualize the circle packing

    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates
        radii: np.array of shape (n) with radius of each circle
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(True)
    for i, (center, radius) in enumerate(zip(centers, radii)):
        circle = Circle(center, radius, alpha=0.5)
        ax.add_patch(circle)
        ax.text(center[0], center[1], str(i), ha="center", va="center")
    plt.title(f"Circle Packing (n={len(centers)}, sum={sum(radii):.6f})")
    plt.show()


if __name__ == "__main__":
    centers, radii, sum_radii = run_packing()
    print(f"Sum of radii: {sum_radii}")
    # visualize(centers, radii)
