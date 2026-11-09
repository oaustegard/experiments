# EVOLVE-BLOCK-START
"""Multi-start (hex-row + mixed-radius ring seeds, incl. 90-degree rotations),
iterated joint SLSQP + exact LP radii, then a wall-clock-limited basin hopping
that mixes Gaussian perturbations with a targeted escape (smallest circle moved
into the largest empty hole), and a final feasibility-checked LP for n=26."""
import time

import numpy as np
from scipy.optimize import linprog, minimize

N = 26
_I, _J = np.triu_indices(N, 1)
_HOP_BUDGET_S = 35.0
_MAX_HOPS = 400
_REFINE_ROUNDS = 3


def _boundary_limits(c):
    """Largest radius each center allows before touching a wall."""
    return np.minimum.reduce([c[:, 0], c[:, 1], 1 - c[:, 0], 1 - c[:, 1]])


def _greedy_radii(c, bnd, passes=100):
    """Fallback: pairwise proportional scaling (always valid)."""
    r = bnd.copy()
    for _ in range(passes):
        changed = False
        for i, j in zip(_I, _J):
            d = np.hypot(*(c[i] - c[j]))
            s = r[i] + r[j]
            if s > d + 1e-15:
                f = d / s if s > 0 else 0.0
                r[i] *= f
                r[j] *= f
                changed = True
        if not changed:
            break
    return np.maximum(r, 0.0)


def _lp_radii(c, margin=1e-10):
    """Exact maximization of sum(r) for fixed centers (tightened pair constraint)."""
    c = np.clip(c, 0.0, 1.0)
    bnd = np.maximum(_boundary_limits(c), 0.0)
    d = np.hypot(*(c[_I] - c[_J]).T)
    m = len(_I)
    A = np.zeros((m, N))
    A[np.arange(m), _I] = 1.0
    A[np.arange(m), _J] = 1.0
    try:
        res = linprog(-np.ones(N), A_ub=A, b_ub=np.maximum(d - 1e-9, 0.0),
                      bounds=[(0.0, b) for b in bnd], method="highs")
        if res.success:
            r = np.maximum(res.x, 0.0)
        else:
            r = _greedy_radii(c, bnd)
    except Exception:
        r = _greedy_radii(c, bnd)
    return r * (1.0 - margin)


def _is_valid(c, r, tol=1e-9):
    if np.any(r < -tol):
        return False
    if np.any(c - r[:, None] < -tol) or np.any(c + r[:, None] > 1 + tol):
        return False
    d = np.hypot(*(c[_I] - c[_J]).T)
    return bool(np.all(r[_I] + r[_J] <= d + tol))


def _refine(c0, r0, maxiter=600):
    """Joint SLSQP over centers and radii, maximizing sum of radii."""
    z0 = np.concatenate([c0.ravel(), r0])

    def obj(z):
        return -np.sum(z[2 * N:])

    def obj_grad(z):
        g = np.zeros_like(z)
        g[2 * N:] = -1.0
        return g

    def cons(z):
        c = z[:2 * N].reshape(N, 2)
        r = z[2 * N:]
        diff = c[_I] - c[_J]
        pair = np.sum(diff ** 2, axis=1) - (r[_I] + r[_J]) ** 2
        bnd = np.concatenate([
            c[:, 0] - r, 1 - c[:, 0] - r,
            c[:, 1] - r, 1 - c[:, 1] - r,
        ])
        return np.concatenate([pair, bnd])

    bounds = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
    try:
        res = minimize(obj, z0, jac=obj_grad, method="SLSQP", bounds=bounds,
                       constraints=[{"type": "ineq", "fun": cons}],
                       options={"maxiter": maxiter, "ftol": 1e-12})
        return np.clip(res.x[:2 * N].reshape(N, 2), 0.0, 1.0)
    except Exception:
        return np.clip(c0, 0.0, 1.0)


def _refine_iterated(c, rounds=_REFINE_ROUNDS):
    """Repeat SLSQP (re-seeded with exact LP radii) until the sum stops improving."""
    best_c = np.clip(c, 0.0, 1.0)
    best_s = float(np.sum(_lp_radii(best_c)))
    cur = best_c
    for _ in range(rounds):
        cand = _refine(cur, _lp_radii(cur))
        s = float(np.sum(_lp_radii(cand)))
        if s > best_s + 1e-10:
            best_s, best_c, cur = s, cand, cand
        else:
            break
    return best_c


def _rows_layout(counts):
    """Staggered hex-style row seed; counts must sum to N."""
    assert sum(counts) == N
    ys = np.linspace(0.1, 0.9, len(counts))
    pts = []
    for k, m in enumerate(counts):
        if m == 1:
            xs = np.array([0.5])
        else:
            half = 0.4 / (m - 1)
            lo, hi = 0.1, 0.9
            if k % 2 == 1:
                lo, hi = lo + half * 0.5, hi - half * 0.5
            xs = np.linspace(lo, hi, m)
        for x in xs:
            pts.append([x, ys[k]])
    return np.clip(np.array(pts, dtype=float), 0.02, 0.98)


def _ring_seed(k=12, inset=0.07):
    """Mixed-radius seed: k boundary circles evenly spaced along the square
    perimeter, plus a 5-4-5 staggered interior hex block."""
    lo, hi = inset, 1.0 - inset
    L = hi - lo
    pts = []
    for t in np.arange(k) / k * 4 * L:
        side, u = int(t // L), t - int(t // L) * L
        if side == 0:
            pts.append((lo + u, lo))
        elif side == 1:
            pts.append((hi, lo + u))
        elif side == 2:
            pts.append((hi - u, hi))
        else:
            pts.append((lo, hi - u))
    for y, xs in [(0.27, [0.25, 0.41, 0.59, 0.75, 0.33]),
                  (0.50, [0.33, 0.50, 0.67, 0.42]),
                  (0.73, [0.25, 0.41, 0.59, 0.75, 0.33])]:
        for x in xs:
            pts.append((x, y))
    c = np.array(pts, dtype=float)
    return np.clip(c, 0.02, 0.98)


def _polish(c):
    """Iterated SLSQP refinement followed by exact LP radii; returns (c, r, sum) or None."""
    c = np.clip(c, 0.0, 1.0)
    candidates = [c, _refine_iterated(c)]
    best = None
    for cc in candidates:
        r = _lp_radii(cc)
        if _is_valid(cc, r):
            s = float(np.sum(r))
            if best is None or s > best[2]:
                best = (cc.copy(), r.copy(), s)
    return best


def _largest_hole(c, r, rng, n_samp=4000):
    """Approximate max-empty-circle: sample points, maximize clearance to
    circles and walls."""
    pts = rng.uniform(0.0, 1.0, size=(n_samp, 2))
    d = np.linalg.norm(pts[:, None, :] - c[None, :, :], axis=2) - r[None, :]
    clear = np.min(d, axis=1)
    wall = np.min(np.minimum(pts, 1.0 - pts), axis=1)
    score = np.minimum(clear, wall)
    return pts[int(np.argmax(score))]


def run_packing():
    """Return (centers, radii, sum_radii) for 26 circles in the unit square."""
    rng = np.random.default_rng(12345)
    t_start = time.perf_counter()

    count_specs = [
        [5, 6, 5, 5, 5],
        [6, 5, 5, 6, 4],
        [5, 5, 6, 5, 5],
        [4, 5, 4, 5, 4, 4],
        [5, 4, 5, 4, 4, 4],
        [4, 4, 4, 7, 7],
        [6, 5, 6, 5, 4],
    ]
    seeds = [_ring_seed(), _ring_seed(k=12, inset=0.09)]
    for counts in count_specs:
        base = _rows_layout(counts)
        seeds.append(base)
        seeds.append(np.clip(base[:, ::-1].copy(), 0.02, 0.98))

    best = None
    for c0 in seeds:
        res = _polish(c0)
        if res is not None and (best is None or res[2] > best[2]):
            best = res

    if best is None:
        c0 = np.clip(_rows_layout([5, 6, 5, 5, 5]), 0.0, 1.0)
        r0 = _lp_radii(c0)
        best = (c0, r0, float(np.sum(r0)))

    # Wall-clock-limited basin hopping: Gaussian kicks plus targeted escape
    # (smallest-radius circle moved into the largest empty hole).
    for it in range(_MAX_HOPS):
        if time.perf_counter() - t_start > _HOP_BUDGET_S:
            break
        frac = it / max(_MAX_HOPS - 1, 1)
        scale = 0.04 * (1.0 - frac) + 0.003
        c = best[0].copy()
        move = rng.random()
        if move < 0.4:
            c += rng.normal(0.0, scale, size=c.shape)
        elif move < 0.7:
            # Perturb a random band of circles more strongly (row-like move).
            band = rng.uniform(0.0, 1.0)
            mask = np.abs(c[:, 1] - band) < 0.15
            c[mask] += rng.normal(0.0, scale * 1.5, size=(int(mask.sum()), 2))
        else:
            # Escape: move the smallest-radius circle into the largest hole.
            k = int(np.argmin(best[1]))
            c[k] = _largest_hole(best[0], best[1], rng)
            c += rng.normal(0.0, scale * 0.2, size=c.shape)
        c = np.clip(c, 0.02, 0.98)
        res = _polish(c)
        if res is not None and res[2] > best[2] + 1e-9:
            best = res

    # Final exact polish from the best centers with a tight SLSQP run.
    final = _polish(_refine(best[0], best[1], maxiter=800))
    if final is not None and final[2] > best[2] + 1e-9:
        best = final

    # Final feasibility repair: LP with tightened pair margin, then fallbacks.
    best_c = best[0]
    best_r = _lp_radii(best_c)
    if not _is_valid(best_c, best_r):
        bnd = np.maximum(_boundary_limits(best_c), 0.0)
        best_r = _greedy_radii(best_c, bnd)
        best_r = best_r * (1.0 - 1e-9)
    best_s = float(np.sum(best_r))
    return best_c, best_r, best_s


# EVOLVE-BLOCK-END


if __name__ == "__main__":
    centers, radii, sum_radii = run_packing()
    print(f"Sum of radii: {sum_radii}")
