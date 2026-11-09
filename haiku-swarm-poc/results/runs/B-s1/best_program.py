# EVOLVE-BLOCK-START
"""LP-screened structured seeds (staggered rows, ring + hex, row/ring hybrid, mixed, random),
SLSQP polish of the best distinct seeds, LP-gated threshold basin hopping with a hashed-basin
cache and stall restarts, then a tight-tolerance SLSQP polish of the top elite packings
finished with exact LP radii (no proportional shrinking) and an explicit validity check. n=26."""
import time
import numpy as np
from scipy.optimize import minimize, linprog, linear_sum_assignment

N = 26
_I, _J = np.triu_indices(N, 1)
_P = len(_I)
_M = 4 * N + _P
_BUDGET = 22.0
_CHAINS = 3
_ELITE = 4
_POLISH = 8
_FINAL_TOP = 3
_T0 = 0.002
_STALL_LIMIT = 30
_GATE_EARLY = 0.12
_GATE_LATE = 0.01
_SQ3 = np.sqrt(3.0)
_COUNT_SETS = [[5, 6, 5, 6, 4], [6, 5, 6, 5, 4], [4, 6, 5, 6, 5], [5, 5, 6, 5, 5],
               [5, 5, 5, 5, 6], [6, 5, 5, 5, 5], [4, 5, 6, 6, 5]]
_RING = [(0.08, 0.08), (0.92, 0.08), (0.08, 0.92), (0.92, 0.92),
         (0.5, 0.05), (0.5, 0.95), (0.05, 0.5), (0.95, 0.5)]


def _wall_dist(c):
    return np.minimum(np.minimum(c[:, 0], c[:, 1]),
                      np.minimum(1.0 - c[:, 0], 1.0 - c[:, 1]))


def _repair(centers, radii):
    """Guarantee validity: clip, cap by walls, shrink any overlapping pair (fallback only)."""
    c = np.clip(np.asarray(centers, dtype=float), 0.0, 1.0)
    r = np.maximum(np.asarray(radii, dtype=float), 0.0)
    r = np.minimum(r, _wall_dist(c))
    r = np.maximum(r, 0.0)
    for i in range(N):
        for j in range(i + 1, N):
            d = np.hypot(c[i, 0] - c[j, 0], c[i, 1] - c[j, 1])
            s = r[i] + r[j]
            if s > d and s > 0.0:
                f = d / s
                r[i] *= f
                r[j] *= f
    r = r * (1.0 - 1e-9)
    return c, r


def _is_valid(c, r, tol=1e-12):
    if not (np.all(np.isfinite(c)) and np.all(np.isfinite(r))):
        return False
    if np.any(r < -tol) or np.any(c < -tol) or np.any(c > 1.0 + tol):
        return False
    if np.any(c - r[:, None] < -tol) or np.any(c + r[:, None] > 1.0 + tol):
        return False
    d = np.hypot(c[_I, 0] - c[_J, 0], c[_I, 1] - c[_J, 1])
    return bool(np.all(d - (r[_I] + r[_J]) >= -tol))


def _lp_radii(c):
    """Optimal radii for fixed centers: max sum r s.t. wall and pair constraints."""
    wall = _wall_dist(c)
    A = np.zeros((_P, N))
    A[np.arange(_P), _I] = 1.0
    A[np.arange(_P), _J] = 1.0
    d = np.hypot(c[_I, 0] - c[_J, 0], c[_I, 1] - c[_J, 1])
    b = d * (1.0 - 1e-9)
    bounds = [(0.0, max(w * (1.0 - 1e-9), 0.0)) for w in wall]
    try:
        res = linprog(-np.ones(N), A_ub=A, b_ub=b, bounds=bounds, method="highs")
        if res.success and np.all(np.isfinite(res.x)):
            return np.maximum(res.x, 0.0)
    except Exception:
        pass
    return None


def _obj(z):
    return -np.sum(z[2 * N:])


def _obj_grad(z):
    g = np.zeros_like(z)
    g[2 * N:] = -1.0
    return g


def _cons(z):
    x, y, r = z[:N], z[N:2 * N], z[2 * N:]
    walls = np.concatenate([x - r, 1 - x - r, y - r, 1 - y - r])
    dx = x[_I] - x[_J]
    dy = y[_I] - y[_J]
    s = r[_I] + r[_J]
    return np.concatenate([walls, dx * dx + dy * dy - s * s])


def _cons_jac(z):
    x, y, r = z[:N], z[N:2 * N], z[2 * N:]
    Jm = np.zeros((_M, 3 * N))
    idx = np.arange(N)
    Jm[idx, idx] = 1.0
    Jm[idx, 2 * N + idx] = -1.0
    Jm[N + idx, idx] = -1.0
    Jm[N + idx, 2 * N + idx] = -1.0
    Jm[2 * N + idx, N + idx] = 1.0
    Jm[2 * N + idx, 2 * N + idx] = -1.0
    Jm[3 * N + idx, N + idx] = -1.0
    Jm[3 * N + idx, 2 * N + idx] = -1.0
    dx = x[_I] - x[_J]
    dy = y[_I] - y[_J]
    s = r[_I] + r[_J]
    p = np.arange(_P) + 4 * N
    Jm[p, _I] = 2 * dx
    Jm[p, _J] = -2 * dx
    Jm[p, N + _I] = 2 * dy
    Jm[p, N + _J] = -2 * dy
    Jm[p, 2 * N + _I] = -2 * s
    Jm[p, 2 * N + _J] = -2 * s
    return Jm


def _optimize(centers, radii, maxiter=300, ftol=1e-12):
    """Joint SLSQP over centers and radii."""
    z0 = np.concatenate([centers[:, 0], centers[:, 1], radii])
    bounds = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
    try:
        res = minimize(_obj, z0, jac=_obj_grad, method="SLSQP", bounds=bounds,
                       constraints=[{"type": "ineq", "fun": _cons, "jac": _cons_jac}],
                       options={"maxiter": maxiter, "ftol": ftol})
        z = res.x if np.all(np.isfinite(res.x)) else z0
    except Exception:
        z = z0
    return np.stack([z[:N], z[N:2 * N]], axis=1), z[2 * N:]


def _finish(centers, radii):
    """Repair, then take exact LP radii for these centers if they are no worse."""
    c, r = _repair(centers, radii)
    rl = _lp_radii(c)
    if rl is not None:
        c2, r2 = _repair(c, rl)
        if np.sum(r2) >= np.sum(r):
            return c2, r2
    return c, r


def _solve(centers, radii):
    """repair -> SLSQP -> LP finish. Always returns a valid packing."""
    c0, r0 = _repair(centers, radii)
    c1, r1 = _optimize(c0, r0)
    return _finish(c1, r1)


def _lp_or(pts, fallback):
    r0 = _lp_radii(pts)
    return fallback if r0 is None else r0


# ---------------------------------------------------------------- seeds

def _rows(counts, margin, transpose):
    """Staggered rows; odd rows are inset by `margin`."""
    ys = np.linspace(0.1, 0.9, len(counts))
    pts = []
    for k, (c, y) in enumerate(zip(counts, ys)):
        lo, hi = (0.08, 0.92) if k % 2 == 0 else (0.08 + margin, 0.92 - margin)
        for x in np.linspace(lo, hi, c):
            pts.append([x, y])
    pts = np.clip(np.array(pts, dtype=float), 0.02, 0.98)
    return pts[:, ::-1].copy() if transpose else pts


def _ring_hex(rng):
    """Corner + edge-midpoint ring of larger circles; interior from a hex lattice."""
    s = float(rng.uniform(0.19, 0.23))
    hp = []
    for row in range(-6, 7):
        y = 0.5 + row * s * _SQ3 / 2.0
        off = 0.0 if row % 2 == 0 else s / 2.0
        for col in range(-6, 7):
            hp.append((0.5 + col * s + off, y))
    hp = np.array(hp, dtype=float)
    d = np.hypot(hp[:, 0] - 0.5, hp[:, 1] - 0.5) + rng.normal(0.0, 0.01, len(hp))
    inner = hp[np.argsort(d)][:N - len(_RING)]
    ring = np.array(_RING, dtype=float) + rng.normal(0.0, 0.01, (len(_RING), 2))
    pts = np.clip(np.vstack([inner, ring]) + rng.normal(0.0, 0.004, (N, 2)), 0.03, 0.97)
    return pts, _lp_or(pts, np.full(N, 0.05))


def _hybrid(rng):
    """Average staggered-row centres with ring-hex centres after optimal matching."""
    a = _rows(_COUNT_SETS[int(rng.integers(0, len(_COUNT_SETS)))], 0.04, False)
    b, _ = _ring_hex(rng)
    cost = np.hypot(a[:, None, 0] - b[None, :, 0], a[:, None, 1] - b[None, :, 1])
    _, ci = linear_sum_assignment(cost)
    mid = 0.5 * (a + b[ci]) + rng.normal(0.0, 0.005, (N, 2))
    mid = np.clip(mid, 0.03, 0.97)
    return mid, _lp_or(mid, np.full(N, 0.05))


def _mixed(rng):
    """Large boundary circles, medium circles staggered in the interior."""
    boundary = [(0.08, 0.08), (0.92, 0.08), (0.08, 0.92), (0.92, 0.92),
                (0.5, 0.06), (0.5, 0.94), (0.06, 0.5), (0.94, 0.5),
                (0.27, 0.06), (0.73, 0.06), (0.27, 0.94), (0.73, 0.94)]
    rad = [0.11] * len(boundary)
    for k, (y, nx) in enumerate(zip([0.3, 0.47, 0.64, 0.8], [4, 4, 3, 3])):
        xs = np.linspace(0.2, 0.8, nx) if k % 2 == 0 else np.linspace(0.25, 0.75, nx)
        for x in xs:
            boundary.append((x, y))
            rad.append(0.07)
    pts = np.clip(np.array(boundary, dtype=float) + rng.normal(0.0, 0.01, (N, 2)), 0.03, 0.97)
    return pts, _lp_or(pts, np.array(rad, dtype=float))


def _random_seed(rng):
    pts = rng.uniform(0.05, 0.95, (N, 2))
    return pts, _lp_or(pts, np.full(N, 0.05))


def _seed_pool(rng):
    """LP-screen all seeds; return list of (lp_total, centres, radii) sorted descending."""
    cands = []

    def add(pts, r0):
        if r0 is not None:
            r0 = np.asarray(r0, dtype=float)
            cands.append((float(np.sum(r0)), pts, r0))

    for cs in _COUNT_SETS:
        for margin in (0.0, 0.04, 0.08):
            for tr in (False, True):
                pts = _rows(cs, margin, tr)
                add(pts, _lp_radii(pts))
    for _ in range(6):
        add(*_ring_hex(rng))
    for _ in range(4):
        add(*_hybrid(rng))
    for seed in (_mixed(rng), _random_seed(rng)):
        add(seed[0], seed[1])
    cands.sort(key=lambda t: -t[0])
    return cands


# ---------------------------------------------------------------- moves

def _hole_point(c, r, keep, rng, n_samp=2000):
    """Point with the largest clearance to kept circles and walls, plus that clearance."""
    pts = rng.uniform(0.02, 0.98, (n_samp, 2))
    ck, rk = c[keep], r[keep]
    d = np.sqrt(((pts[:, None, :] - ck[None, :, :]) ** 2).sum(-1)) - rk[None, :]
    cl = np.minimum(d.min(axis=1), _wall_dist(pts))
    k = int(np.argmax(cl))
    return pts[k], float(max(cl[k], 0.0))


def _step(c, r, rng, frac):
    """Basin-hopping move: jiggle, relocate 1-2 small circles into holes, or swap."""
    cc = c.copy()
    cr = r.copy()
    order = np.argsort(cr)
    u = rng.random()
    if u < 0.35:
        sigma = 0.002 + 0.012 * (1.0 - frac)
        cc = np.clip(cc + rng.normal(0.0, sigma, cc.shape), 0.02, 0.98)
    elif u < 0.85:
        k = 1 if rng.random() < 0.7 else 2
        picks = rng.choice(order[:8], size=k, replace=False)
        keep = np.ones(N, dtype=bool)
        keep[picks] = False
        for m in picks:
            p, clr = _hole_point(cc, cr, keep, rng)
            cc[m] = p
            cr[m] = max(0.5 * clr, 0.01)
            keep[m] = True
    else:
        i = order[rng.integers(0, 6)]
        j = order[-1 - rng.integers(0, 6)]
        cc[[i, j]] = cc[[j, i]]
    return cc, cr


def _key(c):
    return tuple(np.round(c, 4).ravel().tolist())


def _update_elite(elite, tot, c, r):
    if any(abs(tot - e[0]) < 1e-6 for e in elite):
        return
    elite.append((tot, c, r))
    elite.sort(key=lambda t: -t[0])
    del elite[_ELITE:]


def _polish_tight(c, r):
    """Tight SLSQP, then exact LP radii on the resulting centres (validity-checked)."""
    best_c, best_r = _finish(c, r)
    try:
        c3, _ = _optimize(c, r, maxiter=500, ftol=1e-15)
        rl = _lp_radii(c3)
        if rl is not None and _is_valid(c3, rl) and np.sum(rl) > np.sum(best_r):
            best_c, best_r = c3, rl
        c4, r4 = _finish(c3, np.zeros(N) if rl is None else rl)
        if np.sum(r4) > np.sum(best_r):
            best_c, best_r = c4, r4
    except Exception:
        pass
    return best_c, best_r


def construct_packing():
    t0 = time.perf_counter()
    deadline = t0 + _BUDGET
    rng = np.random.default_rng(2024)

    cands = _seed_pool(rng)
    seen = set()

    # Phase 2: SLSQP polish of the top LP-ranked seeds into the elite set.
    elite = []
    for _, pts, r0 in cands[:_POLISH]:
        if elite and time.perf_counter() > t0 + 0.35 * _BUDGET:
            break
        c, r = _solve(pts, r0)
        seen.add(_key(c))
        _update_elite(elite, float(np.sum(r)), c, r)
    next_seed = min(_POLISH, len(cands))
    if not elite:
        c, r = _solve(*_random_seed(rng))
        _update_elite(elite, float(np.sum(r)), c, r)

    chains = []
    for tot, c, r in elite[:_CHAINS]:
        chains.append({"c": c, "r": r, "tot": tot, "btot": tot, "stall": 0})
    best_tot, best_c, best_r = elite[0]

    # Phase 3: LP-gated threshold basin hopping with stall restarts.
    it = 0
    while time.perf_counter() < deadline - 2.0:
        frac = min(1.0, (time.perf_counter() - t0) / _BUDGET)
        thr = _T0 * (1.0 - frac)
        gate = _GATE_EARLY if frac < 0.5 else _GATE_LATE
        ch = chains[it % len(chains)]
        it += 1

        if ch["stall"] > _STALL_LIMIT:
            if next_seed < len(cands):
                _, pts, r0 = cands[next_seed]
                next_seed += 1
            else:
                pts, r0 = _ring_hex(rng)
            c2, r2 = _solve(pts, r0)
            tot2 = float(np.sum(r2))
            seen.add(_key(c2))
            ch.update({"c": c2, "r": r2, "tot": tot2, "btot": tot2, "stall": 0})
            _update_elite(elite, tot2, c2, r2)
            if tot2 > best_tot + 1e-9:
                best_c, best_r, best_tot = c2, r2, tot2
            continue

        cand_c, _ = _step(ch["c"], ch["r"], rng, frac)
        quick = _lp_radii(cand_c)
        if quick is None or float(np.sum(quick)) < ch["tot"] - gate:
            ch["stall"] += 1
            continue

        k = _key(cand_c)
        if k in seen:
            ch["stall"] += 1
            continue
        seen.add(k)

        c2, r2 = _solve(cand_c, quick)
        tot2 = float(np.sum(r2))

        if tot2 > ch["tot"] - thr:
            ch["c"], ch["r"], ch["tot"] = c2, r2, tot2
        if tot2 > ch["btot"] + 1e-9:
            ch["btot"] = tot2
            ch["stall"] = 0
        else:
            ch["stall"] += 1
        if tot2 > best_tot + 1e-9:
            best_c, best_r, best_tot = c2, r2, tot2
        _update_elite(elite, tot2, c2, r2)

    # Phase 4: tight polish of the best packing and top elites, finished with LP radii.
    finals = [(best_tot, best_c, best_r)] + [(t, c, r) for t, c, r in elite[:_FINAL_TOP]]
    best_final = None
    for _, c, r in finals:
        c4, r4 = _polish_tight(c, r)
        tot4 = float(np.sum(r4))
        if best_final is None or tot4 > best_final[0]:
            best_final = (tot4, c4, r4)

    c, r = best_final[1], best_final[2]
    if not _is_valid(c, r):
        c, r = _repair(c, r)
    else:
        r = r * (1.0 - 1e-12)
    return c, r, float(np.sum(r))


# EVOLVE-BLOCK-END


def run_packing():
    """Run the circle packing constructor for n=26"""
    centers, radii, sum_radii = construct_packing()
    return centers, radii, sum_radii


def visualize(centers, radii):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(True)
    for i, (center, radius) in enumerate(zip(centers, radii)):
        ax.add_patch(Circle(center, radius, alpha=0.5))
        ax.text(center[0], center[1], str(i), ha="center", va="center")
    plt.title(f"Circle Packing (n={len(centers)}, sum={sum(radii):.6f})")
    plt.show()


if __name__ == "__main__":
    centers, radii, sum_radii = run_packing()
    print(f"Sum of radii: {sum_radii}")
    visualize(centers, radii)
