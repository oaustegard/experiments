# EVOLVE-BLOCK-START
"""Multistart joint SLSQP (centres + radii) from staggered-row, boundary-ring
plus hex-core, two-scale and random seeds, followed by an elite-population
basin hop (ranked hole relocation, deterministic smallest-circle relocation,
swap, shake). Every candidate is re-optimised jointly and its radii are set by
an exact LP, so the returned packing is always valid. The top elite basins are
each polished with a tight final SLSQP and the best valid result is kept."""
import math
import time

import numpy as np
from scipy.optimize import linprog, minimize
from scipy.spatial import Voronoi

N = 26
BH_ITERS = 300
TIME_SAFETY = 150.0
T0 = 0.004
SEED = 2024
ELITE = 4
POLISH_TOP = 3
JITTER_PER_ELITE = 2
RESTART_EVERY = 100
RANDOM_SEEDS = 4
_I, _J = np.triu_indices(N, 1)
_M = len(_I)
_A = np.arange(N)


def row_seed(counts):
    """Staggered rows with cell-centred x positions so adjacent rows interleave."""
    R = len(counts)
    ys = np.linspace(0.1, 0.9, R)
    pts = []
    for k, m in enumerate(counts):
        off = 0.5 if k % 2 else 0.0
        xs = 0.1 + 0.8 * (np.arange(m) + 0.5 + off * 0.5) / m
        for x in xs:
            pts.append([min(max(x, 0.05), 0.95), ys[k]])
    return np.clip(np.array(pts, dtype=float), 0.02, 0.98)


def ring_hex_seed(rng):
    """12 boundary circles (large corners, edge-centred, edge-offset) plus a
    compressed hex-like core of 14 circles in the interior."""
    ring = [[0.08, 0.08], [0.92, 0.08], [0.08, 0.92], [0.92, 0.92],
            [0.5, 0.06], [0.5, 0.94], [0.06, 0.5], [0.94, 0.5],
            [0.25, 0.06], [0.75, 0.94], [0.06, 0.75], [0.94, 0.25]]
    core = []
    for y, m in zip((0.30, 0.45, 0.60, 0.75), (3, 4, 3, 4)):
        off = 0.0 if m == 4 else 0.0
        xs = np.linspace(0.28, 0.72, m) if m == 4 else np.linspace(0.3, 0.7, m)
        for x in xs:
            core.append([x + off, y])
    P = np.array(ring + core, dtype=float)[:N]
    if len(P) < N:
        P = np.vstack([P, rng.uniform(0.2, 0.8, (N - len(P), 2))])
    return np.clip(P + rng.normal(0.0, 0.01, P.shape), 0.04, 0.96)


def two_scale_seed(rng):
    """Large circles hugging edges and corners, small fillers inside."""
    pts = [[0.08, 0.08], [0.92, 0.08], [0.08, 0.92], [0.92, 0.92]]
    for t in (0.3, 0.5, 0.7):
        pts += [[t, 0.08], [t, 0.92], [0.08, t], [0.92, t]]
    while len(pts) < N:
        pts.append(rng.uniform(0.2, 0.8, 2))
    P = np.array(pts[:N], dtype=float)
    return np.clip(P + rng.normal(0.0, 0.01, P.shape), 0.05, 0.95)


def random_seed(rng):
    """Random topology: a few corner-anchored large circles plus random fillers."""
    pts = []
    corners = [[0.07, 0.07], [0.93, 0.07], [0.07, 0.93], [0.93, 0.93]]
    k = int(rng.integers(2, 5))
    for c in corners[:k]:
        pts.append(c)
    while len(pts) < N:
        pts.append(rng.uniform(0.08, 0.92, 2))
    P = np.array(pts[:N], dtype=float)
    return np.clip(P + rng.normal(0.0, 0.02, P.shape), 0.03, 0.97)


def is_valid(C, r, tol=1e-12):
    if np.any(r < 0):
        return False
    if np.any(C - r[:, None] < -tol) or np.any(C + r[:, None] > 1 + tol):
        return False
    d = np.linalg.norm(C[_I] - C[_J], axis=1)
    return bool(np.all(d >= r[_I] + r[_J] - tol))


def lp_radii(C):
    """Exact LP: maximise sum r for fixed centres (wall and pair constraints)."""
    n = C.shape[0]
    wall = np.minimum(np.minimum(C[:, 0], C[:, 1]),
                      np.minimum(1 - C[:, 0], 1 - C[:, 1]))
    I, J = np.triu_indices(n, 1)
    d = np.linalg.norm(C[I] - C[J], axis=1)
    A = np.zeros((len(I), n))
    A[np.arange(len(I)), I] = 1.0
    A[np.arange(len(I)), J] = 1.0
    bounds = [(0.0, max(0.0, float(w))) for w in wall]
    res = linprog(-np.ones(n), A_ub=A, b_ub=d, bounds=bounds, method="highs")
    if not res.success:
        return None
    return np.maximum(res.x, 0.0)


def safe_radii(C):
    """LP radii shrunk by the smallest epsilon that makes the packing valid."""
    r = lp_radii(C)
    if r is None:
        return None
    for eps in (0.0, 1e-12, 1e-10, 1e-9, 1e-8, 1e-7):
        rr = np.maximum(r - eps, 0.0)
        if is_valid(C, rr):
            return rr
    return None


def fallback_radii(C):
    """Order-dependent pairwise scaling; always yields a valid packing."""
    n = C.shape[0]
    r = np.array([min(x, y, 1 - x, 1 - y) for x, y in C], dtype=float)
    r = np.maximum(r, 0.0)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(C[i] - C[j])
            if r[i] + r[j] > d and r[i] + r[j] > 0:
                s = d / (r[i] + r[j])
                r[i] *= s
                r[j] *= s
    return r


def _cons(z):
    x, y, r = z[:N], z[N:2 * N], z[2 * N:]
    dx = x[_I] - x[_J]
    dy = y[_I] - y[_J]
    s = r[_I] + r[_J]
    gp = dx * dx + dy * dy - s * s
    gw = np.concatenate([x - r, 1 - x - r, y - r, 1 - y - r])
    return np.concatenate([gp, gw])


def _cjac(z):
    x, y, r = z[:N], z[N:2 * N], z[2 * N:]
    dx = x[_I] - x[_J]
    dy = y[_I] - y[_J]
    s = r[_I] + r[_J]
    Jm = np.zeros((_M + 4 * N, 3 * N))
    rows = np.arange(_M)
    Jm[rows, _I] = 2 * dx
    Jm[rows, _J] = -2 * dx
    Jm[rows, N + _I] = 2 * dy
    Jm[rows, N + _J] = -2 * dy
    Jm[rows, 2 * N + _I] = -2 * s
    Jm[rows, 2 * N + _J] = -2 * s
    b = _M
    Jm[b + _A, _A] = 1.0
    Jm[b + _A, 2 * N + _A] = -1.0
    Jm[b + N + _A, _A] = -1.0
    Jm[b + N + _A, 2 * N + _A] = -1.0
    Jm[b + 2 * N + _A, N + _A] = 1.0
    Jm[b + 2 * N + _A, 2 * N + _A] = -1.0
    Jm[b + 3 * N + _A, N + _A] = -1.0
    Jm[b + 3 * N + _A, 2 * N + _A] = -1.0
    return Jm


def _obj(z):
    return -np.sum(z[2 * N:])


def _obj_grad(z):
    g = np.zeros_like(z)
    g[2 * N:] = -1.0
    return g


def joint_opt(C0, r0, maxiter=400, ftol=1e-12):
    """SLSQP over z = [x, y, r] with analytic Jacobians; returns centres."""
    z0 = np.concatenate([C0[:, 0], C0[:, 1], r0])
    bounds = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
    try:
        res = minimize(_obj, z0, jac=_obj_grad, method="SLSQP", bounds=bounds,
                       constraints=[{"type": "ineq", "fun": _cons, "jac": _cjac}],
                       options={"maxiter": maxiter, "ftol": ftol})
        z = res.x
    except Exception:
        z = z0
    return np.column_stack([np.clip(z[:N], 0, 1), np.clip(z[N:2 * N], 0, 1)])


def hole_candidates(C, radii, skip):
    """Candidate hole centres scored by min(clearance to circles, clearance to walls)."""
    others = np.delete(C, skip, axis=0)
    orad = np.delete(radii, skip)
    mirrors = np.vstack([
        others * [-1.0, 1.0],
        others * [1.0, -1.0],
        np.column_stack([2.0 - others[:, 0], others[:, 1]]),
        np.column_stack([others[:, 0], 2.0 - others[:, 1]]),
    ])
    cands = [others]
    try:
        vor = Voronoi(np.vstack([others, mirrors]))
        cands.append(vor.vertices)
    except Exception:
        pass
    cands.append(np.column_stack([others[:, 0], np.zeros(len(others))]))
    cands.append(np.column_stack([others[:, 0], np.ones(len(others))]))
    cands.append(np.column_stack([np.zeros(len(others)), others[:, 1]]))
    cands.append(np.column_stack([np.ones(len(others)), others[:, 1]]))
    g = np.linspace(0.02, 0.98, 30)
    gx, gy = np.meshgrid(g, g)
    cands.append(np.column_stack([gx.ravel(), gy.ravel()]))
    P = np.vstack(cands)
    P = P[np.all((P >= 0.0) & (P <= 1.0), axis=1)]
    if len(P) == 0:
        return np.array([[0.5, 0.5]]), np.array([0.0])
    dist = np.linalg.norm(P[:, None, :] - others[None, :, :], axis=2) - orad[None, :]
    dmin = np.min(dist, axis=1)
    wall = np.minimum(np.minimum(P[:, 0], P[:, 1]), np.minimum(1 - P[:, 0], 1 - P[:, 1]))
    score = np.minimum(dmin, wall)
    order = np.argsort(-score)
    return P[order], score[order]


def pick_hole(C, radii, skip, rng, topk=3):
    """Random choice among the top-k hole candidates."""
    P, _ = hole_candidates(C, radii, skip)
    k = min(topk, len(P))
    return P[int(rng.integers(0, k))]


def local_refine(C, r0):
    """Joint SLSQP from a centre set, then exact radii; returns (C, r, s) or None."""
    C_new = joint_opt(C, r0)
    r_new = safe_radii(C_new)
    if r_new is None or not is_valid(C_new, r_new):
        return None
    return C_new, r_new, float(np.sum(r_new))


def update_elite(elite, C, r, s):
    """Keep the top-ELITE distinct basins (distinct by radius sum)."""
    for e in elite:
        if abs(e[0] - s) < 1e-7:
            return
    elite.append((s, C.copy(), r.copy()))
    elite.sort(key=lambda e: -e[0])
    del elite[ELITE:]


def config_key(C):
    """Rounded configuration signature for tabu bookkeeping."""
    return tuple(np.round(C, 2).ravel().tolist())


def basin_hop(C, r, rng, iters, deadline, elite):
    """Basin hopping with ranked relocate, deterministic smallest-circle relocate,
    swap and shake moves; tabu keys come from move content or rounded configs."""
    cur_C, cur_r, cur_s = C, r, float(np.sum(r))
    best_C, best_r, best_s = C, r, cur_s
    tabu = set()
    for it in range(iters):
        if time.time() > deadline:
            break
        temp = T0 * (1.0 - it / iters) + 1e-6
        if it > 0 and it % RESTART_EVERY == 0 and len(elite) > 1:
            e = elite[(it // RESTART_EVERY) % len(elite)]
            cur_s, cur_C, cur_r = e[0], e[1].copy(), e[2].copy()

        cand = cur_C.copy()
        mode = it % 4
        if mode == 0:
            order = np.argsort(cur_r)
            k = int(order[rng.integers(0, 3)])
            hole = pick_hole(cur_C, cur_r, k, rng, topk=3)
            key = ("hole", k, round(float(hole[0]), 2), round(float(hole[1]), 2))
            cand[k] = hole
        elif mode == 1:
            k = int(np.argmin(cur_r))
            P, _ = hole_candidates(cur_C, cur_r, k)
            idx = (it // 4) % min(5, len(P))
            hole = P[idx]
            key = ("minhole", k, round(float(hole[0]), 3), round(float(hole[1]), 3))
            cand[k] = hole
        elif mode == 2:
            order = np.argsort(cur_r)
            k = int(order[rng.integers(0, 3)])
            j = int(order[-1 - rng.integers(0, 3)])
            key = ("swap", min(k, j), max(k, j))
            cand[[k, j]] = cand[[j, k]]
        else:
            cand = cand + rng.normal(0.0, 0.015 * temp / T0 + 0.003, cand.shape)
            key = ("shake", config_key(np.clip(cand, 0.0, 1.0)))
        if key in tabu:
            continue
        cand = np.clip(cand, 0.0, 1.0)
        r0 = lp_radii(cand)
        if r0 is None:
            r0 = np.full(N, 0.02)
        out = local_refine(cand, r0)
        if out is None:
            tabu.add(key)
            continue
        C_new, r_new, s = out
        delta = s - cur_s
        if delta > 1e-9 or rng.random() < math.exp(min(0.0, delta) / temp):
            cur_C, cur_r, cur_s = C_new, r_new, s
        else:
            tabu.add(key)
        if s > best_s + 1e-9 and is_valid(C_new, r_new):
            best_C, best_r, best_s = C_new, r_new, s
        update_elite(elite, C_new, r_new, s)
    return best_C, best_r, best_s


def repair_pass(C):
    """Re-run the LP on fixed centres with a small safety margin; keep the valid option."""
    cands = []
    r_lp = lp_radii(C)
    if r_lp is not None:
        for m in (0.0, 1e-12, 1e-9):
            rr = np.maximum(r_lp - m, 0.0)
            if is_valid(C, rr):
                cands.append(rr)
                break
    if not cands:
        cands.append(fallback_radii(C))
    best = max(cands, key=lambda rr: float(np.sum(rr)))
    return C, best


def construct_packing():
    t0 = time.time()
    rng = np.random.default_rng(SEED)

    starts = [row_seed(c) for c in ([5, 6, 5, 6, 4], [4, 5, 6, 5, 6], [5, 5, 5, 5, 6])]
    starts.append(ring_hex_seed(rng))
    starts.append(two_scale_seed(rng))
    for _ in range(RANDOM_SEEDS):
        starts.append(random_seed(rng))

    best = {"s": -1.0, "C": None, "r": None}
    elite = []

    def consider(C, r):
        if C is None or r is None or not is_valid(C, r):
            return
        s = float(np.sum(r))
        if s > best["s"]:
            best.update(s=s, C=C.copy(), r=r.copy())

    for C0 in starts:
        r0 = lp_radii(C0)
        if r0 is None:
            r0 = np.full(N, 0.05)
        out = local_refine(joint_opt(C0, r0), r0)
        if out is not None:
            consider(out[0], out[1])
            update_elite(elite, out[0], out[1], out[2])
        r_seed = safe_radii(C0)
        if r_seed is not None:
            consider(C0, r_seed)

    if best["C"] is None:
        C = row_seed([5, 6, 5, 6, 4])
        r = fallback_radii(C)
        return C, r, float(np.sum(r))

    # Jitter restarts from each elite basin.
    for e in list(elite):
        for _ in range(JITTER_PER_ELITE):
            Cj = np.clip(e[1] + rng.normal(0.0, 0.02, e[1].shape), 0.0, 1.0)
            r0 = lp_radii(Cj)
            if r0 is None:
                continue
            out = local_refine(Cj, r0)
            if out is not None:
                consider(out[0], out[1])
                update_elite(elite, out[0], out[1], out[2])

    deadline = t0 + TIME_SAFETY
    C, r, s = basin_hop(best["C"], best["r"], rng, BH_ITERS, deadline, elite)
    consider(C, r)

    # Final polish on the top elite basins (not just the best), each with a
    # tight SLSQP and exact LP radii; keep the best valid result overall.
    for e in elite[:POLISH_TOP]:
        Cp = joint_opt(e[1], e[2], maxiter=800, ftol=1e-15)
        consider(Cp, safe_radii(Cp))
        Cr, rr = repair_pass(e[1])
        consider(Cr, rr)

    Cp = joint_opt(best["C"], best["r"], maxiter=800, ftol=1e-15)
    consider(Cp, safe_radii(Cp))

    C, r, s = best["C"], best["r"], best["s"]
    if not is_valid(C, r):
        r = fallback_radii(C)
        s = float(np.sum(r))
    return C, r, s


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_packing():
    """Run the circle packing constructor for n=26"""
    centers, radii, sum_radii = construct_packing()
    centers = np.asarray(centers, dtype=float)
    radii = np.asarray(radii, dtype=float)
    return centers, radii, float(np.sum(radii))


if __name__ == "__main__":
    centers, radii, sum_radii = run_packing()
    print(f"Sum of radii: {sum_radii}")
