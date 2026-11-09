# EVOLVE-BLOCK-START
"""Hex/ring/two-scale greedy multi-start SLSQP, LP radius recovery, Metropolis
basin hopping over a pool (rattler reinsertion + small-jitter moves), and a
small-jitter final polish (n=26)."""
import time
import numpy as np
from scipy.optimize import minimize, linprog

N = 26
_IU, _JU = np.triu_indices(N, 1)
_P = len(_IU)
_ROWS = np.arange(_P)
_K = np.arange(N)
_NVAR = 3 * N
_BUDGET = 16.0
_POOL = 6
_T0 = 0.004


def _walls(c):
    return np.minimum(np.minimum(c[:, 0], c[:, 1]), np.minimum(1 - c[:, 0], 1 - c[:, 1]))


def _repair(c, r_hint=None):
    """Return (centers, radii) guaranteed valid: inside the square, no overlaps."""
    c = np.clip(np.asarray(c, dtype=float), 0.0, 1.0)
    r = np.maximum(_walls(c), 0.0)
    if r_hint is not None:
        r = np.minimum(r, np.maximum(np.asarray(r_hint, dtype=float), 0.0))
    for i in range(N):
        for j in range(i + 1, N):
            d = np.hypot(c[i, 0] - c[j, 0], c[i, 1] - c[j, 1])
            s = r[i] + r[j]
            if s > d and s > 0.0:
                f = d / s * (1.0 - 1e-12)
                r[i] *= f
                r[j] *= f
    return c, np.maximum(r, 0.0)


def _lp_radii(c):
    """Exact max-sum radii for fixed centers (LP)."""
    c = np.clip(c, 0.0, 1.0)
    d = np.linalg.norm(c[_IU] - c[_JU], axis=1)
    A = np.zeros((_P, N))
    A[_ROWS, _IU] = 1.0
    A[_ROWS, _JU] = 1.0
    bounds = [(0.0, max(float(w), 0.0)) for w in _walls(c)]
    try:
        res = linprog(-np.ones(N), A_ub=A, b_ub=d, bounds=bounds, method="highs")
    except Exception:
        return None
    if res.status != 0 or res.x is None or not np.all(np.isfinite(res.x)):
        return None
    return np.maximum(res.x, 0.0)


def _finalize(c, r_hint):
    """Best valid packing for centers c: repaired SLSQP radii or repaired LP radii."""
    cands = [_repair(c, r_hint)]
    r_lp = _lp_radii(c)
    if r_lp is not None:
        cands.append(_repair(c, r_lp))
    return max(cands, key=lambda t: float(np.sum(t[1])))


def _obj(z):
    return -np.sum(z[2 * N:])


def _obj_grad(z):
    g = np.zeros(_NVAR)
    g[2 * N:] = -1.0
    return g


def _optimize(c0, r0=None, maxiter=400, ftol=1e-11):
    """SLSQP over centers and radii with analytic Jacobian, then repair + LP."""
    c0 = np.clip(np.asarray(c0, dtype=float), 0.0, 1.0)
    if r0 is None:
        _, r0 = _repair(c0)
    r0 = np.maximum(np.asarray(r0, dtype=float), 1e-3)
    z0 = np.concatenate([c0[:, 0], c0[:, 1], np.minimum(r0, 0.5)])

    def cons(z):
        x, y, r = z[:N], z[N:2 * N], z[2 * N:]
        pair = (x[_IU] - x[_JU]) ** 2 + (y[_IU] - y[_JU]) ** 2 - (r[_IU] + r[_JU]) ** 2
        wall = np.concatenate([x - r, 1 - x - r, y - r, 1 - y - r])
        return np.concatenate([pair, wall])

    def cons_jac(z):
        x, y, r = z[:N], z[N:2 * N], z[2 * N:]
        dx = x[_IU] - x[_JU]
        dy = y[_IU] - y[_JU]
        s = r[_IU] + r[_JU]
        J = np.zeros((_P + 4 * N, _NVAR))
        J[_ROWS, _IU] = 2 * dx
        J[_ROWS, _JU] = -2 * dx
        J[_ROWS, N + _IU] = 2 * dy
        J[_ROWS, N + _JU] = -2 * dy
        J[_ROWS, 2 * N + _IU] = -2 * s
        J[_ROWS, 2 * N + _JU] = -2 * s
        b = _P
        J[b + _K, _K] = 1.0
        J[b + _K, 2 * N + _K] = -1.0
        J[b + N + _K, _K] = -1.0
        J[b + N + _K, 2 * N + _K] = -1.0
        J[b + 2 * N + _K, N + _K] = 1.0
        J[b + 2 * N + _K, 2 * N + _K] = -1.0
        J[b + 3 * N + _K, N + _K] = -1.0
        J[b + 3 * N + _K, 2 * N + _K] = -1.0
        return J

    bounds = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
    try:
        res = minimize(
            _obj, z0, jac=_obj_grad, method="SLSQP", bounds=bounds,
            constraints=[{"type": "ineq", "fun": cons, "jac": cons_jac}],
            options={"maxiter": maxiter, "ftol": ftol},
        )
        z = res.x if np.all(np.isfinite(res.x)) else z0
    except Exception:
        z = z0
    c = np.column_stack([z[:N], z[N:2 * N]])
    return _finalize(c, z[2 * N:])


def _hex_rows(counts, rng, jitter):
    """Staggered rows (hex-like) with alternating horizontal offsets."""
    R = len(counts)
    pts = []
    for k, cnt in enumerate(counts):
        y = (k + 0.5) / R
        shift = 0.5 / cnt if k % 2 else 0.0
        for m in range(cnt):
            x = (m + 0.5) / cnt * (1.0 - 0.5 / cnt) + shift
            pts.append([min(max(x, 0.05), 0.95), y])
    P = np.array(pts) + rng.normal(0.0, jitter, (N, 2))
    if rng.random() < 0.5:
        P = P[:, ::-1]
    return np.clip(P, 0.03, 0.97)


def _graded_radii(c):
    """Graded start radii: wall clearance, shrunk for interior circles."""
    w = _walls(c)
    interior = 1.0 - np.minimum(np.abs(c[:, 0] - 0.5), np.abs(c[:, 1] - 0.5)) * 2.0
    return np.maximum(np.minimum(w, 0.12 + 0.05 * (1.0 - interior)), 1e-3)


def _ring_start(rng):
    """Center, inner ring of 8, outer ring of 17 (1 + 8 + 17 = 26)."""
    c = np.zeros((N, 2))
    c[0] = [0.5, 0.5]
    a0 = rng.uniform(0, 0.3)
    for i in range(8):
        a = 2 * np.pi * i / 8 + a0
        c[1 + i] = [0.5 + 0.2 * np.cos(a), 0.5 + 0.2 * np.sin(a)]
    for i in range(17):
        a = 2 * np.pi * i / 17
        c[9 + i] = [0.5 + 0.42 * np.cos(a), 0.5 + 0.42 * np.sin(a)]
    return np.clip(c + rng.normal(0, 0.005, c.shape), 0.03, 0.97)


def _clear(Q, c, r):
    """Clearance of sample points Q from walls and from circles (c, r)."""
    d = np.linalg.norm(Q[:, None, :] - c[None, :, :], axis=2) - r[None, :]
    return np.minimum(_walls(Q), d.min(axis=1))


def _hole(c, r, rng, m=3000):
    """Coarse random sampling followed by local hill-climbing refinement."""
    Q = rng.uniform(0.0, 1.0, (m, 2))
    clr = _clear(Q, c, r)
    top = np.argsort(clr)[-5:]
    best_q, best_v = Q[top[-1]].copy(), float(clr[top[-1]])
    for t in top:
        q = Q[t].copy()
        v = float(_clear(q[None, :], c, r)[0])
        step = 0.02
        for _ in range(40):
            cand = np.clip(q + rng.normal(0.0, step, (8, 2)), 0.0, 1.0)
            vals = _clear(cand, c, r)
            b = int(np.argmax(vals))
            if vals[b] > v:
                q, v = cand[b], float(vals[b])
            else:
                step *= 0.8
        if v > best_v:
            best_q, best_v = q, v
    return best_q, max(best_v, 1e-3)


def _greedy(rng, cap_big=0.21, cap_small=0.11, m=3000):
    """Largest-first greedy with a two-scale radius cap."""
    c = np.zeros((N, 2))
    r = np.zeros(N)
    for k in range(N):
        Q = rng.uniform(0.0, 1.0, (m, 2))
        if k == 0:
            clr = _walls(Q)
        else:
            clr = _clear(Q, c[:k], r[:k])
        cap = cap_big if k < 5 else cap_small
        i = int(np.argmax(clr))
        c[k] = Q[i]
        r[k] = max(min(float(clr[i]), cap), 0.0)
    return c, r


def _add(pool, c, r, K=_POOL):
    """Keep the top K distinct basins (by sum of radii)."""
    s = float(np.sum(r))
    for ps, _, _ in pool:
        if abs(ps - s) < 1e-7:
            return
    pool.append((s, c.copy(), r.copy()))
    pool.sort(key=lambda t: -t[0])
    del pool[K:]


def _contacts(c, r, tol=1e-6):
    """Active pair/wall contacts per circle (low count means rattler-like)."""
    d = np.linalg.norm(c[_IU] - c[_JU], axis=1)
    touch = (d - (r[_IU] + r[_JU])) < tol
    cnt = np.bincount(_IU[touch], minlength=N) + np.bincount(_JU[touch], minlength=N)
    w = np.stack([c[:, 0] - r, 1 - c[:, 0] - r, c[:, 1] - r, 1 - c[:, 1] - r], axis=1)
    return cnt + (w < tol).sum(axis=1)


def _perturb(c, r, rng):
    """Rattler reinsertion at refined holes, or a small-jitter move."""
    c = c.copy()
    r = r.copy()
    if rng.random() < 0.6:
        cnt = _contacts(c, r)
        order = np.lexsort((r, cnt))
        k = int(rng.integers(1, 3))
        idx = rng.choice(order[:6], size=k, replace=False)
        for i in idx:
            keep = _K != i
            p, clr = _hole(c[keep], r[keep], rng)
            c[i] = p
            r[i] = clr
    else:
        sigma = float(rng.choice([0.002, 0.005, 0.02]))
        c = c + rng.normal(0.0, sigma, c.shape)
    return np.clip(c, 0.0, 1.0), r


def construct_packing():
    t0 = time.perf_counter()
    rng = np.random.default_rng(777)
    pool = []

    # Phase 1: diverse starts (hex rows, ring, two-scale greedy, random).
    starts = []
    for pat in [[5, 6, 5, 6, 4], [4, 6, 5, 6, 5], [5, 5, 6, 5, 5], [6, 5, 6, 5, 4]]:
        c0 = _hex_rows(pat, rng, 0.015)
        starts.append((c0, _graded_radii(c0)))
    starts.append((_ring_start(rng), None))
    for _ in range(3):
        starts.append(_greedy(rng))
    for _ in range(2):
        starts.append((rng.uniform(0.1, 0.9, (N, 2)), None))

    for c0, r0 in starts:
        if pool and time.perf_counter() - t0 > 0.4 * _BUDGET:
            break
        c, r = _optimize(c0, r0, maxiter=500, ftol=1e-11)
        _add(pool, c, r)

    if not pool:
        c, r = _repair(_ring_start(rng))
        _add(pool, c, r)

    # Phase 2: Metropolis basin hopping. Current state drifts; pool keeps the best.
    cur_s, cur_c, cur_r = pool[0][0], pool[0][1].copy(), pool[0][2].copy()
    t_hop = 0.4 * _BUDGET
    while True:
        elapsed = time.perf_counter() - t0
        if elapsed > _BUDGET - 2.5:
            break
        frac = min(1.0, (elapsed - t_hop) / max(_BUDGET - 2.5 - t_hop, 1e-6))
        T = max(1e-5, _T0 * (1.0 - frac))

        if rng.random() < 0.2:
            _, base_c, base_r = pool[int(rng.integers(len(pool)))]
        else:
            base_c, base_r = cur_c, cur_r
        c, r = _perturb(base_c, base_r, rng)
        cn, rn = _optimize(c, r, maxiter=400, ftol=1e-11)
        s = float(np.sum(rn))
        if s > cur_s or rng.random() < np.exp((s - cur_s) / T):
            cur_s, cur_c, cur_r = s, cn, rn
        _add(pool, cn, rn)

    # Phase 3: tight polish plus small-jitter hops around the best basin.
    for _, cb, rb in list(pool[:3]):
        if time.perf_counter() - t0 > _BUDGET + 1.5:
            break
        cn, rn = _optimize(cb, rb, maxiter=1000, ftol=1e-14)
        _add(pool, cn, rn)
    while time.perf_counter() - t0 < _BUDGET + 1.5:
        _, cb, rb = pool[0]
        c = np.clip(cb + rng.normal(0.0, 0.002, cb.shape), 0.0, 1.0)
        cn, rn = _optimize(c, rb, maxiter=300, ftol=1e-13)
        _add(pool, cn, rn)

    best_c, best_r = _repair(pool[0][1], pool[0][2])
    return best_c, best_r, float(np.sum(best_r))


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_packing():
    """Run the circle packing constructor for n=26"""
    centers, radii, sum_radii = construct_packing()
    return centers, radii, sum_radii


if __name__ == "__main__":
    centers, radii, sum_radii = run_packing()
    print(f"Sum of radii: {sum_radii}")
