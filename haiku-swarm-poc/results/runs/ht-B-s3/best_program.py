# EVOLVE-BLOCK-START
import itertools
import time

import numpy as np
from scipy.optimize import minimize

N = 11
SQRT3 = np.sqrt(3.0)
TRI = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, SQRT3 / 2.0]])
CEN = TRI.mean(axis=0)

TRIPLES = np.array(list(itertools.combinations(range(N), 3)), dtype=int)
TI, TJ, TK = TRIPLES[:, 0], TRIPLES[:, 1], TRIPLES[:, 2]
M = len(TRIPLES)
ROWS = np.arange(M)

# Half-plane constraints a*x + b*y + c >= 0 describing the triangle.
HP = np.array([
    [0.0, 1.0, 0.0],
    [SQRT3, -1.0, 0.0],
    [-SQRT3, -1.0, SQRT3],
])
# Linear form for all points: HPF @ flat(P) + HPC >= 0
HPF = np.zeros((3 * N, 2 * N))
HPC = np.zeros(3 * N)
for _i in range(N):
    for _l, (_a, _b, _c) in enumerate(HP):
        HPF[3 * _i + _l, 2 * _i] = _a
        HPF[3 * _i + _l, 2 * _i + 1] = _b
        HPC[3 * _i + _l] = _c

# Mirror symmetry about x = 0.5 with 5 pairs and one axis point.
K_PAIRS = 5
NR = 2 * K_PAIRS + 1
T_SYM = np.zeros((2 * N, NR))
C_SYM = np.zeros(2 * N)
for _p in range(K_PAIRS):
    _a, _b = 2 * _p, 2 * _p + 1
    T_SYM[2 * _a, 2 * _p] = 1.0
    T_SYM[2 * _a + 1, 2 * _p + 1] = 1.0
    T_SYM[2 * _b, 2 * _p] = -1.0
    C_SYM[2 * _b] = 1.0
    T_SYM[2 * _b + 1, 2 * _p + 1] = 1.0
_c = 2 * K_PAIRS
T_SYM[2 * _c + 1, NR - 1] = 1.0
C_SYM[2 * _c] = 0.5

EYE_FULL = np.eye(2 * N)
ZERO_FULL = np.zeros(2 * N)

NSYM = 48          # symmetric multistart count
TOPSYM = 4         # symmetric layouts released to free refinement
NFREE = 16         # free starts (half uniform, half boundary-biased)
HOPS = 200         # basin-hopping budget
RESEED_EVERY = 40  # inject a fresh symmetric solution every this many hops
HOP_TIME_CAP = 60.0  # wall-clock guard (seconds) for the hopping loop only
POOL = 10
TEMP_REL = 0.03
WINDOW = 1.5       # epigraph uses triples within 1.5x the current minimum during hops
SEED = 20240611
POLISH_REL = 0.05
POLISH_ROUNDS = 3
POLISH_TOP = 3     # pool members receiving the final polish
SYM_JITTER_COPIES = 2


def _signed_areas(P):
    x, y = P[:, 0], P[:, 1]
    return 0.5 * ((x[TJ] - x[TI]) * (y[TK] - y[TI]) - (x[TK] - x[TI]) * (y[TJ] - y[TI]))


def _min_area(P):
    a = np.abs(_signed_areas(P))
    if not np.all(np.isfinite(a)):
        return -1.0
    return float(np.min(a))


def _project(P):
    """Push points into the triangle using the half-plane constraints."""
    P = np.array(P, dtype=float, copy=True).reshape(N, 2)
    P[:, 0] = np.clip(P[:, 0], 0.0, 1.0)
    P[:, 1] = np.clip(P[:, 1], 0.0, SQRT3 / 2.0)
    for _ in range(100):
        changed = False
        for i in range(N):
            x, y = P[i]
            for a, b, c in HP:
                val = a * x + b * y + c
                if val < 0.0:
                    nrm = a * a + b * b
                    x -= val * a / nrm
                    y -= val * b / nrm
                    changed = True
            P[i] = (x, y)
        if not changed:
            break
    return P


def _sample(rng):
    """Uniform random points in the triangle."""
    u = rng.random(N)
    v = rng.random(N)
    m = u + v > 1.0
    u[m], v[m] = 1.0 - u[m], 1.0 - v[m]
    return TRI[0] + u[:, None] * (TRI[1] - TRI[0]) + v[:, None] * (TRI[2] - TRI[0])


def _sample_boundary_biased(rng):
    """Uniform start with 3-5 points forced onto the triangle's edges or vertices."""
    P = _sample(rng)
    k = int(rng.integers(3, 6))
    idx = rng.choice(N, size=k, replace=False)
    e = rng.integers(3, size=k)
    t = rng.random(k)
    t[rng.random(k) < 0.2] = 0.0  # some points land exactly on vertices
    A = TRI[e]
    B = TRI[(e + 1) % 3]
    P[idx] = A + t[:, None] * (B - A)
    return _project(P)


def _sym_reduce(P):
    q = np.zeros(NR)
    for p in range(K_PAIRS):
        a, b = 2 * p, 2 * p + 1
        q[2 * p] = 0.5 * (P[a, 0] + 1.0 - P[b, 0])
        q[2 * p + 1] = 0.5 * (P[a, 1] + P[b, 1])
    q[NR - 1] = P[2 * K_PAIRS, 1]
    return q


def _sym_expand(q):
    return (T_SYM @ q + C_SYM).reshape(N, 2)


def _area_jac(P, s):
    """Jacobian of s * signed_area over all triples, w.r.t. flat coordinates."""
    x, y = P[:, 0], P[:, 1]
    xi, yi = x[TI], y[TI]
    xj, yj = x[TJ], y[TJ]
    xk, yk = x[TK], y[TK]
    J = np.zeros((M, 2 * N))
    J[ROWS, 2 * TI] = 0.5 * s * (yj - yk)
    J[ROWS, 2 * TI + 1] = 0.5 * s * (xk - xj)
    J[ROWS, 2 * TJ] = 0.5 * s * (yk - yi)
    J[ROWS, 2 * TJ + 1] = 0.5 * s * (xi - xk)
    J[ROWS, 2 * TK] = 0.5 * s * (yi - yj)
    J[ROWS, 2 * TK + 1] = 0.5 * s * (xj - xi)
    return J


def _solve(sg, T, C, q0, idx=None, ftol=1e-13, maxiter=300):
    """Maximize t s.t. sg * signed_area >= t (over triples idx) and points inside.
    Variables: flat = T q + C, plus epigraph variable t."""
    NQ = T.shape[1]
    sel = np.arange(M) if idx is None else np.asarray(idx, dtype=int)
    nA = len(sel)

    def full(q):
        return (T @ q + C).reshape(N, 2)

    def cons(w):
        q, t = w[:-1], w[-1]
        P = full(q)
        area = sg[sel] * _signed_areas(P)[sel] - t
        hp = HPF @ (T @ q + C) + HPC
        return np.concatenate([area, hp])

    def cjac(w):
        q = w[:-1]
        P = full(q)
        Ja = _area_jac(P, sg)[sel]
        Jf = np.vstack([Ja, HPF])
        J = np.zeros((nA + 3 * N, NQ + 1))
        J[:, :NQ] = Jf @ T
        J[:nA, -1] = -1.0
        return J

    def obj(w):
        return -w[-1]

    def gobj(w):
        g = np.zeros(NQ + 1)
        g[-1] = -1.0
        return g

    q0 = np.asarray(q0, dtype=float)
    z0 = np.concatenate([q0, [_min_area(full(q0))]])
    res = minimize(obj, z0, jac=gobj, method="SLSQP",
                   constraints=[{"type": "ineq", "fun": cons, "jac": cjac}],
                   options={"maxiter": maxiter, "ftol": ftol})
    return full(res.x[:-1])


def _refine(P0, sym=False, window=None):
    """Alternate SLSQP epigraph solves with orientation-sign re-derivation.
    With window set (free mode), the epigraph uses only near-minimal triples.
    Acceptance always uses the true minimum over all triples."""
    P = _project(P0)
    if sym:
        P = _project(_sym_expand(_sym_reduce(P)))
    best, best_a = P, _min_area(P)
    for _ in range(4):
        sg = np.where(_signed_areas(best) >= 0.0, 1.0, -1.0)
        idx = None
        if window is not None and not sym:
            a = np.abs(_signed_areas(best))
            idx = np.where(a <= a.min() * window)[0]
        try:
            if sym:
                Q = _solve(sg, T_SYM, C_SYM, _sym_reduce(best))
            else:
                Q = _solve(sg, EYE_FULL, ZERO_FULL, best.ravel(), idx=idx)
            Q = _project(Q)
        except Exception:
            break
        a = _min_area(Q)
        if np.isfinite(a) and a > best_a + 1e-12:
            best, best_a = Q, a
        else:
            break
    return best, best_a


def _polish_once(P):
    """One active-set SLSQP pass; accepted only if the true minimum improves."""
    best, best_a = P, _min_area(P)
    sg = np.where(_signed_areas(P) >= 0.0, 1.0, -1.0)
    a = np.abs(_signed_areas(P))
    windows = [
        np.where(a <= a.min() * (1.0 + POLISH_REL))[0],
        None,  # full constraint set as a fallback
    ]
    for idx in windows:
        try:
            Q = _solve(sg, EYE_FULL, ZERO_FULL, P.ravel(), idx=idx,
                       ftol=1e-16, maxiter=500)
            Q = _project(Q)
        except Exception:
            continue
        b = _min_area(Q)
        if np.isfinite(b) and b > best_a + 1e-14:
            best, best_a = Q, b
    return best, best_a


def _polish(P):
    """Repeat the active-set polish while the true minimum keeps improving."""
    best, best_a = P, _min_area(P)
    for _ in range(POLISH_ROUNDS):
        Q, b = _polish_once(best)
        if b > best_a + 1e-14:
            best, best_a = Q, b
        else:
            break
    return best, best_a


def _bottleneck_mask(P, rel=1.02):
    a = np.abs(_signed_areas(P))
    idx = np.where(a <= a.min() * rel)[0]
    mask = np.zeros(N, dtype=bool)
    mask[TI[idx]] = True
    mask[TJ[idx]] = True
    mask[TK[idx]] = True
    return mask


def _add(pool, v, P, cap=POOL):
    """Insert (v, P) unless a solution with the same value already exists."""
    for u, _ in pool:
        if abs(u - v) < 1e-9:
            return pool
    pool.append((v, P))
    pool.sort(key=lambda x: -x[0])
    del pool[cap:]
    return pool


def _symmetric_seed(rng):
    """Fresh mirror-symmetric SLSQP solution, or None on failure."""
    try:
        P, a = _refine(_sym_expand(_sym_reduce(_sample(rng))), sym=True)
    except Exception:
        return None
    return P, a


def _sym_images(P):
    """The six symmetries of the equilateral triangle, applied about its centroid."""
    out = []
    c, s = np.cos(2.0 * np.pi / 3.0), np.sin(2.0 * np.pi / 3.0)
    R = np.array([[c, -s], [s, c]])
    F = np.array([[-1.0, 0.0], [0.0, 1.0]])  # reflection about x = 0.5
    Q = P - CEN
    for G in (np.eye(2), F):
        A = G.copy()
        for _ in range(3):
            out.append(_project(CEN + Q @ A.T))
            A = R @ A
    return out


def _search():
    rng = np.random.default_rng(SEED)
    t_start = time.monotonic()

    # Phase 1: mirror-symmetric multistart in the 11-variable subspace.
    sym_results = []
    for _ in range(NSYM):
        r = _symmetric_seed(rng)
        if r is not None:
            sym_results.append((r[1], r[0]))
    sym_results.sort(key=lambda x: -x[0])

    # Phase 2: release symmetry and refine the best symmetric layouts.
    pool = []
    for a, P in sym_results[:TOPSYM]:
        try:
            Q, b = _refine(P, sym=False, window=WINDOW)
        except Exception:
            continue
        pool = _add(pool, b, Q)

    # Phase 3: free starts, half uniform and half boundary-biased.
    for k in range(NFREE):
        start = _sample(rng) if k % 2 == 0 else _sample_boundary_biased(rng)
        try:
            Q, b = _refine(start, sym=False, window=WINDOW)
        except Exception:
            continue
        pool = _add(pool, b, Q)

    if not pool:
        return None

    # Phase 4: annealed basin hopping with periodic symmetric reseeding.
    # The wall-clock guard only stops the loop early; it never changes the
    # per-hop computation, so results are reproducible whenever it is not hit.
    cur_v, cur = pool[0]
    best_v, best = pool[0]
    temp_base = best_v
    frac_local = 0.0
    for h in range(HOPS):
        if time.monotonic() - t_start > HOP_TIME_CAP:
            break
        frac = h / HOPS
        if h > 0 and h % RESEED_EVERY == 0:
            r = _symmetric_seed(rng)
            if r is not None:
                try:
                    Q, v = _refine(r[0], sym=False, window=WINDOW)
                except Exception:
                    Q, v = None, -1.0
                if Q is not None and np.isfinite(v):
                    pool = _add(pool, v, Q)
                    if v > cur_v:
                        cur_v, cur = v, Q
                    if v > best_v + 1e-12:
                        best_v, best = v, Q
            temp_base = cur_v
            frac_local = 0.0
        else:
            frac_local = (h % RESEED_EVERY) / RESEED_EVERY
        temp = TEMP_REL * (1.0 - frac_local) * temp_base + 1e-12

        if rng.random() < 0.7:
            base = cur
        else:
            base = pool[int(rng.integers(len(pool)))][1]
        scale = 0.05 * (1.0 - frac) + 0.004
        if rng.random() < 0.5:
            mask = _bottleneck_mask(base)
            noise = (rng.normal(0.0, 2.0 * scale, (N, 2)) * mask[:, None]
                     + rng.normal(0.0, 0.1 * scale, (N, 2)) * (~mask)[:, None])
        else:
            noise = rng.normal(0.0, scale, (N, 2))
        Q0 = base + noise
        if rng.random() < 0.15:
            Q0[int(rng.integers(N))] = _sample(rng)[0]
        try:
            Q, v = _refine(Q0, sym=False, window=WINDOW)
        except Exception:
            continue
        if not np.isfinite(v):
            continue
        pool = _add(pool, v, Q)
        if v >= cur_v or rng.random() < np.exp((v - cur_v) / temp):
            cur_v, cur = v, Q
        if v > best_v + 1e-12:
            best_v, best = v, Q
    return best, pool


def heilbronn_triangle11() -> np.ndarray:
    """
    Construct 11 points on or inside the equilateral triangle (0,0), (1,0),
    (0.5, sqrt(3)/2) maximizing the minimum triangle area. Deterministic:
    mirror-symmetric multistart, boundary-biased free starts, windowed SLSQP
    refinement, annealed basin hopping (wall-clock guarded), then full-epigraph
    polishing of the incumbent, top pool members, and their triangle-symmetry images.
    """
    try:
        res = _search()
    except Exception:
        res = None
    if res is None:
        P, pool = None, []
    else:
        P, pool = res

    if P is None or not np.all(np.isfinite(P)):
        P = _sample(np.random.default_rng(SEED))
        pool = []
    out = _project(P)

    # Final polish: incumbent, top pool members, and their six symmetry images
    # (plus a tiny deterministic jitter so SLSQP takes different paths).
    seeds = [out] + [Q for _, Q in pool[:POLISH_TOP]]
    jrng = np.random.default_rng(SEED + 7)
    candidates = []
    for S in seeds:
        for T in _sym_images(_project(S)):
            candidates.append(T)
            for _ in range(SYM_JITTER_COPIES):
                candidates.append(_project(T + jrng.normal(0.0, 1e-4, (N, 2))))

    best_out, best_val = out, _min_area(out)
    for C in candidates:
        try:
            Q, qa = _polish(_project(C))
            Q = _project(Q)
            if np.all(np.isfinite(Q)) and qa > best_val:
                best_out, best_val = Q, qa
        except Exception:
            continue
    out = _project(best_out)
    return np.asarray(out, dtype=float).reshape(N, 2)


# EVOLVE-BLOCK-END
