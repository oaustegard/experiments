# EVOLVE-BLOCK-START
import math
import time
import numpy as np
from scipy.optimize import minimize

_S32 = math.sqrt(3.0) / 2.0          # apothem factor of a unit hexagon / outer hexagon
_MARGIN = 1e-7                        # clearance used inside annealing penalty
_VMARG = 1e-6                         # clearance required of the final packing
_PMARG = 3e-6                         # clearance enforced inside SLSQP polish
_N = 11
_NORM = [
    (math.cos(math.pi / 6 + k * math.pi / 3), math.sin(math.pi / 6 + k * math.pi / 3))
    for k in range(6)
]
_NX = np.array([p[0] for p in _NORM], dtype=float)
_NY = np.array([p[1] for p in _NORM], dtype=float)
_K6 = np.arange(6, dtype=float) * math.pi / 3


def _verts(x, y, a):
    return [
        (x + math.cos(a + k * math.pi / 3), y + math.sin(a + k * math.pi / 3))
        for k in range(6)
    ]


def _contain(V):
    # Minimal outer side R (outer hexagon centred at origin, angle 0) containing vertices V.
    best = -1e18
    for vx, vy in V:
        for nx, ny in _NORM:
            d = nx * vx + ny * vy
            if d > best:
                best = d
    return best / _S32


def _side(P):
    return max(_contain(_verts(*p)) for p in P)


def _sep_gap(Vp, ap, Vq, aq, need):
    # Best separating-axis gap over the 6 edge normals (positive means disjoint).
    best = -1e18
    for base in (ap, aq):
        for k in range(3):
            t = base + math.pi / 6 + k * math.pi / 3
            ux, uy = math.cos(t), math.sin(t)
            pp = [vx * ux + vy * uy for vx, vy in Vp]
            qq = [vx * ux + vy * uy for vx, vy in Vq]
            gap = max(min(qq) - max(pp), min(pp) - max(qq))
            if gap > best:
                best = gap
                if best >= need:
                    return best
    return best


def _pair_pen(Vp, ap, Vq, aq):
    g = _sep_gap(Vp, ap, Vq, aq, _MARGIN)
    return 0.0 if g >= _MARGIN else _MARGIN - g


def _evaluate(P):
    n = len(P)
    V = [_verts(*p) for p in P]
    cont = [_contain(v) for v in V]
    pen = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            pv = _pair_pen(V[i], P[i][2], V[j], P[j][2])
            pen[i][j] = pv
            pen[j][i] = pv
    return V, cont, pen


def _sumsq(pen, n):
    s = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            s += pen[i][j] * pen[i][j]
    return s


def _valid(P, margin=_VMARG):
    # Exact final check: every pair separated by at least `margin`.
    n = len(P)
    V = [_verts(*p) for p in P]
    for i in range(n):
        for j in range(i + 1, n):
            if _sep_gap(V[i], P[i][2], V[j], P[j][2], margin) < margin:
                return False
    return True


def _anneal(P0, seed, tlimit, s0=0.3, T0=0.02):
    # Simulated annealing on (x, y, angle) with penalty-based feasibility.
    rng = np.random.default_rng(seed)
    P = [list(map(float, p)) for p in P0]
    n = len(P)
    V, cont, pen = _evaluate(P)
    S = _sumsq(pen, n)
    R = max(cont)
    best_R = R if S == 0.0 else float("inf")
    best_P = [list(p) for p in P] if S == 0.0 else None
    cur = R + 10.0 * S
    t0 = time.time()
    while True:
        f = (time.time() - t0) / tlimit
        if f >= 1.0:
            break
        s = s0 * (0.002 / s0) ** f
        lam = 10.0 * (1000.0 / 10.0) ** f
        T = T0 * (2e-4 / T0) ** f

        i = int(rng.integers(n))
        old_p = list(P[i])
        old_v = V[i]
        old_c = cont[i]
        old_pen = list(pen[i])

        r = rng.random()
        if r < 0.45:
            P[i][0] += s * rng.normal()
            P[i][1] += s * rng.normal()
        elif r < 0.75:
            P[i][2] += 2.0 * s * rng.normal()
        else:
            c = s * rng.random()
            P[i][0] -= c * P[i][0]
            P[i][1] -= c * P[i][1]

        newV = _verts(*P[i])
        V[i] = newV
        cont[i] = _contain(newV)
        for j in range(n):
            if j != i:
                pv = _pair_pen(newV, P[i][2], V[j], P[j][2])
                pen[i][j] = pv
                pen[j][i] = pv

        newS = _sumsq(pen, n)
        newR = max(cont)
        new_cost = newR + lam * newS
        if new_cost <= cur or rng.random() < math.exp(-(new_cost - cur) / T):
            cur = new_cost
            S = newS
            R = newR
            if S == 0.0 and R < best_R:
                best_R = R
                best_P = [list(p) for p in P]
        else:
            P[i] = old_p
            V[i] = old_v
            cont[i] = old_c
            for j in range(n):
                pen[i][j] = old_pen[j]
                pen[j][i] = old_pen[j]
    return best_R, best_P


def _run_anneal(P0, seed, tlimit, s0=0.3, T0=0.02):
    try:
        return _anneal(P0, seed, tlimit, s0, T0)
    except Exception:
        return float("inf"), None


def _choose_pairs(P, dmax=4.5):
    # Every nearby pair gets a separating-axis constraint with the largest gap.
    n = len(P)
    V = [np.array(_verts(*p), dtype=float) for p in P]
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if math.hypot(P[i][0] - P[j][0], P[i][1] - P[j][1]) > dmax:
                continue
            best = None
            for base in (P[i][2], P[j][2]):
                for k in range(3):
                    tt = base + math.pi / 6 + k * math.pi / 3
                    u = np.array([math.cos(tt), math.sin(tt)])
                    pi_ = V[i] @ u
                    pj_ = V[j] @ u
                    for L, H, pl, ph in ((i, j, pi_, pj_), (j, i, pj_, pi_)):
                        g = ph.min() - pl.max()
                        if best is None or g > best[0]:
                            best = (g, L, H, u[0], u[1], 0.5 * (pl.max() + ph.min()))
            g, L, H, ux, uy, t0 = best
            pairs.append((L, H, ux, uy, t0))
    return pairs


def _jitter(P, rng, sig_xy, sig_a):
    Q = [list(map(float, p)) for p in P]
    for p in Q:
        p[0] += sig_xy * float(rng.normal())
        p[1] += sig_xy * float(rng.normal())
        p[2] += sig_a * float(rng.normal())
    return Q


def _polish(P0, rounds=10, seed=99):
    # SLSQP on (x, y, angle) for all hexagons, outer side R, and one offset per chosen axis.
    n = len(P0)
    rng = np.random.default_rng(seed)
    cur = [list(map(float, p)) for p in P0]
    best_R = _side(cur)
    best_P = cur
    fails = 0
    for _ in range(rounds):
        pairs = _choose_pairs(cur)
        Xc = np.array(cur, dtype=float)
        z0 = np.concatenate([
            Xc[:, 0], Xc[:, 1], Xc[:, 2],
            [_side(cur)],
            np.array([p[4] for p in pairs], dtype=float),
        ])

        def cons(z, pairs=pairs):
            x = z[0:n]
            y = z[n:2 * n]
            a = z[2 * n:3 * n]
            R = z[3 * n]
            t = z[3 * n + 1:]
            ang = a[:, None] + _K6[None, :]
            vx = x[:, None] + np.cos(ang)
            vy = y[:, None] + np.sin(ang)
            cvals = (R * _S32 - (vx[:, :, None] * _NX[None, None, :]
                                 + vy[:, :, None] * _NY[None, None, :])).ravel()
            parts = [cvals]
            for idx, (L, H, ux, uy, _) in enumerate(pairs):
                pl = vx[L] * ux + vy[L] * uy
                ph = vx[H] * ux + vy[H] * uy
                parts.append(t[idx] - pl - _PMARG)
                parts.append(ph - t[idx] - _PMARG)
            return np.concatenate(parts)

        def obj(z):
            return z[3 * n]

        try:
            res = minimize(
                obj, z0, method="SLSQP",
                constraints=[{"type": "ineq", "fun": cons}],
                options={"maxiter": 300, "ftol": 1e-12},
            )
            z = res.x
            Pn = np.stack([z[0:n], z[n:2 * n], z[2 * n:3 * n]], axis=1).tolist()
        except Exception:
            Pn = None

        improved = False
        if Pn is not None:
            Rn = _side(Pn)
            if Rn < best_R - 1e-11 and _valid(Pn):
                best_R = Rn
                best_P = Pn
                cur = Pn
                improved = True
        if improved:
            fails = 0
            continue
        fails += 1
        if fails >= 3:
            break
        cur = _jitter(best_P, rng, 2e-3, 2e-3)
    return best_R, best_P


def _multistart_polish(P, rng, k=2, sig=0.01):
    # Several jittered restarts of the polish; keep the best valid result.
    best_R, best_P = _side(P), P
    for _ in range(k):
        Q = _jitter(best_P, rng, sig, sig)
        try:
            R2, P2 = _polish(Q, seed=int(rng.integers(10 ** 9)))
        except Exception:
            continue
        if R2 < best_R - 1e-11 and _valid(P2):
            best_R, best_P = R2, P2
    return best_R, best_P


def _lattice_start(scale=1.003, tilt_deg=0.0):
    # Triangular-lattice seed: 11 lattice points nearest the origin, optionally tilted.
    u = (math.sqrt(3.0) * math.cos(math.pi / 6), math.sqrt(3.0) * math.sin(math.pi / 6))
    v = (0.0, math.sqrt(3.0))
    pts = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            x = i * u[0] + j * v[0]
            y = i * u[1] + j * v[1]
            pts.append((x * x + y * y, x, y))
    pts.sort()
    t = math.radians(tilt_deg)
    c, s = math.cos(t), math.sin(t)
    return [[scale * (c * x - s * y), scale * (s * x + c * y), t]
            for _, x, y in pts[:_N]]


def _rot_start(P, deg):
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    return [[c * x - s * y, s * x + c * y, a + t] for x, y, a in P]


def _mixed_start(rng, scale=1.0):
    # Lattice centres with a random subset of hexagons turned by 30 degrees.
    lat = _lattice_start(scale)
    out = []
    for x, y, _ in lat:
        a = math.pi / 6 if rng.random() < 0.5 else 0.0
        out.append([x, y, a])
    return out


def _rand_point(rng, half):
    while True:
        x, y = rng.uniform(-half, half, 2)
        if max(nx * x + ny * y for nx, ny in _NORM) <= half * _S32:
            return [float(x), float(y), float(rng.uniform(0.0, math.pi / 3))]


def _rand_start(rng, half=3.2):
    return [_rand_point(rng, half) for _ in range(_N)]


def _sym_start(rng, half=2.8):
    pts = [[0.0, 0.0, float(rng.uniform(0.0, math.pi / 3))]]
    while len(pts) < _N:
        p = _rand_point(rng, half)
        pts.append(p)
        pts.append([-p[0], -p[1], p[2]])
    return pts[:_N]


def _ring_start(rng, scale=1.02):
    # Centre hexagon with 6 same-orientation neighbours, plus 4 random outside hexagons.
    th = float(rng.uniform(0.0, math.pi / 3))
    pts = [[0.0, 0.0, th]]
    d = math.sqrt(3.0) * scale
    for k in range(6):
        ang = math.pi / 6 + k * math.pi / 3
        pts.append([d * math.cos(ang), d * math.sin(ang), th])
    while len(pts) < _N:
        pts.append(_rand_point(rng, 3.2))
    return pts


def _perturb(P, rng, sig_xy, sig_a):
    Q = _jitter(P, rng, sig_xy, sig_a)
    if rng.random() < 0.3:
        i = int(rng.integers(_N))
        Q[i] = _rand_point(rng, 2.8)
    return Q


def _crossover(PA, PB, rng):
    th = float(rng.uniform(0.0, 2.0 * math.pi))
    ux, uy = math.cos(th), math.sin(th)
    child = [list(map(float, p)) for p in PA if p[0] * ux + p[1] * uy < 0.0]
    child += [list(map(float, p)) for p in PB if p[0] * ux + p[1] * uy >= 0.0]
    order = rng.permutation(len(child))
    child = [child[int(i)] for i in order[:_N]]
    while len(child) < _N:
        child.append(_rand_point(rng, 3.0))
    return child


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon,
    maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), rows (x, y, angle_degrees).
        outer_hex_data: np.ndarray of shape (3,), (x, y, angle_degrees) of the outer hexagon.
        outer_hex_side_length: float, side length of the outer hexagon.
    """
    t_start = time.time()
    search_end = t_start + 30.0

    base = [
        [0.0, 0.0, 0.0],
        [-2.5, 0.0, 0.0],
        [2.5, 0.0, 0.0],
        [-1.25, 2.17, 0.0],
        [1.25, 2.17, 0.0],
        [-1.25, -2.17, 0.0],
        [1.25, -2.17, 0.0],
        [-3.75, 2.17, 0.0],
        [3.75, 2.17, 0.0],
        [-3.75, -2.17, 0.0],
        [3.75, -2.17, 0.0],
    ]
    lat = _lattice_start()

    # Stage 1: structured (incl. tilted lattice), mixed, ring and random starts.
    rng = np.random.default_rng(4242)
    seeds = [
        base,
        lat,
        _lattice_start(tilt_deg=7.0),
        _lattice_start(scale=1.02, tilt_deg=-8.0),
        _rot_start(base, 30.0),
        _rot_start(lat, 15.0),
        _mixed_start(rng),
        _mixed_start(rng),
        _sym_start(rng),
        _ring_start(rng),
        _rand_start(rng),
        _rand_start(rng),
    ]
    results = []
    for k, P0 in enumerate(seeds):
        R, P = _run_anneal(P0, seed=12345 + k, tlimit=1.0)
        if P is not None:
            results.append((R, P))
    if not results:
        results.append((_side(base), [list(map(float, p)) for p in base]))

    # Stage 2: elite pool with basin hopping, crossover, and fresh lattice restarts.
    rng2 = np.random.default_rng(777)
    results.sort(key=lambda t: t[0])
    elite = results[:4]
    it = 0
    while time.time() < search_end - 1.0:
        remaining = search_end - time.time()
        tl = min(1.0, remaining - 0.5)
        if tl < 0.3:
            break
        it += 1
        r = rng2.random()
        if it % 6 == 0:
            tilt = float(rng2.uniform(-10.0, 10.0))
            Q = _lattice_start(scale=1.01, tilt_deg=tilt)
            s0, T0 = 0.2, 0.01
        elif len(elite) >= 2 and r < 0.4:
            i, j = rng2.choice(len(elite), 2, replace=False)
            Q = _crossover(elite[int(i)][1], elite[int(j)][1], rng2)
            s0, T0 = 0.15, 0.008
        else:
            _, Pb = elite[int(rng2.integers(len(elite)))]
            Q = _perturb(Pb, rng2, sig_xy=0.03, sig_a=0.02)
            s0, T0 = 0.05, 0.004
        R, P = _run_anneal(Q, seed=int(rng2.integers(10 ** 9)), tlimit=tl,
                           s0=s0, T0=T0)
        if P is not None:
            results.append((R, P))
            if len(elite) < 4 or R < elite[-1][0] - 1e-9:
                elite.append((R, P))
                elite.sort(key=lambda t: t[0])
                elite = elite[:4]

    # Stage 3: SLSQP polish on the top distinct annealed basins.
    results.sort(key=lambda t: t[0])
    distinct = []
    for R, P in results:
        if all(abs(R - r0) > 1e-6 for r0, _ in distinct):
            distinct.append((R, P))
        if len(distinct) >= 10:
            break
    pool = []
    for R, P in distinct:
        pool.append(P)
        try:
            _, P2 = _polish(P, rounds=12)
            pool.append(P2)
        except Exception:
            pass

    valid_pool = []
    for P in pool:
        try:
            if _valid(P):
                valid_pool.append((_side(P), P))
        except Exception:
            continue
    valid_pool.sort(key=lambda t: t[0])

    # Stage 4: multi-start jitter-and-polish on the best valid candidates.
    rng3 = np.random.default_rng(2024)
    for _, P in valid_pool[:3]:
        if time.time() > t_start + 52.0:
            break
        try:
            R3, P3 = _multistart_polish(P, rng3, k=3, sig=0.008)
            if _valid(P3):
                valid_pool.append((R3, P3))
        except Exception:
            pass
    valid_pool.sort(key=lambda t: t[0])

    best_R, best_P = float("inf"), None
    for _, P in valid_pool:
        try:
            if _valid(P):
                R = _side(P)
                if R < best_R:
                    best_R, best_P = R, P
        except Exception:
            continue

    if best_P is None:
        # Fallback: verified baseline arrangement, side computed from its own containment.
        best_P = [list(map(float, p)) for p in base]
        best_R = _side(best_P)

    side = best_R + 1e-6
    inner = np.array(
        [[x, y, math.degrees(a)] for x, y, a in best_P], dtype=float
    )
    return inner, np.array([0.0, 0.0, 0.0]), float(side)


# EVOLVE-BLOCK-END
