# EVOLVE-BLOCK-START
"""Mixed/hex/drop seeds, basin hopping with Voronoi-hole gap moves,
exact LP radii, SLSQP polish, swap polish, and a final strict-improvement
perturbation stage."""
import time
import numpy as np
from scipy.optimize import minimize, linprog

N = 26
TIME_BUDGET = 30.0
FINAL_BUDGET = 34.0
PERTURB_END = 38.0
SEED_FRACTION = 0.4
CHAINS = 4
MAX_IT = 420
T_START = 0.003
T_END = 0.0002
RESET_EVERY = 70
RESET_JITTER = 0.005
RELAX_INFLATE = 0.11
LP_MARGIN = 1e-7
FINAL_SHRINK = 1.0 - 1e-7
TOL = 1e-9
SWAP_TRIALS = 30
PERTURB_SIGMA = 0.002

ROW_PATTERNS = [
    [5, 6, 5, 6, 4],
    [6, 5, 6, 5, 4],
    [4, 5, 6, 5, 6],
    [5, 5, 6, 5, 5],
    [6, 5, 5, 5, 5],
    [7, 6, 7, 6],
    [6, 7, 6, 7],
    [4, 6, 5, 6, 5],
    [5, 6, 6, 5, 4],
    [6, 6, 7, 7],
    [5, 5, 5, 5, 6],
    [3, 5, 6, 5, 4, 3],
    [4, 5, 4, 5, 4, 4],
]
DROP_PATTERNS = [
    [5, 6, 5, 6, 5],
    [5, 5, 6, 5, 6],
]


def hex_start(counts, rng=None, jitter=0.0):
    rows = len(counts)
    ys = np.linspace(0.08, 0.92, rows)
    cmax = max(counts)
    pts = []
    for k, c in enumerate(counts):
        shift = 0.025 if k % 2 else -0.025
        xs = np.linspace(0.08, 0.92, c) + shift * (c < cmax)
        for x in xs:
            pts.append((x, ys[k]))
    centers = np.array(pts, dtype=float)
    if rng is not None and jitter > 0:
        centers = centers + rng.normal(0, jitter, centers.shape)
    return np.clip(centers, 0.03, 0.97)


def mixed_start(rng, big=0.13, jitter=0.0):
    fixed = [
        (big, big), (1 - big, big), (big, 1 - big), (1 - big, 1 - big),
        (0.5, big * 0.9), (0.5, 1 - big * 0.9),
        (big * 0.9, 0.5), (1 - big * 0.9, 0.5),
    ]
    need = N - len(fixed)
    spacing = 0.165
    lattice = []
    y = 0.22
    row = 0
    while y < 0.8:
        x = 0.22 + (0.0 if row % 2 == 0 else spacing / 2)
        while x < 0.8:
            lattice.append((x, y))
            x += spacing
        y += spacing * np.sqrt(3) / 2
        row += 1
    lattice = np.array(lattice, dtype=float).reshape(-1, 2)
    if len(lattice) >= need:
        d = np.linalg.norm(lattice - 0.5, axis=1)
        interior = lattice[np.argsort(d)[:need]]
    else:
        extra = rng.uniform(0.2, 0.8, size=(need - len(lattice), 2))
        interior = np.vstack([lattice, extra])
    centers = np.vstack([np.array(fixed, dtype=float), interior])
    if jitter > 0:
        centers = centers + rng.normal(0, jitter, centers.shape)
    return np.clip(centers, 0.03, 0.97)


def _energy(flat, r):
    n = len(r)
    I, J = np.triu_indices(n, 1)
    P = flat.reshape(n, 2)
    diff = P[I] - P[J]
    d = np.sqrt((diff ** 2).sum(axis=1)) + 1e-12
    ov = np.maximum(r[I] + r[J] - d, 0.0)
    E = float(np.sum(ov ** 2))
    coef = (-2.0 * ov / d)[:, None] * diff
    G = np.zeros((n, 2))
    np.add.at(G, I, coef)
    np.add.at(G, J, -coef)
    lo = np.maximum(r[:, None] - P, 0.0)
    hi = np.maximum(P + r[:, None] - 1.0, 0.0)
    E += float(np.sum(lo ** 2) + np.sum(hi ** 2))
    G += -2.0 * lo + 2.0 * hi
    return E, G.ravel()


def relax(c, r_val=0.095):
    n = len(c)
    r = np.full(n, r_val)
    res = minimize(_energy, c.ravel().copy(), args=(r,), jac=True,
                   method="L-BFGS-B", bounds=[(0.0, 1.0)] * (2 * n),
                   options={"maxiter": 500})
    return np.clip(res.x.reshape(n, 2), 0.0, 1.0)


def lp_radii(c):
    n = len(c)
    c = np.clip(c, 0.0, 1.0)
    I, J = np.triu_indices(n, 1)
    M = len(I)
    wall = np.minimum(np.minimum(c[:, 0], c[:, 1]),
                      np.minimum(1 - c[:, 0], 1 - c[:, 1]))
    D = np.linalg.norm(c[I] - c[J], axis=1)
    A = np.zeros((M, n))
    rows = np.arange(M)
    A[rows, I] = 1.0
    A[rows, J] = 1.0
    bounds = [(0.0, max(float(w), 0.0)) for w in wall]
    res = linprog(-np.ones(n), A_ub=A, b_ub=D, bounds=bounds, method="highs")
    if res.status != 0 or res.x is None:
        return None
    return np.maximum(res.x - LP_MARGIN, 0.0)


def is_valid(c, r, tol=TOL):
    n = len(c)
    I, J = np.triu_indices(n, 1)
    if np.any(r < -tol):
        return False
    wall = np.minimum(np.minimum(c[:, 0], c[:, 1]),
                      np.minimum(1 - c[:, 0], 1 - c[:, 1]))
    if np.any(r > wall + tol):
        return False
    D = np.linalg.norm(c[I] - c[J], axis=1)
    return bool(np.all(r[I] + r[J] <= D + tol))


def evaluate(c):
    c = np.clip(c, 0.0, 1.0)
    r = lp_radii(c)
    if r is None or not is_valid(c, r):
        return c, np.zeros(len(c)), 0.0
    return c, r, float(r.sum())


def optimize(c0, r0, maxiter=400, ftol=1e-10):
    n = len(c0)
    I, J = np.triu_indices(n, 1)
    M = len(I)
    z0 = np.concatenate([c0[:, 0], c0[:, 1], r0])
    bounds = [(0.0, 1.0)] * (2 * n) + [(0.0, 0.5)] * n
    z0 = np.clip(z0, [b[0] for b in bounds], [b[1] for b in bounds])

    def obj(z):
        return -np.sum(z[2 * n:])

    def obj_jac(z):
        g = np.zeros_like(z)
        g[2 * n:] = -1.0
        return g

    def wall_fun(z):
        x, y, r = z[:n], z[n:2 * n], z[2 * n:]
        return np.concatenate([x - r, 1 - x - r, y - r, 1 - y - r])

    def wall_jac(z):
        Jm = np.zeros((4 * n, 3 * n))
        idx = np.arange(n)
        Jm[idx, idx] = 1.0
        Jm[idx, 2 * n + idx] = -1.0
        Jm[n + idx, idx] = -1.0
        Jm[n + idx, 2 * n + idx] = -1.0
        Jm[2 * n + idx, n + idx] = 1.0
        Jm[2 * n + idx, 2 * n + idx] = -1.0
        Jm[3 * n + idx, n + idx] = -1.0
        Jm[3 * n + idx, 2 * n + idx] = -1.0
        return Jm

    def pair_fun(z):
        x, y, r = z[:n], z[n:2 * n], z[2 * n:]
        dx = x[I] - x[J]
        dy = y[I] - y[J]
        s = r[I] + r[J]
        return dx * dx + dy * dy - s * s

    def pair_jac(z):
        x, y, r = z[:n], z[n:2 * n], z[2 * n:]
        dx = x[I] - x[J]
        dy = y[I] - y[J]
        s = r[I] + r[J]
        Jm = np.zeros((M, 3 * n))
        k = np.arange(M)
        Jm[k, I] = 2 * dx
        Jm[k, J] = -2 * dx
        Jm[k, n + I] = 2 * dy
        Jm[k, n + J] = -2 * dy
        Jm[k, 2 * n + I] = -2 * s
        Jm[k, 2 * n + J] = -2 * s
        return Jm

    constraints = [
        {"type": "ineq", "fun": wall_fun, "jac": wall_jac},
        {"type": "ineq", "fun": pair_fun, "jac": pair_jac},
    ]
    res = minimize(obj, z0, jac=obj_jac, bounds=bounds, constraints=constraints,
                   method="SLSQP", options={"maxiter": maxiter, "ftol": ftol})
    z = res.x
    return np.stack([z[:n], z[n:2 * n]], axis=1), z[2 * n:]


def polish_from(c0):
    c0 = relax(c0)
    r0 = lp_radii(c0)
    if r0 is None:
        return None
    try:
        c1, _ = optimize(c0, r0)
    except Exception:
        return None
    c1, r1, s1 = evaluate(c1)
    return (c1, r1, s1) if s1 > 0 else None


def drop_seed(counts):
    out = polish_from(hex_start(counts))
    if out is None:
        return None
    c27, r27, _ = out
    if len(c27) != N + 1:
        return None
    c26 = np.delete(c27, int(np.argmin(r27)), axis=0)
    return polish_from(c26)


def voronoi_holes(c):
    """Voronoi vertices of the centers that lie inside the square (empty-space candidates)."""
    try:
        from scipy.spatial import Voronoi
        v = Voronoi(c).vertices
        keep = np.all((v >= 0.03) & (v <= 0.97), axis=1)
        return v[keep]
    except Exception:
        return np.empty((0, 2))


def gap_reinsert(c, r, rng, samples=1000):
    """Move the smallest circle to the largest hole among random samples and Voronoi vertices."""
    n = len(c)
    c = c.copy()
    i = int(np.argmin(r))
    others = np.delete(np.arange(n), i)
    rand = rng.uniform(0.03, 0.97, size=(samples, 2))
    cand = np.vstack([rand, voronoi_holes(c[others])])
    d = np.linalg.norm(cand[:, None, :] - c[others][None, :, :], axis=2)
    clear = np.min(d - r[others][None, :], axis=1)
    wall = np.minimum(np.minimum(cand[:, 0], cand[:, 1]),
                      np.minimum(1 - cand[:, 0], 1 - cand[:, 1]))
    c[i] = cand[int(np.argmax(np.minimum(clear, wall)))]
    return c


def row_shift(c, rng):
    c2 = c.copy()
    k = int(rng.integers(len(c)))
    band = np.abs(c[:, 1] - c[k, 1]) < 0.06
    c2[band, 0] += rng.normal(0.0, 0.02)
    return np.clip(c2, 0.02, 0.98)


def perturb(c, r, rng):
    u = rng.random()
    if u < 0.35:
        return gap_reinsert(c, r, rng)
    if u < 0.55:
        return row_shift(c, rng)
    c2 = c.copy()
    if u < 0.8:
        small = int(rng.choice(np.argsort(r)[:3]))
        large = int(rng.choice(np.argsort(-r)[:6]))
        c2[[small, large]] = c2[[large, small]]
        return c2
    k = int(rng.integers(1, 3))
    movers = rng.choice(np.argsort(r)[:3 * k], size=k, replace=False)
    c2[movers] = rng.uniform(0.06, 0.94, size=(k, 2))
    return c2


def swap_polish(c, r, s, rng, deadline):
    best = (c, r, s)
    for _ in range(SWAP_TRIALS):
        if time.time() > deadline:
            break
        i, j = rng.choice(len(c), size=2, replace=False)
        c2 = best[0].copy()
        c2[[i, j]] = c2[[j, i]]
        out = polish_from(c2)
        if out is not None and out[2] > best[2] + 1e-9:
            best = out
    return best


def construct_packing():
    t0 = time.time()
    rng = np.random.default_rng(2024)

    seeds = []
    for counts in DROP_PATTERNS:
        seeds.append(("drop", counts))
    for k in range(6):
        seeds.append(("mixed", mixed_start(rng, big=0.12 + 0.01 * k, jitter=0.01 if k % 2 else 0.0)))
    for pattern in ROW_PATTERNS:
        seeds.append(("hex", hex_start(pattern)))
        seeds.append(("hex", hex_start(pattern, rng, 0.01)))

    pool = []
    for kind, s in seeds:
        if time.time() - t0 > SEED_FRACTION * TIME_BUDGET:
            break
        out = drop_seed(s) if kind == "drop" else polish_from(s)
        if out is not None and len(out[0]) == N:
            pool.append(out)

    if not pool:
        g = (np.arange(6) + 0.5) / 6
        pts = np.array([[x, y] for y in g for x in g][:N])
        r = np.full(N, 1.0 / 12.0 - 1e-9)
        return pts, r, float(r.sum())

    pool.sort(key=lambda t: -t[2])
    safe_c, safe_r, safe_s = pool[0]
    best_c, best_r, best_s = pool[0]

    chains = [(c.copy(), r.copy(), s) for c, r, s in pool[:CHAINS]]
    while len(chains) < CHAINS:
        chains.append((best_c.copy(), best_r.copy(), best_s))

    it = 0
    while it < MAX_IT and time.time() - t0 < TIME_BUDGET:
        frac = min(it / max(MAX_IT - 1, 1), 1.0)
        T = T_START * (T_END / T_START) ** frac
        scale = 0.02 * (1.0 - frac) + 0.003

        if it > 0 and it % RESET_EVERY == 0:
            for k in range(1, CHAINS):
                cj = np.clip(best_c + rng.normal(0, RESET_JITTER, best_c.shape), 0.02, 0.98)
                out = polish_from(cj)
                if out is not None:
                    chains[k] = out

        k_chain = it % CHAINS
        c_cur, r_cur, s_cur = chains[k_chain]

        c = perturb(c_cur, r_cur, rng)
        c = np.clip(c + rng.normal(0, scale, c.shape), 0.02, 0.98)
        c = relax(c, RELAX_INFLATE)
        r0 = lp_radii(c)
        if r0 is None:
            it += 1
            continue

        try:
            c1, _ = optimize(c, r0)
            c1, r1, s1 = evaluate(c1)
        except Exception:
            it += 1
            continue

        if s1 > 0 and (s1 >= s_cur or rng.random() < np.exp((s1 - s_cur) / T)):
            chains[k_chain] = (c1, r1, s1)
        if s1 > best_s + 1e-9:
            best_c, best_r, best_s = c1, r1, s1
        it += 1

    # Swap-polish the top distinct layouts.
    cands = [(best_c, best_r, best_s)] + [(c, r, s) for c, r, s in chains]
    cands = [x for x in cands if x[2] > 0]
    cands.sort(key=lambda t: -t[2])
    distinct = []
    seen = set()
    for c, r, s in cands:
        key = round(s, 6)
        if key not in seen:
            seen.add(key)
            distinct.append((c, r, s))
        if len(distinct) == 3:
            break
    for c, r, s in distinct:
        if time.time() - t0 > FINAL_BUDGET:
            break
        c_b, r_b, s_b = swap_polish(c, r, s, rng, t0 + FINAL_BUDGET)
        if s_b > best_s:
            best_c, best_r, best_s = c_b, r_b, s_b

    try:
        c1, _ = optimize(best_c, best_r, maxiter=800, ftol=1e-12)
        c1, r1, s1 = evaluate(c1)
        if s1 > best_s:
            best_c, best_r, best_s = c1, r1, s1
    except Exception:
        pass

    # Final stage: small-noise perturbation + relax-then-polish; strict improvements only.
    while time.time() - t0 < PERTURB_END:
        cp = np.clip(best_c + rng.normal(0, PERTURB_SIGMA, best_c.shape), 0.02, 0.98)
        out = polish_from(cp)
        if out is not None and out[2] > best_s + 1e-9 and is_valid(out[0], out[1]):
            best_c, best_r, best_s = out

    r_t = lp_radii(best_c)
    if r_t is not None:
        r_t = r_t * FINAL_SHRINK
        if is_valid(best_c, r_t, tol=1e-12) and float(r_t.sum()) > best_s:
            best_r, best_s = r_t, float(r_t.sum())

    if not is_valid(best_c, best_r):
        best_c, best_r, best_s = safe_c, safe_r, safe_s
    return best_c, best_r, float(best_s)


# EVOLVE-BLOCK-END


def run_packing():
    """Run the circle packing constructor for n=26"""
    centers, radii, sum_radii = construct_packing()
    return centers, radii, sum_radii


def visualize(centers, radii):
    """Visualize the circle packing"""
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
