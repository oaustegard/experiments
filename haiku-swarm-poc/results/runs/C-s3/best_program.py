# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles:
staggered-row seeds -> SLSQP (full pairwise constraints) -> exact LP radii,
then basin hopping around the best basin (relocate small circles into gaps,
jitter), improvement-only acceptance, final polish and strict-validity repair."""
import time
import numpy as np
from scipy.optimize import minimize, linprog

N = 26
_I, _J = np.triu_indices(N, 1)


def lp_radii(centers):
    """Maximize sum(r) for fixed centers: r_i + r_j <= d_ij, r_i <= wall distance."""
    c = np.clip(centers, 0.0, 1.0)
    wall = np.minimum(np.minimum(c[:, 0], 1 - c[:, 0]), np.minimum(c[:, 1], 1 - c[:, 1]))
    d = np.sqrt(((c[_I] - c[_J]) ** 2).sum(axis=1))
    m = len(_I)
    A = np.zeros((m, N))
    A[np.arange(m), _I] = 1.0
    A[np.arange(m), _J] = 1.0
    res = linprog(-np.ones(N), A_ub=A, b_ub=d,
                  bounds=[(0, w) for w in wall], method="highs")
    if res.status != 0:
        return compute_max_radii(c)
    return np.maximum(res.x, 0.0)


def row_layout(pattern, rng, jitter):
    rows = len(pattern)
    pts = []
    for k, m in enumerate(pattern):
        y = (k + 0.5) / rows
        for j in range(m):
            x = (j + 0.5) / m
            pts.append([x, y])
    pts = np.array(pts)
    pts += rng.normal(scale=jitter, size=pts.shape)
    return np.clip(pts, 0.03, 0.97)


_BOUNDS = [(0, 1)] * (2 * N) + [(0, 0.5)] * N
_NPAIR = len(_I)
_IDX = np.arange(N)
_ROWS = 4 * N + np.arange(_NPAIR)


def _obj(z):
    return -np.sum(z[2 * N:])


def _obj_grad(z):
    g = np.zeros(3 * N)
    g[2 * N:] = -1.0
    return g


def _cons(z):
    x, y, r = z[:N], z[N:2 * N], z[2 * N:]
    dx = x[_I] - x[_J]
    dy = y[_I] - y[_J]
    s = r[_I] + r[_J]
    return np.concatenate([x - r, 1 - x - r, y - r, 1 - y - r,
                           dx * dx + dy * dy - s * s])


def _cons_jac(z):
    x, y, r = z[:N], z[N:2 * N], z[2 * N:]
    J = np.zeros((4 * N + _NPAIR, 3 * N))
    idx = _IDX
    J[idx, idx] = 1; J[idx, 2 * N + idx] = -1
    J[N + idx, idx] = -1; J[N + idx, 2 * N + idx] = -1
    J[2 * N + idx, N + idx] = 1; J[2 * N + idx, 2 * N + idx] = -1
    J[3 * N + idx, N + idx] = -1; J[3 * N + idx, 2 * N + idx] = -1
    dx = x[_I] - x[_J]
    dy = y[_I] - y[_J]
    s = r[_I] + r[_J]
    J[_ROWS, _I] = 2 * dx
    J[_ROWS, _J] = -2 * dx
    J[_ROWS, N + _I] = 2 * dy
    J[_ROWS, N + _J] = -2 * dy
    J[_ROWS, 2 * N + _I] = -2 * s
    J[_ROWS, 2 * N + _J] = -2 * s
    return J


def optimize(centers0, r0, maxiter=300, ftol=1e-12):
    z0 = np.concatenate([centers0[:, 0], centers0[:, 1], r0])
    res = minimize(_obj, z0, jac=_obj_grad, method="SLSQP", bounds=_BOUNDS,
                   constraints=[{"type": "ineq", "fun": _cons, "jac": _cons_jac}],
                   options={"maxiter": maxiter, "ftol": ftol})
    z = res.x
    return np.clip(np.stack([z[:N], z[N:2 * N]], axis=1), 0.0, 1.0)


def _gap_point(centers, radii, skip, rng, ncand=1500):
    mask = np.ones(len(centers), bool)
    mask[skip] = False
    others = centers[mask]
    orad = radii[mask]
    cand = rng.uniform(0.03, 0.97, (ncand, 2))
    dd = np.sqrt(((cand[:, None, :] - others[None, :, :]) ** 2).sum(-1)) - orad[None, :]
    clear = dd.min(axis=1)
    wall = np.minimum.reduce([cand[:, 0], cand[:, 1], 1 - cand[:, 0], 1 - cand[:, 1]])
    clear = np.minimum(clear, wall)
    top = np.argsort(-clear)[:6]
    return cand[rng.choice(top)]


def construct_packing():
    rng = np.random.default_rng(12345)
    patterns = [
        [5, 5, 5, 5, 6],
        [5, 6, 5, 6, 4],
        [6, 5, 6, 5, 4],
        [5, 5, 4, 5, 4, 3],
        [4, 5, 4, 5, 4, 4],
        [5, 4, 5, 4, 5, 3],
        [6, 5, 5, 5, 5],
    ]
    t0 = time.time()
    best_c, best_r, best_s = None, None, -1.0
    trial = 0
    # Stage 1: structured seeds
    while time.time() - t0 < 9.0 and trial < 40:
        pat = patterns[trial % len(patterns)]
        jit = 0.0 if trial < len(patterns) else 0.03 + 0.01 * (trial % 4)
        c0 = row_layout(pat, rng, jit)
        try:
            r0 = lp_radii(c0)
            if r0.sum() > best_s:
                best_c, best_r, best_s = c0.copy(), r0, r0.sum()
            c1 = optimize(c0, r0)
            r1 = lp_radii(c1)
            if r1.sum() > best_s:
                best_c, best_r, best_s = c1.copy(), r1, r1.sum()
        except Exception:
            pass
        trial += 1

    # Stage 2: basin hopping around the best basin (improvement-only)
    total = 19.0
    it = 0
    while time.time() - t0 < total:
        it += 1
        c = best_c.copy()
        r = best_r.copy()
        mode = rng.random()
        if mode < 0.5:
            k = int(rng.integers(1, 3))
            order = np.argsort(r)
            pick = list(order[:k + 2])
            rng.shuffle(pick)
            pick = pick[:k]
            for i in pick:
                c[i] = _gap_point(c, r, i, rng)
                r[i] = 0.0
        elif mode < 0.9:
            sig = 0.003 + 0.02 * rng.random()
            c = np.clip(c + rng.normal(0, sig, c.shape), 0.02, 0.98)
        else:
            i, j = rng.choice(N, 2, replace=False)
            c[[i, j]] = c[[j, i]]
            c = np.clip(c + rng.normal(0, 0.005, c.shape), 0.02, 0.98)
        try:
            r0 = lp_radii(c)
            cn = optimize(c, r0, maxiter=200, ftol=1e-11)
            rn = lp_radii(cn)
            sn = float(rn.sum())
            if sn > best_s + 1e-10:
                best_c, best_r, best_s = cn, rn, sn
        except Exception:
            pass

    # Final polish
    try:
        cp = optimize(best_c, best_r, maxiter=500, ftol=1e-14)
        rp = lp_radii(cp)
        if rp.sum() > best_s:
            best_c, best_r, best_s = cp, rp, float(rp.sum())
    except Exception:
        pass

    centers = np.clip(best_c, 0.0, 1.0)
    radii = np.maximum(best_r - 1e-10, 0.0)
    wall = np.minimum(np.minimum(centers[:, 0], 1 - centers[:, 0]),
                      np.minimum(centers[:, 1], 1 - centers[:, 1]))
    radii = np.minimum(radii, np.maximum(wall - 1e-12, 0.0))
    d = np.sqrt(((centers[_I] - centers[_J]) ** 2).sum(axis=1))
    for _ in range(10):
        over = radii[_I] + radii[_J] - d
        bad = np.where(over > -1e-13)[0]
        if len(bad) == 0:
            break
        for k in bad:
            s = d[k] / (radii[_I[k]] + radii[_J[k]]) * (1 - 1e-12)
            if s < 1:
                radii[_I[k]] *= s
                radii[_J[k]] *= s
    return centers, radii, float(np.sum(radii))


def compute_max_radii(centers):
    """Fallback greedy radii computation."""
    n = centers.shape[0]
    radii = np.array([min(x, y, 1 - x, 1 - y) for x, y in centers])
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if radii[i] + radii[j] > dist and dist > 0:
                scale = dist / (radii[i] + radii[j])
                radii[i] *= scale
                radii[j] *= scale
    return radii


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

    # Draw unit square
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(True)

    # Draw circles
    for i, (center, radius) in enumerate(zip(centers, radii)):
        circle = Circle(center, radius, alpha=0.5)
        ax.add_patch(circle)
        ax.text(center[0], center[1], str(i), ha="center", va="center")

    plt.title(f"Circle Packing (n={len(centers)}, sum={sum(radii):.6f})")
    plt.show()


if __name__ == "__main__":
    centers, radii, sum_radii = run_packing()
    print(f"Sum of radii: {sum_radii}")
    # AlphaEvolve improved this to 2.635

    # Uncomment to visualize:
    # visualize(centers, radii)
