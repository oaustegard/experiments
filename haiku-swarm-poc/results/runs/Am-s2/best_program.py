# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles"""
import time
import numpy as np


def _layout(counts, stagger=False):
    """Explicit row layout; optionally staggered (hex-like) columns."""
    k = len(counts)
    mx = max(counts)
    pts = []
    for r, c in enumerate(counts):
        y = (r + 0.5) / k
        for i in range(c):
            if stagger:
                w = mx + 0.5
                off = 0.25 if r % 2 == 0 else 0.75
                x = (i + off + (mx - c) * 0.5) / w
            else:
                x = (i + 0.5) / c
            pts.append([x, y])
    return np.array(pts, dtype=float)


def construct_packing():
    """
    Explicit row layouts -> SLSQP polish (centers+radii) -> basin hopping
    around the best packing within a time budget -> LP-exact radii.

    Returns:
        Tuple of (centers, radii, sum_of_radii)
    """
    t0 = time.time()
    budget = 20.0
    patterns = [[5, 5, 6, 5, 5], [5, 6, 5, 6, 4], [6, 5, 5, 5, 5],
                [6, 5, 6, 5, 4], [4, 6, 5, 6, 5]]
    bases = []
    for p in patterns:
        bases.append(_layout(p))
        bases.append(_layout(p, stagger=True))

    rng = np.random.RandomState(0)
    best_c = np.clip(bases[0], 0.01, 0.99)
    best_r = compute_max_radii(best_c)
    best_s = float(np.sum(best_r))

    def attempt(start):
        nonlocal best_c, best_r, best_s
        try:
            c2 = _polish(np.clip(start, 0.03, 0.97))
            r2 = compute_max_radii(c2)
            s2 = float(np.sum(r2))
            if s2 > best_s:
                best_c, best_r, best_s = c2, r2, s2
        except Exception:
            pass

    # Phase 1: structured starts
    for b in bases:
        if time.time() - t0 > budget * 0.5:
            break
        attempt(b)
    for scale in (0.02, 0.04):
        for b in bases:
            if time.time() - t0 > budget * 0.5:
                break
            attempt(b + rng.uniform(-scale, scale, b.shape))

    # Phase 2: basin hopping around the incumbent
    while time.time() - t0 < budget:
        s = best_c.copy()
        mode = rng.rand()
        if mode < 0.6:
            s += rng.uniform(-1, 1, s.shape) * rng.choice([0.01, 0.02, 0.035])
        else:
            s += rng.uniform(-0.01, 0.01, s.shape)
            j = int(np.argmin(best_r))
            s[j] = rng.uniform(0.1, 0.9, 2)
        attempt(s)
    return best_c, best_r, best_s


def _polish(centers):
    from scipy.optimize import minimize

    n = centers.shape[0]
    r0 = compute_max_radii(centers)
    v0 = np.concatenate([centers[:, 0], centers[:, 1], r0 * 0.95])
    I, J = np.triu_indices(n, 1)
    m = len(I)
    ar = np.arange(m)
    k = np.arange(n)

    def obj(v):
        return -np.sum(v[2 * n:])

    def obj_jac(v):
        g = np.zeros(3 * n)
        g[2 * n:] = -1.0
        return g

    def cons(v):
        x, y, r = v[:n], v[n:2 * n], v[2 * n:]
        d = (x[I] - x[J]) ** 2 + (y[I] - y[J]) ** 2 - (r[I] + r[J]) ** 2
        return np.concatenate([d, x - r, 1 - x - r, y - r, 1 - y - r])

    def cons_jac(v):
        x, y, r = v[:n], v[n:2 * n], v[2 * n:]
        Jm = np.zeros((m + 4 * n, 3 * n))
        dx = 2 * (x[I] - x[J])
        dy = 2 * (y[I] - y[J])
        s = -2 * (r[I] + r[J])
        Jm[ar, I] = dx
        Jm[ar, J] = -dx
        Jm[ar, n + I] = dy
        Jm[ar, n + J] = -dy
        Jm[ar, 2 * n + I] = s
        Jm[ar, 2 * n + J] = s
        Jm[m + k, k] = 1
        Jm[m + k, 2 * n + k] = -1
        Jm[m + n + k, k] = -1
        Jm[m + n + k, 2 * n + k] = -1
        Jm[m + 2 * n + k, n + k] = 1
        Jm[m + 2 * n + k, 2 * n + k] = -1
        Jm[m + 3 * n + k, n + k] = -1
        Jm[m + 3 * n + k, 2 * n + k] = -1
        return Jm

    res = minimize(
        obj, v0, jac=obj_jac, method="SLSQP",
        bounds=[(0, 1)] * (2 * n) + [(0, 0.5)] * n,
        constraints=[{"type": "ineq", "fun": cons, "jac": cons_jac}],
        options={"maxiter": 400, "ftol": 1e-12},
    )
    v = res.x
    return np.clip(np.stack([v[:n], v[n:2 * n]], axis=1), 0.0, 1.0)


def compute_max_radii(centers):
    """
    Maximize sum of radii for fixed centers subject to staying inside the
    unit square and not overlapping. Uses an LP; falls back to scaling.
    """
    n = centers.shape[0]
    border = np.minimum.reduce(
        [centers[:, 0], centers[:, 1], 1 - centers[:, 0], 1 - centers[:, 1]]
    )
    try:
        from scipy.optimize import linprog

        iu, ju = np.triu_indices(n, 1)
        A = np.zeros((len(iu), n))
        A[np.arange(len(iu)), iu] = 1.0
        A[np.arange(len(iu)), ju] = 1.0
        b = np.linalg.norm(centers[iu] - centers[ju], axis=1)
        res = linprog(
            -np.ones(n),
            A_ub=A,
            b_ub=b,
            bounds=[(0, max(float(bd), 0.0)) for bd in border],
            method="highs",
        )
        if res.success:
            return np.maximum(res.x - 1e-12, 0.0)
    except Exception:
        pass

    radii = np.maximum(border, 0.0)
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(centers[i] - centers[j])
            if radii[i] + radii[j] > dist:
                s = dist / (radii[i] + radii[j])
                radii[i] *= s
                radii[j] *= s
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
    visualize(centers, radii)