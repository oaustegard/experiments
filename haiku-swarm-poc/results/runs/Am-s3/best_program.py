# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles"""
import time
import numpy as np

N = 26


def _rows_layout(counts=(5, 5, 6, 5, 5), ys=(0.1, 0.3, 0.5, 0.7, 0.9)):
    """Staggered rows of circles."""
    pts = []
    for c, y in zip(counts, ys):
        xs = np.linspace(0.1, 0.9, c) if c == 5 else np.linspace(0.08, 0.92, c)
        pts.extend([[x, y] for x in xs])
    pts = pts[:N]
    k = 0
    while len(pts) < N:  # pad if a layout is short
        pts.append([0.5 + 0.01 * k, 0.5])
        k += 1
    return np.array(pts)


def _ring_layout(k, radius, phase, m=0.1):
    """16 border circles + (26-16-k) center circles + ring of k circles."""
    pts = [[m, m], [1 - m, m], [m, 1 - m], [1 - m, 1 - m]]
    for t in (0.3, 0.5, 0.7):
        pts += [[t, m], [t, 1 - m], [m, t], [1 - m, t]]
    if k == 9:
        pts.append([0.5, 0.5])
    for i in range(k):
        a = phase + 2 * np.pi * i / k
        pts.append([0.5 + radius * np.cos(a), 0.5 + radius * np.sin(a)])
    return np.array(pts[:N])


def compute_max_radii(centers):
    """Max-sum radii for fixed centers via LP (proportional-scaling fallback)."""
    n = centers.shape[0]
    border = np.array([max(min(x, y, 1 - x, 1 - y), 0.0) for x, y in centers])
    try:
        from scipy.optimize import linprog

        I, J = np.triu_indices(n, 1)
        A = np.zeros((len(I), n))
        A[np.arange(len(I)), I] = 1.0
        A[np.arange(len(I)), J] = 1.0
        d = np.linalg.norm(centers[I] - centers[J], axis=1)
        res = linprog(-np.ones(n), A_ub=A, b_ub=d,
                      bounds=[(0, b) for b in border], method="highs")
        if res.success:
            return np.maximum(res.x - 1e-12, 0.0)
    except Exception:
        pass

    radii = border.copy()
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(centers[i] - centers[j])
            if radii[i] + radii[j] > dist:
                s = dist / (radii[i] + radii[j])
                radii[i] *= s
                radii[j] *= s
    return radii


def _refine(centers, maxiter=300):
    """Joint local refinement of centers and radii (SLSQP, analytic Jacobians)."""
    from scipy.optimize import minimize

    n = centers.shape[0]
    I, J = np.triu_indices(n, 1)
    m = len(I)
    ar = np.arange(m)
    k = np.arange(n)
    r0 = np.minimum(compute_max_radii(centers), 0.5)
    z0 = np.concatenate([centers[:, 0], centers[:, 1], r0])

    def obj(z):
        return -np.sum(z[2 * n:])

    def obj_jac(z):
        g = np.zeros(3 * n)
        g[2 * n:] = -1.0
        return g

    def cons(z):
        x, y, r = z[:n], z[n:2 * n], z[2 * n:]
        pair = (x[I] - x[J]) ** 2 + (y[I] - y[J]) ** 2 - (r[I] + r[J]) ** 2
        wall = np.concatenate([x - r, 1 - x - r, y - r, 1 - y - r])
        return np.concatenate([pair, wall])

    def cons_jac(z):
        x, y, r = z[:n], z[n:2 * n], z[2 * n:]
        Jm = np.zeros((m + 4 * n, 3 * n))
        dx = x[I] - x[J]
        dy = y[I] - y[J]
        s = r[I] + r[J]
        Jm[ar, I] = 2 * dx
        Jm[ar, J] = -2 * dx
        Jm[ar, n + I] = 2 * dy
        Jm[ar, n + J] = -2 * dy
        Jm[ar, 2 * n + I] = -2 * s
        Jm[ar, 2 * n + J] = -2 * s
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
        obj, z0, jac=obj_jac, method="SLSQP",
        constraints=[{"type": "ineq", "fun": cons, "jac": cons_jac}],
        bounds=[(0, 1)] * (2 * n) + [(0, 0.5)] * n,
        options={"maxiter": maxiter, "ftol": 1e-12},
    )
    z = res.x
    return np.clip(np.stack([z[:n], z[n:2 * n]], axis=1), 0.0, 1.0)


def _grid_fallback():
    g = np.linspace(0.1, 0.9, 5)
    c = np.array([[x, y] for y in g for x in g] + [[0.2, 0.2]])
    r = np.full(N, 0.1)
    r[25] = (np.sqrt(2) - 1) * 0.1
    r *= 1 - 1e-12
    return c, r, float(r.sum())


def construct_packing():
    """
    Explicit layouts (staggered rows, rings) -> LP radii -> SLSQP polish ->
    time-bounded seeded search (jitter, partial jitter, smallest-circle
    relocation) around the best few packings.

    Returns:
        Tuple of (centers, radii, sum_of_radii)
    """
    t0 = time.time()
    best = _grid_fallback()
    pool = []  # (sum, centers)

    def consider(c):
        nonlocal best
        if c.shape != (N, 2):
            return
        r = compute_max_radii(c)
        s = float(np.sum(r))
        pool.append((s, c))
        if s > best[2]:
            best = (c, r, s)

    ys5 = (0.1, 0.3, 0.5, 0.7, 0.9)
    ys_h = (0.09, 0.27, 0.5, 0.73, 0.91)
    starts = [_rows_layout(),
              _rows_layout((5, 6, 5, 6, 4), ys5),
              _rows_layout((4, 6, 6, 6, 4), ys5),
              _rows_layout((6, 5, 4, 5, 6), ys_h),
              _rows_layout((5, 6, 4, 6, 5), ys_h),
              _ring_layout(10, 0.30, 0.0),
              _ring_layout(10, 0.28, np.pi / 10),
              _ring_layout(9, 0.27, 0.0),
              _ring_layout(9, 0.27, np.pi / 9)]
    for s in starts:
        consider(s)
        try:
            consider(_refine(s))
        except Exception:
            pass

    rng = np.random.default_rng(0)
    it = 0
    while time.time() - t0 < 28 and it < 120:
        it += 1
        pool = sorted(pool, key=lambda t: -t[0])[:8]
        top = pool[:4]
        base = top[rng.integers(len(top))][1].copy()
        move = it % 3
        try:
            if move == 0:
                sigma = rng.choice([0.01, 0.02, 0.035, 0.06])
                p = base + rng.normal(0, sigma, base.shape)
            elif move == 1:
                p = base.copy()
                idx = rng.choice(N, size=int(rng.integers(3, 9)), replace=False)
                p[idx] += rng.normal(0, 0.06, (len(idx), 2))
            else:
                r = compute_max_radii(base)
                p = base.copy()
                i = int(np.argmin(r))
                p[i] = rng.uniform(0.1, 0.9, 2)
                p += rng.normal(0, 0.005, p.shape)
            p = np.clip(p, 0.02, 0.98)
            consider(_refine(p))
        except Exception:
            pass

    c, r, s = best
    return c, r, float(np.sum(r))


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