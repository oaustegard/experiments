# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles"""
import numpy as np

N = 26


def initial_layout(rows):
    """Explicit staggered-row arrangement given as [(y, count), ...]."""
    pts = []
    for y, k in rows:
        if k == 5:
            xs = 0.1 + 0.2 * np.arange(5)
        else:
            xs = (np.arange(k) + 0.5) / k
        for x in xs:
            pts.append([x, y])
    return np.clip(np.array(pts, dtype=float), 0.01, 0.99)


def construct_packing():
    """
    Build 26 circles: explicit staggered-row layouts, smooth joint refinement
    of centers and radii (SLSQP, analytic Jacobians) from the layouts and a
    few deterministic perturbations, then a short basin-hopping stage around
    the best result. Final radii are always recomputed by an exact LP so the
    result is valid.

    Returns:
        Tuple of (centers, radii, sum_of_radii)
    """
    bases = [
        initial_layout([(0.095, 5), (0.290, 5), (0.500, 6), (0.710, 5), (0.905, 5)]),
        initial_layout([(0.095, 6), (0.290, 5), (0.500, 5), (0.710, 5), (0.905, 5)]),
        initial_layout([(0.095, 5), (0.290, 6), (0.500, 5), (0.710, 5), (0.905, 5)]),
    ]
    best_c = bases[0]
    best_r = compute_max_radii(best_c)
    best_s = float(np.sum(best_r))

    rng = np.random.default_rng(0)
    starts = []
    for b in bases:
        starts.append(b)
        for sigma in (0.01, 0.02, 0.03, 0.015):
            starts.append(np.clip(b + rng.normal(0, sigma, b.shape), 0.03, 0.97))

    def attempt(start):
        try:
            c = refine(start)
            r = compute_max_radii(c)
            if np.all(np.isfinite(r)):
                return c, r, float(np.sum(r))
        except Exception:
            pass
        return None

    for start in starts:
        out = attempt(start)
        if out is not None and out[2] > best_s:
            best_c, best_r, best_s = out

    # Basin hopping around the incumbent
    for it in range(14):
        sigma = (0.008, 0.015, 0.025)[it % 3]
        start = np.clip(best_c + rng.normal(0, sigma, best_c.shape), 0.03, 0.97)
        out = attempt(start)
        if out is not None and out[2] > best_s:
            best_c, best_r, best_s = out

    return best_c, best_r, best_s


def refine(centers):
    """Maximize sum of radii over centers and radii jointly (SLSQP)."""
    from scipy.optimize import minimize

    n = centers.shape[0]
    r0 = compute_max_radii(centers)
    z0 = np.concatenate([centers[:, 0], centers[:, 1], r0 * 0.98])

    ii, jj = np.triu_indices(n, 1)
    m = len(ii)
    ar = np.arange(m)

    eye = np.eye(n)
    zero = np.zeros((n, n))
    Jb = np.vstack([
        np.hstack([eye, zero, -eye]),    # x - r >= 0
        np.hstack([-eye, zero, -eye]),   # 1 - x - r >= 0
        np.hstack([zero, eye, -eye]),    # y - r >= 0
        np.hstack([zero, -eye, -eye]),   # 1 - y - r >= 0
    ])

    g = np.zeros(3 * n)
    g[2 * n:] = -1.0

    def unpack(z):
        return z[:n], z[n:2 * n], z[2 * n:]

    def cons(z):
        x, y, r = unpack(z)
        dx = x[ii] - x[jj]
        dy = y[ii] - y[jj]
        d = np.sqrt(dx * dx + dy * dy + 1e-18)
        bord = np.concatenate([x - r, 1 - x - r, y - r, 1 - y - r])
        return np.concatenate([d - r[ii] - r[jj], bord])

    def cons_jac(z):
        x, y, r = unpack(z)
        dx = x[ii] - x[jj]
        dy = y[ii] - y[jj]
        d = np.sqrt(dx * dx + dy * dy + 1e-18)
        J = np.zeros((m, 3 * n))
        J[ar, ii] = dx / d
        J[ar, jj] = -dx / d
        J[ar, n + ii] = dy / d
        J[ar, n + jj] = -dy / d
        J[ar, 2 * n + ii] = -1.0
        J[ar, 2 * n + jj] = -1.0
        return np.vstack([J, Jb])

    bounds = [(0.0, 1.0)] * (2 * n) + [(0.0, 0.5)] * n
    res = minimize(
        lambda z: -np.sum(z[2 * n:]), z0, jac=lambda z: g, method="SLSQP",
        bounds=bounds,
        constraints=[{"type": "ineq", "fun": cons, "jac": cons_jac}],
        options={"maxiter": 400, "ftol": 1e-11},
    )
    x, y, _ = unpack(res.x)
    return np.clip(np.column_stack([x, y]), 1e-6, 1 - 1e-6)


def compute_max_radii(centers):
    """
    Maximize sum of radii for fixed centers (linear program):
        r_i <= distance to square borders
        r_i + r_j <= dist(i, j)
    Falls back to pairwise rescaling if scipy is unavailable.
    """
    n = centers.shape[0]
    border = np.minimum.reduce(
        [centers[:, 0], centers[:, 1], 1 - centers[:, 0], 1 - centers[:, 1]]
    )
    border = np.maximum(border, 0.0)

    try:
        from scipy.optimize import linprog

        ii, jj = np.triu_indices(n, 1)
        dists = np.linalg.norm(centers[ii] - centers[jj], axis=1)
        m = len(dists)
        A = np.zeros((m, n))
        A[np.arange(m), ii] = 1.0
        A[np.arange(m), jj] = 1.0
        res = linprog(
            -np.ones(n),
            A_ub=A,
            b_ub=dists,
            bounds=[(0, b) for b in border],
            method="highs",
        )
        if res.success:
            return np.maximum(res.x - 1e-12, 0.0)
    except Exception:
        pass

    radii = border.copy()
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(centers[i] - centers[j])
            if radii[i] + radii[j] > dist:
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
    visualize(centers, radii)