# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles"""
import time
import numpy as np

N = 26


def _seed(rows, ys, jitter, seed):
    """Staggered-row layout of 26 centers, optionally jittered."""
    pts = []
    for k, (cnt, y) in enumerate(zip(rows, ys)):
        pts.extend([(i + 0.5) / cnt, y] for i in range(cnt))
    pts = np.array(pts, dtype=float)
    if jitter:
        pts += np.random.default_rng(seed).normal(0, jitter, pts.shape)
    return np.clip(pts, 0.05, 0.95)


def _refine(c0, maxiter=400):
    """Jointly optimize centers and radii with SLSQP (analytic Jacobians)."""
    from scipy.optimize import minimize

    n = N
    I, J = np.triu_indices(n, 1)
    m = len(I)
    ar = np.arange(m)

    def pair(v):
        d = np.hypot(v[I] - v[J], v[n + I] - v[n + J])
        return d - v[2 * n + I] - v[2 * n + J]

    def pair_jac(v):
        dx, dy = v[I] - v[J], v[n + I] - v[n + J]
        d = np.maximum(np.hypot(dx, dy), 1e-12)
        M = np.zeros((m, 3 * n))
        M[ar, I], M[ar, J] = dx / d, -dx / d
        M[ar, n + I], M[ar, n + J] = dy / d, -dy / d
        M[ar, 2 * n + I] = M[ar, 2 * n + J] = -1.0
        return M

    e, Z = np.eye(n), np.zeros((n, n))
    A = np.vstack([
        np.hstack([e, Z, -e]),    # x - r >= 0
        np.hstack([-e, Z, -e]),   # 1 - x - r >= 0
        np.hstack([Z, e, -e]),    # y - r >= 0
        np.hstack([Z, -e, -e]),   # 1 - y - r >= 0
    ])
    off = np.concatenate([np.zeros(n), np.ones(n), np.zeros(n), np.ones(n)])

    v0 = np.concatenate([c0[:, 0], c0[:, 1], np.full(n, 0.07)])
    g = np.concatenate([np.zeros(2 * n), -np.ones(n)])
    res = minimize(
        lambda v: -np.sum(v[2 * n:]), v0, jac=lambda v: g,
        bounds=[(0, 1)] * (2 * n) + [(0, 0.5)] * n,
        constraints=[
            {"type": "ineq", "fun": pair, "jac": pair_jac},
            {"type": "ineq", "fun": lambda v: A @ v + off, "jac": lambda v: A},
        ],
        method="SLSQP", options={"maxiter": maxiter, "ftol": 1e-12},
    )
    v = res.x
    return np.column_stack([v[:n], v[n:2 * n]])


def construct_packing():
    """
    Build 26 circles from staggered seeds refined by SLSQP, followed by
    basin-hopping perturbations of the best layout; final radii come from
    an exact LP so the result is always valid.

    Returns:
        Tuple of (centers, radii, sum_of_radii)
    """
    t0 = time.time()
    budget = 22.0

    base = _seed([5, 5, 6, 5, 5], [0.1, 0.3, 0.5, 0.7, 0.9], 0, 0)
    best_c = base
    best_r = compute_max_radii(base)
    best_s = float(np.sum(best_r))

    ys6 = [0.09, 0.29, 0.5, 0.71, 0.91]
    ys5 = [0.1, 0.3, 0.5, 0.7, 0.9]
    seeds = [
        _seed([6, 5, 6, 5, 4], ys6, 0, 0),
        _seed([5, 6, 5, 6, 4], ys6, 0, 0),
        _seed([4, 5, 6, 5, 6], ys6, 0, 0),
        _seed([5, 5, 6, 5, 5], ys5, 0.01, 1),
        _seed([6, 5, 5, 5, 5], ys5, 0.01, 2),
    ]
    for k in range(1, 4):
        seeds.append(_seed([6, 5, 6, 5, 4], ys6, 0.015, k))

    def consider(c0):
        nonlocal best_c, best_r, best_s
        try:
            c = np.clip(_refine(c0), 0.0, 1.0)
            r = compute_max_radii(c)
            s = float(np.sum(r))
            if s > best_s:
                best_c, best_r, best_s = c, r, s
        except Exception:
            pass

    for c0 in seeds:
        if time.time() - t0 > budget * 0.5:
            break
        consider(c0)

    # Basin hopping around the incumbent
    rng = np.random.default_rng(7)
    it = 0
    while time.time() - t0 < budget:
        sig = (0.01, 0.02, 0.035)[it % 3]
        c0 = np.clip(best_c + rng.normal(0, sig, best_c.shape), 0.03, 0.97)
        if it % 4 == 3:  # occasionally relocate the smallest circle
            j = int(np.argmin(best_r))
            c0[j] = rng.uniform(0.1, 0.9, 2)
        consider(c0)
        it += 1
    return best_c, best_r, best_s


def compute_max_radii(centers):
    """
    Maximize sum of radii for fixed centers subject to containment in the
    unit square and non-overlap (LP). Falls back to a scaling heuristic.
    """
    n = centers.shape[0]
    border = np.maximum(np.minimum.reduce(
        [centers[:, 0], centers[:, 1], 1 - centers[:, 0], 1 - centers[:, 1]]
    ), 0.0)
    try:
        from scipy.optimize import linprog

        I, J = np.triu_indices(n, 1)
        rhs = np.linalg.norm(centers[I] - centers[J], axis=1)
        A = np.zeros((len(I), n))
        A[np.arange(len(I)), I] = 1
        A[np.arange(len(I)), J] = 1
        res = linprog(-np.ones(n), A_ub=A, b_ub=rhs,
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