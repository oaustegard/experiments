# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles"""
import time
import numpy as np


def _ring_centers(m=0.08, inner=0.25):
    """16 on the border, 9 on an inner ring, 1 at the center."""
    pts = []
    L = 1 - 2 * m
    for k in range(16):
        s = k * L / 4
        side = int(s // L)
        u = s - side * L
        if side == 0:
            pts.append([m + u, m])
        elif side == 1:
            pts.append([1 - m, m + u])
        elif side == 2:
            pts.append([1 - m - u, 1 - m])
        else:
            pts.append([m, 1 - m - u])
    for i in range(9):
        a = 2 * np.pi * i / 9
        pts.append([0.5 + inner * np.cos(a), 0.5 + inner * np.sin(a)])
    pts.append([0.5, 0.5])
    return np.array(pts)


def _row_centers():
    """Staggered rows (5,5,6,5,5)."""
    pts = []
    for y, c in zip([0.1, 0.3, 0.5, 0.7, 0.9], [5, 5, 6, 5, 5]):
        if c == 5:
            xs = 0.1 + 0.2 * np.arange(5)
        else:
            xs = (np.arange(6) + 0.5) / 6
        for x in xs:
            pts.append([x, y])
    return np.array(pts)


def _hex_centers():
    """Hex-like rows (5,6,5,6,4) with alternating offsets."""
    pts = []
    counts = [5, 6, 5, 6, 4]
    ys = np.linspace(0.09, 0.91, 5)
    for y, c in zip(ys, counts):
        xs = (np.arange(c) + 0.5) / c
        for x in xs:
            pts.append([x, y])
    return np.array(pts)


def _polish(centers, radii, maxiter=300):
    """SLSQP refinement of centers and radii (analytic Jacobians)."""
    from scipy.optimize import minimize

    n = len(centers)
    iu, ju = np.triu_indices(n, 1)
    m = len(iu)
    ar = np.arange(m)

    def pair(z):
        x, y, r = z[:n], z[n:2 * n], z[2 * n:]
        s = r[iu] + r[ju]
        return (x[iu] - x[ju]) ** 2 + (y[iu] - y[ju]) ** 2 - s * s

    def pair_jac(z):
        x, y, r = z[:n], z[n:2 * n], z[2 * n:]
        dx = x[iu] - x[ju]
        dy = y[iu] - y[ju]
        s = r[iu] + r[ju]
        J = np.zeros((m, 3 * n))
        J[ar, iu] = 2 * dx
        J[ar, ju] = -2 * dx
        J[ar, n + iu] = 2 * dy
        J[ar, n + ju] = -2 * dy
        J[ar, 2 * n + iu] = -2 * s
        J[ar, 2 * n + ju] = -2 * s
        return J

    def wall(z):
        x, y, r = z[:n], z[n:2 * n], z[2 * n:]
        return np.concatenate([x - r, 1 - x - r, y - r, 1 - y - r])

    I = np.eye(n)
    Z = np.zeros((n, n))
    wall_J = np.block([[I, Z, -I], [-I, Z, -I], [Z, I, -I], [Z, -I, -I]])
    g = np.zeros(3 * n)
    g[2 * n:] = -1

    z0 = np.concatenate([centers[:, 0], centers[:, 1], radii])
    res = minimize(
        lambda z: -np.sum(z[2 * n:]), z0, jac=lambda z: g, method="SLSQP",
        bounds=[(0, 1)] * (2 * n) + [(0, 0.5)] * n,
        constraints=[
            {"type": "ineq", "fun": pair, "jac": pair_jac},
            {"type": "ineq", "fun": wall, "jac": lambda z: wall_J},
        ],
        options={"maxiter": maxiter, "ftol": 1e-11},
    )
    return np.clip(np.stack([res.x[:n], res.x[n:2 * n]], axis=1), 0, 1)


def _evaluate(s):
    """Polish a start, return best (centers, radii, sum) among raw/polished."""
    s = np.clip(s, 0.01, 0.99)
    r0 = compute_max_radii(s)
    best = (s, r0, float(np.sum(r0)))
    try:
        c = _polish(s, r0 * 0.9)
        r = compute_max_radii(c)
        if np.sum(r) > best[2]:
            best = (c, r, float(np.sum(r)))
    except Exception:
        pass
    return best


def _gap_point(centers, radii, rng, skip, k=300):
    """Sample points; return the one with largest free clearance."""
    cand = rng.uniform(0.05, 0.95, (k, 2))
    keep = np.ones(len(centers), bool)
    keep[skip] = False
    d = np.linalg.norm(cand[:, None, :] - centers[None, keep, :], axis=2)
    free = (d - radii[keep][None, :]).min(axis=1)
    free = np.minimum(free, np.minimum(cand, 1 - cand).min(axis=1))
    return cand[int(np.argmax(free))]


def construct_packing():
    """
    Multi-start from structured layouts (ring, rows, hex), then basin hopping
    (perturb / swap / relocate, SLSQP polish, exact LP radii). Time limited.

    Returns:
        Tuple of (centers, radii, sum_of_radii)
    """
    t0 = time.time()
    rng = np.random.default_rng(0)
    ring = _ring_centers()
    starts = [ring, _row_centers(), _hex_centers(),
              _ring_centers(0.09, 0.27), _ring_centers(0.07, 0.23)]
    for _ in range(4):
        starts.append(np.clip(ring + rng.normal(0, 0.02, ring.shape), 0.03, 0.97))

    best = None
    for k, s in enumerate(starts):
        if k > 2 and time.time() - t0 > 10:
            break
        cand = _evaluate(s)
        if best is None or cand[2] > best[2]:
            best = cand

    cur = best
    stall = 0
    it = 0
    while time.time() - t0 < 28:
        it += 1
        sigma = [0.008, 0.015, 0.025, 0.04][it % 4]
        p = cur[0] + rng.normal(0, sigma, cur[0].shape)
        mode = it % 5
        if mode == 0:
            i = int(np.argmin(cur[1]))
            p[i] = _gap_point(cur[0], cur[1], rng, i)
        elif mode == 1:
            i, j = rng.choice(len(p), 2, replace=False)
            p[[i, j]] = p[[j, i]]
        cand = _evaluate(p)
        if cand[2] > best[2]:
            best = cand
        if cand[2] > cur[2] - 1e-9:
            if cand[2] > cur[2] + 1e-9:
                stall = 0
            cur = cand
        else:
            stall += 1
        if stall > 15:
            cur = best
            stall = 0
    return best[0], best[1], best[2]


def compute_max_radii(centers):
    """
    Optimal radii for fixed centers (LP maximizing the sum of radii subject
    to border and non-overlap constraints); greedy scaling as fallback.
    """
    n = centers.shape[0]
    border = np.minimum(
        np.minimum(centers[:, 0], 1 - centers[:, 0]),
        np.minimum(centers[:, 1], 1 - centers[:, 1]),
    )
    border = np.maximum(border, 0)
    try:
        from scipy.optimize import linprog

        iu, ju = np.triu_indices(n, 1)
        A = np.zeros((len(iu), n))
        A[np.arange(len(iu)), iu] = 1
        A[np.arange(len(iu)), ju] = 1
        b = np.linalg.norm(centers[iu] - centers[ju], axis=1)
        res = linprog(-np.ones(n), A_ub=A, b_ub=b,
                      bounds=[(0, v) for v in border], method="highs")
        if res.success:
            return np.maximum(res.x - 1e-12, 0)
    except Exception:
        pass

    radii = border.copy()
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            if radii[i] + radii[j] > d:
                s = d / (radii[i] + radii[j])
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