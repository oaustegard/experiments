# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles"""
import time
import numpy as np


def _border(c):
    return np.minimum(np.minimum(c[:, 0], 1 - c[:, 0]),
                      np.minimum(c[:, 1], 1 - c[:, 1]))


def _row_centers():
    counts = [5, 5, 6, 5, 5]
    ys = [0.1, 0.3, 0.5, 0.7, 0.9]
    pts = []
    for row, (cnt, y) in enumerate(zip(counts, ys)):
        if cnt == 6:
            xs = (np.arange(6) + 0.5) / 6.0
        elif row % 2 == 0:
            xs = 0.1 + 0.2 * np.arange(5)
        else:
            xs = 0.12 + 0.19 * np.arange(5) + 0.02
        for x in xs:
            pts.append([x, y])
    return np.array(pts)


def _slsqp(c0, n, iu, ju):
    """Jointly optimise centers and radii with SLSQP (analytic Jacobians)."""
    from scipy.optimize import minimize
    m = len(iu)
    ar = np.arange(m)
    k = np.arange(n)

    def obj(z):
        return -np.sum(z[2 * n:])

    def obj_jac(z):
        g = np.zeros(3 * n)
        g[2 * n:] = -1.0
        return g

    def cons(z):
        x, y, r = z[:n], z[n:2 * n], z[2 * n:]
        pair = (x[iu] - x[ju]) ** 2 + (y[iu] - y[ju]) ** 2 - (r[iu] + r[ju]) ** 2
        return np.concatenate([pair, x - r, 1 - x - r, y - r, 1 - y - r])

    def cons_jac(z):
        x, y, r = z[:n], z[n:2 * n], z[2 * n:]
        J = np.zeros((m + 4 * n, 3 * n))
        dx = 2 * (x[iu] - x[ju])
        dy = 2 * (y[iu] - y[ju])
        dr = -2 * (r[iu] + r[ju])
        J[ar, iu] = dx
        J[ar, ju] = -dx
        J[ar, n + iu] = dy
        J[ar, n + ju] = -dy
        J[ar, 2 * n + iu] = dr
        J[ar, 2 * n + ju] = dr
        J[m + k, k] = 1;              J[m + k, 2 * n + k] = -1
        J[m + n + k, k] = -1;         J[m + n + k, 2 * n + k] = -1
        J[m + 2 * n + k, n + k] = 1;  J[m + 2 * n + k, 2 * n + k] = -1
        J[m + 3 * n + k, n + k] = -1; J[m + 3 * n + k, 2 * n + k] = -1
        return J

    r0 = compute_max_radii(np.clip(c0, 0.0, 1.0))
    z0 = np.concatenate([c0[:, 0], c0[:, 1], np.maximum(r0, 0.01)])
    bounds = [(0, 1)] * (2 * n) + [(0, 0.5)] * n
    res = minimize(obj, z0, jac=obj_jac, method="SLSQP", bounds=bounds,
                   constraints=[{"type": "ineq", "fun": cons, "jac": cons_jac}],
                   options={"maxiter": 300, "ftol": 1e-10})
    z = res.x
    return np.column_stack([z[:n], z[n:2 * n]])


def construct_packing():
    """
    Multi-start SLSQP followed by basin-hopping around the best layout
    (fixed seed, time-capped); final radii come from the exact LP.

    Returns:
        Tuple of (centers, radii, sum_of_radii)
    """
    n = 26
    iu, ju = np.triu_indices(n, 1)
    rng = np.random.default_rng(7)
    base = _row_centers()
    best_c = np.clip(base, 0.02, 0.98)
    best_s = float(np.sum(compute_max_radii(best_c)))

    t0 = time.time()
    limit = 28.0
    trial = 0

    def attempt(c0):
        nonlocal best_s, best_c
        c0 = np.clip(c0, 0.03, 0.97)
        try:
            c = np.clip(_slsqp(c0, n, iu, ju), 0.0, 1.0)
            s = float(np.sum(compute_max_radii(c)))
        except Exception:
            return
        if s > best_s:
            best_s, best_c = s, c

    try:
        # Phase 1: diverse starts
        while time.time() - t0 < 10.0 and trial < 14:
            if trial % 3 == 2:
                c0 = rng.uniform(0.05, 0.95, size=(n, 2))
            else:
                c0 = base + rng.normal(0, 0.02 + 0.01 * (trial % 4), size=(n, 2))
            trial += 1
            attempt(c0)

        # Phase 2: basin-hopping around the incumbent
        it = 0
        while time.time() - t0 < limit:
            it += 1
            c0 = best_c.copy()
            if it % 3 == 0:
                r = compute_max_radii(c0)
                w = np.argsort(r)[: 3]
                c0[rng.choice(w)] = rng.uniform(0.05, 0.95, size=2)
                c0 += rng.normal(0, 0.005, size=c0.shape)
            else:
                c0 += rng.normal(0, 0.01 + 0.01 * (it % 3), size=c0.shape)
            attempt(c0)
    except Exception:
        pass

    radii = compute_max_radii(best_c)
    return best_c, radii, float(np.sum(radii))


def compute_max_radii(centers):
    """LP: maximize sum(r) s.t. r_i <= border dist, r_i + r_j <= dist(i,j)."""
    n = centers.shape[0]
    border = _border(centers)
    try:
        from scipy.optimize import linprog

        i, j = np.triu_indices(n, 1)
        d = np.linalg.norm(centers[i] - centers[j], axis=1)
        A = np.zeros((len(i), n))
        A[np.arange(len(i)), i] = 1.0
        A[np.arange(len(i)), j] = 1.0
        res = linprog(-np.ones(n), A_ub=A, b_ub=d,
                      bounds=[(0, max(b, 0)) for b in border],
                      method="highs")
        if res.success:
            return np.maximum(res.x - 1e-12, 0)
    except Exception:
        pass

    radii = np.maximum(border, 0)
    for a in range(n):
        for b in range(a + 1, n):
            dist = np.linalg.norm(centers[a] - centers[b])
            if radii[a] + radii[b] > dist:
                s = dist / (radii[a] + radii[b])
                radii[a] *= s
                radii[b] *= s
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