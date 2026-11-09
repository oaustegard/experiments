# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles"""
import time
import numpy as np

N = 26


def _row_layout(counts):
    """Explicit staggered-row layout: counts[k] circles evenly spread in row k."""
    m = len(counts)
    pts = []
    for k, c in enumerate(counts):
        y = (k + 0.5) / m
        for x in (np.arange(c) + 0.5) / c:
            pts.append((x, y))
    return np.array(pts, dtype=float)


def _refine(c0, maxiter=200):
    """Jointly polish centers and radii with SLSQP (analytic Jacobians)."""
    from scipy.optimize import minimize

    n = c0.shape[0]
    I, J = np.triu_indices(n, 1)
    P = len(I)
    ar = np.arange(P)
    eye = np.eye(n)
    zero = np.zeros((n, n))

    z0 = np.concatenate([c0[:, 0], c0[:, 1], compute_max_radii(c0)])
    grad = np.concatenate([np.zeros(2 * n), -np.ones(n)])

    def cons(z):
        x, y, r = z[:n], z[n:2 * n], z[2 * n:]
        pair = (x[I] - x[J]) ** 2 + (y[I] - y[J]) ** 2 - (r[I] + r[J]) ** 2
        wall = np.concatenate([x - r, 1 - x - r, y - r, 1 - y - r])
        return np.concatenate([pair, wall])

    wall_jac = np.block([
        [eye, zero, -eye],
        [-eye, zero, -eye],
        [zero, eye, -eye],
        [zero, -eye, -eye],
    ])

    def jac(z):
        x, y, r = z[:n], z[n:2 * n], z[2 * n:]
        Jp = np.zeros((P, 3 * n))
        dx = 2 * (x[I] - x[J])
        dy = 2 * (y[I] - y[J])
        s = -2 * (r[I] + r[J])
        Jp[ar, I] = dx
        Jp[ar, J] = -dx
        Jp[ar, n + I] = dy
        Jp[ar, n + J] = -dy
        Jp[ar, 2 * n + I] = s
        Jp[ar, 2 * n + J] = s
        return np.vstack([Jp, wall_jac])

    res = minimize(
        lambda z: -np.sum(z[2 * n:]), z0, jac=lambda z: grad, method="SLSQP",
        bounds=[(0.0, 1.0)] * (2 * n) + [(0.0, 0.5)] * n,
        constraints=[{"type": "ineq", "fun": cons, "jac": jac}],
        options={"maxiter": maxiter, "ftol": 1e-12},
    )
    z = res.x
    return np.clip(np.stack([z[:n], z[n:2 * n]], axis=1), 0.0, 1.0)


def _score(c):
    r = compute_max_radii(c)
    return float(np.sum(r)), r


def construct_packing():
    """
    Construct explicit staggered-row arrangements of 26 circles, polish them
    with SLSQP, then run a short deterministic basin-hopping stage (perturb /
    relocate the smallest circle, re-polish). Final radii always come from an
    exact LP on the final centers, so the packing is valid.

    Returns:
        Tuple of (centers, radii, sum_of_radii)
    """
    t0 = time.perf_counter()
    budget = 22.0
    rng = np.random.RandomState(1)

    layouts = [
        [5, 5, 6, 5, 5],
        [5, 6, 5, 6, 4],
        [6, 5, 6, 5, 4],
        [4, 6, 5, 6, 5],
        [6, 5, 4, 5, 6],
        [5, 5, 5, 5, 5, 1],
    ]
    base = [_row_layout(l) for l in layouts]

    pool = []  # (score, centers, radii)
    for c in base:
        s, r = _score(c)
        pool.append((s, c, r))
        try:
            c2 = _refine(c)
            s2, r2 = _score(c2)
            pool.append((s2, c2, r2))
        except Exception:
            pass

    # Jittered restarts of the starting layouts
    for k in range(6):
        if time.perf_counter() - t0 > budget * 0.4:
            break
        for c in base[:4]:
            cj = np.clip(c + rng.normal(0, 0.02, c.shape), 0.03, 0.97)
            try:
                c2 = _refine(cj)
                s2, r2 = _score(c2)
                pool.append((s2, c2, r2))
            except Exception:
                pass

    pool.sort(key=lambda t: -t[0])
    best_s, best_c, best_r = pool[0]
    cur_s, cur_c = best_s, best_c

    # Basin hopping around the best solutions
    it = 0
    while time.perf_counter() - t0 < budget:
        it += 1
        # restart from one of the top pool entries now and then
        if it % 6 == 0:
            _, cur_c, _ = pool[rng.randint(min(4, len(pool)))]
            cur_s = _score(cur_c)[0]
        trial = cur_c.copy()
        mode = rng.randint(3)
        if mode == 0:
            trial += rng.normal(0, rng.uniform(0.005, 0.04), trial.shape)
        elif mode == 1:
            r = compute_max_radii(trial)
            k = int(np.argmin(r + 0.01 * rng.rand(N)))
            trial[k] = rng.uniform(0.05, 0.95, 2)
            trial += rng.normal(0, 0.005, trial.shape)
        else:
            r = compute_max_radii(trial)
            ks = np.argsort(r)[:4]
            trial[ks] += rng.normal(0, 0.05, (len(ks), 2))
        trial = np.clip(trial, 0.02, 0.98)
        try:
            c2 = _refine(trial, maxiter=150)
        except Exception:
            continue
        s2, r2 = _score(c2)
        if s2 > best_s:
            best_s, best_c, best_r = s2, c2, r2
        if s2 > cur_s - 1e-4:
            cur_s, cur_c = s2, c2

    centers = np.array(best_c, dtype=float)
    assert centers.shape == (N, 2)
    return centers, best_r, float(np.sum(best_r))


def compute_max_radii(centers):
    """
    Compute the radii maximizing their sum for fixed centers, subject to
    staying inside the unit square and not overlapping.

    Solved exactly as a linear program; falls back to a proportional
    scaling heuristic if scipy is unavailable.

    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates

    Returns:
        np.array of shape (n) with radius of each circle
    """
    n = centers.shape[0]
    wall = np.minimum(
        np.minimum(centers[:, 0], 1 - centers[:, 0]),
        np.minimum(centers[:, 1], 1 - centers[:, 1]),
    )
    wall = np.maximum(wall, 0.0)

    try:
        from scipy.optimize import linprog

        I, J = np.triu_indices(n, 1)
        m = len(I)
        A = np.zeros((m, n))
        A[np.arange(m), I] = 1.0
        A[np.arange(m), J] = 1.0
        b = np.linalg.norm(centers[I] - centers[J], axis=1)
        res = linprog(
            -np.ones(n),
            A_ub=A,
            b_ub=b,
            bounds=[(0, w) for w in wall],
            method="highs",
        )
        if res.success:
            return np.maximum(res.x - 1e-10, 0.0)
    except Exception:
        pass

    # Fallback heuristic
    radii = wall.copy()
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if radii[i] + radii[j] > dist and radii[i] + radii[j] > 0:
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