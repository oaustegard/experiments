# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles"""
import numpy as np


def _rows(counts):
    k = len(counts)
    ys = (np.arange(k) + 0.5) / k
    pts = []
    for idx, (c, y) in enumerate(zip(counts, ys)):
        xs = (np.arange(c) + 0.5) / c
        if c == max(counts):
            pass
        elif idx % 2:
            xs = 0.5 / max(counts) + (np.arange(c)) * (1 - 1 / max(counts)) / max(c - 1, 1)
        pts += [[x, y] for x in xs]
    return np.array(pts)


def _ring():
    # 16 circles around the border (4 corners + 3 per side), 10 inside
    pts = []
    e = 0.09
    t = np.linspace(e, 1 - e, 5)
    for x in t[:-1]:
        pts.append([x, e])
    for y in t[:-1]:
        pts.append([1 - e, y])
    for x in t[1:]:
        pts.append([x, 1 - e])
    for y in t[1:]:
        pts.append([e, y])
    pts = pts[:16]
    inner = [[0.3, 0.3], [0.5, 0.3], [0.7, 0.3],
             [0.2, 0.5], [0.4, 0.5], [0.6, 0.5], [0.8, 0.5],
             [0.3, 0.7], [0.5, 0.7], [0.7, 0.7]]
    return np.array(pts + inner)


def _slsqp(start, iu, ju, maxiter=300):
    from scipy.optimize import minimize
    n = len(start)
    m = len(iu)
    r0 = compute_max_radii(start) * 0.95
    ar = np.arange(m)
    eye = np.eye(n)

    def obj(v):
        return -np.sum(v[2 * n:])

    def gobj(v):
        g = np.zeros(3 * n)
        g[2 * n:] = -1
        return g

    def cons(v):
        x, y, r = v[:n], v[n:2 * n], v[2 * n:]
        d = np.hypot(x[iu] - x[ju], y[iu] - y[ju]) - r[iu] - r[ju]
        return np.concatenate([d, x - r, 1 - x - r, y - r, 1 - y - r])

    def jcons(v):
        x, y = v[:n], v[n:2 * n]
        dx = x[iu] - x[ju]
        dy = y[iu] - y[ju]
        dist = np.maximum(np.hypot(dx, dy), 1e-12)
        J = np.zeros((m + 4 * n, 3 * n))
        J[ar, iu] = dx / dist
        J[ar, ju] = -dx / dist
        J[ar, n + iu] = dy / dist
        J[ar, n + ju] = -dy / dist
        J[ar, 2 * n + iu] = -1
        J[ar, 2 * n + ju] = -1
        o = m
        J[o:o + n, :n] = eye
        J[o:o + n, 2 * n:] = -eye
        o += n
        J[o:o + n, :n] = -eye
        J[o:o + n, 2 * n:] = -eye
        o += n
        J[o:o + n, n:2 * n] = eye
        J[o:o + n, 2 * n:] = -eye
        o += n
        J[o:o + n, n:2 * n] = -eye
        J[o:o + n, 2 * n:] = -eye
        return J

    v0 = np.concatenate([start[:, 0], start[:, 1], r0])
    res = minimize(obj, v0, jac=gobj, method="SLSQP",
                   bounds=[(0, 1)] * (2 * n) + [(0, 0.5)] * n,
                   constraints=[{"type": "ineq", "fun": cons, "jac": jcons}],
                   options={"maxiter": maxiter, "ftol": 1e-12})
    return np.clip(np.stack([res.x[:n], res.x[n:2 * n]], axis=1), 0, 1)


def construct_packing():
    """
    Multi-start SLSQP from several structured layouts (staggered rows and
    border ring + interior), followed by basin hopping around the best
    solution. Radii are always re-solved by an exact LP so output is valid.

    Returns:
        Tuple of (centers, radii, sum_of_radii)
    """
    bases = [_rows([5, 5, 6, 5, 5]), _rows([5, 6, 5, 5, 5]),
             _rows([6, 5, 5, 5, 5]), _rows([4, 5, 4, 5, 4, 4]),
             _ring()]
    bases = [b for b in bases if len(b) == 26]
    best_c = bases[0]
    best_r = compute_max_radii(best_c)
    best_s = float(np.sum(best_r))

    try:
        import time
        n = 26
        iu, ju = np.triu_indices(n, 1)
        t0 = time.time()
        k = 0
        while time.time() - t0 < 30 and k < 30:
            rng = np.random.RandomState(k)
            base = bases[k % len(bases)]
            amp = 0.01 + 0.012 * (k % 4)
            start = np.clip(base + rng.uniform(-amp, amp, base.shape), 0.03, 0.97)
            k += 1
            try:
                cand = _slsqp(start, iu, ju)
            except Exception:
                continue
            r = compute_max_radii(cand)
            s = float(np.sum(r))
            if s > best_s:
                best_c, best_r, best_s = cand, r, s
        # basin hopping around the best
        h = 0
        while time.time() - t0 < 45 and h < 20:
            rng = np.random.RandomState(100 + h)
            start = np.clip(best_c + rng.normal(0, 0.01 + 0.01 * (h % 3), best_c.shape),
                            0.03, 0.97)
            h += 1
            try:
                cand = _slsqp(start, iu, ju, maxiter=400)
            except Exception:
                continue
            r = compute_max_radii(cand)
            s = float(np.sum(r))
            if s > best_s:
                best_c, best_r, best_s = cand, r, s
    except Exception:
        pass

    return best_c, best_r, best_s


def compute_max_radii(centers):
    """
    Maximize sum of radii subject to non-overlap and containment in the
    unit square (linear program); falls back to proportional scaling.
    """
    n = centers.shape[0]
    border = np.minimum(np.minimum(centers[:, 0], centers[:, 1]),
                        np.minimum(1 - centers[:, 0], 1 - centers[:, 1]))
    try:
        from scipy.optimize import linprog
        i, j = np.triu_indices(n, 1)
        d = np.linalg.norm(centers[i] - centers[j], axis=1)
        A = np.zeros((len(i), n))
        A[np.arange(len(i)), i] = 1
        A[np.arange(len(i)), j] = 1
        res = linprog(-np.ones(n), A_ub=A, b_ub=d,
                      bounds=[(0, max(b, 0)) for b in border], method="highs")
        if res.success:
            return np.maximum(res.x - 1e-12, 0)
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