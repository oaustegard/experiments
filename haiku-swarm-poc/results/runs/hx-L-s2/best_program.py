# EVOLVE-BLOCK-START
import numpy as np

S3 = np.sqrt(3.0)


def _normals(ang_deg):
    # Edge normals of a unit hexagon with vertices at ang_deg + 60k degrees.
    a = np.deg2rad(ang_deg + 30.0 + 60.0 * np.arange(3))
    return np.stack([np.cos(a), np.sin(a)], axis=1)


def _verts(x, y, ang_deg):
    a = np.deg2rad(ang_deg + 60.0 * np.arange(6))
    return np.stack([x + np.cos(a), y + np.sin(a)], axis=1)


def _disjoint(h1, h2, tol=1e-9):
    # Separating-axis test: True if the two unit hexagons do not overlap.
    p1 = _verts(*h1)
    p2 = _verts(*h2)
    axes = np.vstack([_normals(h1[2]), _normals(h2[2])])
    for n in axes:
        a = p1 @ n
        b = p2 @ n
        if a.max() <= b.min() + tol or b.max() <= a.min() + tol:
            return True
    return False


def _all_disjoint(inner):
    k = len(inner)
    for i in range(k):
        for j in range(i + 1, k):
            if not _disjoint(inner[i], inner[j]):
                return False
    return True


def _required_side(inner):
    # Outer hexagon centered at origin with angle 0 (vertices at 60k degrees,
    # edge normals at 30+60k degrees, inradius R*sqrt(3)/2).
    nout = _normals(0.0)
    worst = 0.0
    for x, y, t in inner:
        v = _verts(x, y, t)
        worst = max(worst, (v @ nout.T).max())
    return worst / (S3 / 2.0)


def _lattice_start():
    # Honeycomb-compatible lattice of unit hexagons, all at angle 0.
    # Lattice spacing sqrt(3) (slightly enlarged) along normal directions keeps
    # neighbouring hexagons edge-to-edge without overlap.
    s = S3 * 1.0001
    e1 = s * np.array([np.cos(np.pi / 6), np.sin(np.pi / 6)])
    e2 = s * np.array([0.0, 1.0])
    pts = []
    for i in range(-5, 6):
        for j in range(-5, 6):
            c = i * e1 + j * e2
            pts.append(c)
    pts = np.array(pts)
    nout = _normals(0.0)
    key = (pts @ nout.T).max(axis=1) + 1e-6 * np.hypot(pts[:, 0], pts[:, 1])
    order = np.argsort(key)
    chosen = pts[order[:11]]
    return np.column_stack([chosen, np.zeros(11)])


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger
    regular hexagon, maximizing 1/outer_hex_side_length.

    Returns
        inner_hex_data: np.ndarray of shape (11,3), rows (x, y, angle_degrees).
        outer_hex_data: np.ndarray of shape (3,), (x, y, angle_degrees).
        outer_hex_side_length: float.
    """
    rng = np.random.default_rng(0)

    best = _lattice_start()
    best_R = _required_side(best)

    iters = 4000
    for it in range(iters):
        sigma = 0.25 * (0.999 ** it) + 0.005
        cand = best.copy()
        i = rng.integers(11)
        cand[i, 0:2] += rng.normal(0.0, sigma, 2)
        cand[i, 2] += rng.normal(0.0, 20.0 * sigma)

        ok = True
        for j in range(11):
            if j != i and not _disjoint(cand[i], cand[j]):
                ok = False
                break
        if not ok:
            continue

        R = _required_side(cand)
        if R < best_R - 1e-12:
            best = cand
            best_R = R

    outer_side = best_R + 1e-9
    inner_hex_data = np.array(best, dtype=float)
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    return inner_hex_data, outer_hex_data, outer_side

# EVOLVE-BLOCK-END
