# EVOLVE-BLOCK-START
import numpy as np


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    n = 11
    s3 = np.sqrt(3)
    # Tight honeycomb lattice (same orientation as outer hexagon)
    pts = np.array([[1.5 * i, s3 * (j + i / 2)]
                    for i in range(-6, 7) for j in range(-9, 10)])
    # Outer hexagon edge normals
    ang = np.radians(30 + 60 * np.arange(6))
    normals = np.stack([np.cos(ang), np.sin(ang)], axis=1)

    best = None
    # candidate outer centers: lattice cell center, vertex, edge midpoints
    for off in ([0, 0], [1.0, 0], [0.75, s3 / 4], [0.5, 0], [0, s3 / 2]):
        c = (pts - np.array(off)) * (1 + 1e-6)  # minimal gap
        proj = c @ normals.T  # (N, 6)
        score = proj.max(axis=1)
        idx = np.argsort(score)[:n]
        apothem = proj[idx].max() + s3 / 2
        R = apothem * 2 / s3
        if best is None or R < best[0]:
            best = (R, c[idx])

    R, centers = best
    inner_hex_data = np.column_stack([centers, np.zeros(n)])

    outer_hex_data = np.array([0, 0, 0])  # centered at origin
    outer_hex_side_length = float(R) + 1e-7  # tighter tolerance (lattice already has 1e-6 relative gap)

    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
