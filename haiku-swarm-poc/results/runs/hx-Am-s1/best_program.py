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
    # Tight honeycomb lattice (flat-top, same orientation as outer hexagon).
    s = 1 + 1e-6  # minimal gap so touching edges do not count as overlap
    h = np.sqrt(3) / 2
    pts = [
        (0, 0),
        (0, 2 * h), (0, -2 * h),
        (1.5, h), (1.5, -h), (-1.5, h), (-1.5, -h),
        (3, 0), (-3, 0),
        (1.5, 3 * h), (-1.5, -3 * h),
    ]
    inner_hex_data = np.array([[s * x, s * y, 0.0] for x, y in pts])

    outer_hex_data = np.array([0, 0, 0])  # centered at origin
    outer_hex_side_length = 1 + 3 * s  # ≈ 4.000003

    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
