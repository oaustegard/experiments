"""Problem definitions. Each problem is OpenEvolve's own example: its initial
program, its evaluator and its config's system message, unchanged.

score: circle packing is sum_radii / 2.635983 (best known). The two AlphaEvolve
problems use the evaluator's combined_score, which divides by AlphaEvolve's
published construction, as SwarmEvolve's Table 1 does.
"""

from pathlib import Path

import yaml

OE = Path("/tmp/src/openevolve/examples")
AE = OE / "alphaevolve_math_problems"

PROBLEMS = {
    "circle": {
        "src": OE / "circle_packing", "config": "config_phase_1.yaml", "jail_dir": "problem",
        "entry": "run_packing", "eval_timeout": 60, "has_stages": True,
        "interface": ("The program is a complete Python file. The evaluator imports it and calls `run_packing()`, "
                      "which must return `(centers, radii, sum_radii)`: centers a numpy array of shape (26, 2), "
                      "radii of shape (26,), all circles inside the unit square and none overlapping. An invalid "
                      "packing scores 0. numpy and scipy are available."),
        "score_label": "score (sum_radii / 2.635983)", "show": ("sum_radii", "validity", "eval_time"),
    },
    "heilbronn": {
        "src": AE / "heilbronn_triangle", "config": "config.yaml", "jail_dir": "problem_heilbronn",
        "entry": "heilbronn_triangle11", "eval_timeout": 360, "has_stages": False,
        "interface": ("The program is a complete Python file. The evaluator imports it and calls "
                      "`heilbronn_triangle11()`, which must return a numpy array of shape (11, 2): points on or "
                      "inside the triangle with vertices (0,0), (1,0), (0.5, sqrt(3)/2) (tolerance 1e-6). The "
                      "score is the smallest triangle area over all triples of points, as a fraction of the "
                      "triangle's area, divided by 0.036529889880030156. A point outside scores 0. numpy and "
                      "scipy are available."),
        "score_label": "score (min triangle area / AlphaEvolve's)", "show": ("min_area_normalized", "eval_time"),
    },
    "hexagon": {
        "src": AE / "hexagon_packing" / "11", "config": "config.yaml", "jail_dir": "problem_hexagon",
        "entry": "hexagon_packing_11", "eval_timeout": 360, "has_stages": False,
        "interface": ("The program is a complete Python file. The evaluator imports it and calls "
                      "`hexagon_packing_11()`, which must return `(inner_hex_data, outer_hex_data, "
                      "outer_hex_side_length)`: inner_hex_data a numpy array of shape (11, 3), one (x, y, "
                      "angle_degrees) row per unit hexagon; outer_hex_data (x, y, angle_degrees) of the outer "
                      "hexagon; and its side length. The inner hexagons must be disjoint and inside the outer "
                      "one, or the score is 0. The score is (1 / outer side length) divided by 1/3.930092. numpy "
                      "and scipy are available."),
        "score_label": "score (1/outer side / AlphaEvolve's)", "show": ("inv_outer_hex_side_length", "eval_time"),
    },
}


def get(name: str) -> dict:
    p = dict(PROBLEMS[name])
    p["name"] = name
    cfg = yaml.safe_load((p["src"] / p["config"]).read_text())
    p["system_message"] = cfg["prompt"]["system_message"]
    return p
