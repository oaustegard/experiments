"""Heavier polish of the small-n Komlós records (n = 3..7): more seeds,
longer annealed ascent, finer rationalization.  Updates komlos_results.json
in place (per_size entries) when a strictly better certified value is found.
"""

import json

import numpy as np

from komlos import (
    column_norms_ok,
    exact_disc_fraction,
    project_columns,
    rationalize,
    softmaxmin_ascent,
)
from run_komlos import baseline_at, seed_matrices

ITERS = 2500
RESTARTS = 14


def main() -> None:
    with open("komlos_results.json") as f:
        results = json.load(f)
    rng = np.random.default_rng(777)
    for entry in results["per_size"]:
        n = entry["n"]
        if not 3 <= n <= 7:
            continue
        best_val = entry["search_best_float"]
        best_V = None
        seeds = seed_matrices(n, rng)
        while len(seeds) < RESTARTS:
            seeds.append(project_columns(rng.standard_normal((n, n))))
        for si, V0 in enumerate(seeds):
            V, val = softmaxmin_ascent(V0, iters=ITERS, lr=0.015, seed=100 + si,
                                       beta_start=6.0, beta_end=120.0)
            if val > best_val:
                best_val, best_V = val, V
        print(f"n={n}: polished float {best_val:.6f} "
              f"(was {entry['search_best_float']:.6f})")
        if best_V is not None:
            for md in (64, 256, 1024):
                VF = rationalize(best_V, max_den=md)
                assert column_norms_ok(VF)
                cert = exact_disc_fraction(VF)
                old = entry.get("certified", {}).get("disc_float", -1)
                if cert["disc_float"] > old:
                    entry["search_best_float"] = best_val
                    entry["certified"] = {
                        "disc_exact": cert["disc_exact_str"],
                        "disc_float": cert["disc_float"],
                        "max_denominator": md,
                        "matrix_num": [[f.numerator for f in row] for row in VF],
                        "matrix_den": [[f.denominator for f in row] for row in VF],
                    }
            print(f"      certified {entry['certified']['disc_float']:.6f} "
                  f"= {entry['certified']['disc_exact']} "
                  f"(baseline {baseline_at(n):.6f})")
    with open("komlos_results.json", "w") as f:
        json.dump(results, f, indent=1)
    print("updated komlos_results.json")


if __name__ == "__main__":
    main()
