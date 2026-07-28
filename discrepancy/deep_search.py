"""Deep per-size search pass: many restarts, long annealing, fine
rationalization.  Updates komlos_results.json entries whenever a strictly
better certified value is found.  The 400-iter stage-2 defaults demonstrably
under-search (n=7 jumped 1.7257 -> 1.8302 under this schedule).

Usage: python3 deep_search.py [N_LO=3] [N_HI=16] [RESTARTS=24] [ITERS=3000]
"""

import json
import sys
import time

import numpy as np

from komlos import (
    column_norms_ok,
    exact_disc_fraction,
    project_columns,
    rationalize,
    softmaxmin_ascent,
)
from run_komlos import baseline_at, seed_matrices

N_LO = int(sys.argv[1]) if len(sys.argv) > 1 else 3
N_HI = int(sys.argv[2]) if len(sys.argv) > 2 else 16
RESTARTS = int(sys.argv[3]) if len(sys.argv) > 3 else 24
ITERS = int(sys.argv[4]) if len(sys.argv) > 4 else 3000


def main() -> None:
    with open("komlos_results.json") as f:
        R = json.load(f)
    for entry in R["per_size"]:
        n = entry["n"]
        if not N_LO <= n <= N_HI:
            continue
        t0 = time.time()
        rng = np.random.default_rng(9000 + n)
        prev = entry.get("certified", {}).get("disc_float",
                                              entry["search_best_float"])
        best_val, best_V = -1.0, None
        seeds = seed_matrices(n, rng)
        while len(seeds) < RESTARTS:
            seeds.append(project_columns(rng.standard_normal((n, n))))
        for si, V0 in enumerate(seeds):
            V, val = softmaxmin_ascent(V0, iters=ITERS, lr=0.015,
                                       seed=1000 * n + si,
                                       beta_start=6.0, beta_end=150.0)
            if val > best_val:
                best_val, best_V = val, V
        improved = False
        if best_val > prev + 1e-9 and best_V is not None:
            for md in (256, 1024, 4096):
                VF = rationalize(best_V, max_den=md)
                assert column_norms_ok(VF)
                c = exact_disc_fraction(VF)
                if c["disc_float"] > entry.get("certified", {}).get(
                        "disc_float", -1):
                    entry["search_best_float"] = best_val
                    entry["certified"] = {
                        "disc_exact": c["disc_exact_str"],
                        "disc_float": c["disc_float"],
                        "max_denominator": md,
                        "matrix_num": [[x.numerator for x in row]
                                       for row in VF],
                        "matrix_den": [[x.denominator for x in row]
                                       for row in VF],
                    }
                    improved = True
        cur = entry.get("certified", {}).get("disc_float", prev)
        print(f"n={n:2d}: float {best_val:.6f}  certified {cur:.6f}  "
              f"baseline {baseline_at(n):.6f}  "
              f"{'IMPROVED' if improved else '-'}  "
              f"[{time.time()-t0:.0f}s]", flush=True)
        with open("komlos_results.json", "w") as f:
            json.dump(R, f, indent=1)
    print("done")


if __name__ == "__main__":
    main()
