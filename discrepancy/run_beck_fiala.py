"""Target B pipeline: exact D(t, n) values for small t via CEGAR + SAT.

For each t and n, scan k upward while cegar_exists(t, n, k) finds a witness;
the first UNSAT k proves D(t, n) = k - 1 exactly (m = t*n rows is
WLOG-complete: a degree-<=t system on n elements has at most t*n nonempty
sets counted with multiplicity, and duplicates/empties never change disc).

Every witness is double-certified: exhaustive enumeration + SAT UNSAT of the
(k-1)-bounded coloring.  Witness systems and certificates go to
bf_results.json.

Usage: python3 run_beck_fiala.py [T_LIST=2,3,4] [N_MAX=10] [CONF_BUDGET=2000000]
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np

from beck_fiala import (
    canonical_system,
    cegar_exists,
    disc_exact,
    max_degree,
    sat_disc_geq,
)

T_LIST = ([int(x) for x in sys.argv[1].split(",")]
          if len(sys.argv) > 1 else [2, 3, 4])
N_MAX = int(sys.argv[2]) if len(sys.argv) > 2 else 10
CONF_BUDGET = int(sys.argv[3]) if len(sys.argv) > 3 else 3_000_000


def main() -> None:
    results: list[dict] = []
    for t in T_LIST:
        print(f"=== t = {t} (Beck-Fiala cap 2t-2 = {2*t-2}) ===")
        best_seen = 0
        for n in range(3, N_MAX + 1):
            t0 = time.time()
            # start scanning at the best already found (monotone in n:
            # adding unused elements never lowers the max)
            k = max(best_seen, 1)
            witness = None
            exact = True
            while True:
                r = cegar_exists(t, n, k + 1, conf_budget=CONF_BUDGET)
                if r["exists"] is None:
                    exact = False  # budget out: D(t,n) >= k, k+1 undecided
                    break
                if not r["exists"]:
                    break
                k += 1
                witness = r["witness"]
            row: dict = {"t": t, "n": n, "D": k, "exact": exact,
                         "seconds": round(time.time() - t0, 1)}
            if witness is not None:
                d, _ = disc_exact(witness)
                assert d >= k and max_degree(witness) <= t
                assert sat_disc_geq(witness, k), "SAT cross-check failed"
                row["witness_sets"] = canonical_system(witness)
                row["witness_disc"] = d
            best_seen = max(best_seen, k)
            rel = "=" if exact else ">="
            print(f"  D({t},{n}) {rel} {k}   [{row['seconds']}s]"
                  + (f"  witness: {row.get('witness_sets')}"
                     if witness is not None else ""))
            results.append(row)
        print(f"  running max over n <= {N_MAX}: D({t}) >= {best_seen} "
              f"(upper bounds: {2*t-2} BF; {2*t-3} Bednarchak-Helm t>=3)\n")

    with open("bf_results.json", "w") as f:
        json.dump(results, f, indent=1)
    print("wrote bf_results.json")


if __name__ == "__main__":
    main()
