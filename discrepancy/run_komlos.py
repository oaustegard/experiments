"""Target A pipeline: per-size Komlós lower-bound records K(n).

Stages:
  1. Verify Delta = delta on the Kunisky family k = 1..4 (the exact
     discrepancy of A-hat^{T_k} equals its delta lower bound — provable in
     one line, verified here computationally as a calibration).
  2. Per-size search, n = 2..N_MAX: multi-seed smoothed max-min ascent over
     normalized matrices (float screening).
  3. Rationalize + exactly certify (Fraction arithmetic, exhaustive
     enumeration) every candidate that beats the Kunisky-family baseline
     at its size.

Outputs: komlos_results.json + printed table.

Usage: python3 run_komlos.py [N_MAX=16] [ITERS=400] [SEEDS=6]
"""

from __future__ import annotations

import json
import sys
import time
from fractions import Fraction

import numpy as np

from komlos import (
    exact_disc_fraction,
    exact_disc_sqrt2,
    column_norms_ok,
    float_disc,
    kunisky_delta_lower_bound,
    kunisky_normalized_int_pair,
    project_columns,
    rationalize,
    softmaxmin_ascent,
)

N_MAX = int(sys.argv[1]) if len(sys.argv) > 1 else 16
ITERS = int(sys.argv[2]) if len(sys.argv) > 2 else 400
SEEDS = int(sys.argv[3]) if len(sys.argv) > 3 else 6


def kunisky_float_matrix(k: int) -> np.ndarray:
    A, B, scale = kunisky_normalized_int_pair(k)
    return (A + 2 ** 0.5 * B) / scale


def baseline_at(n: int) -> float:
    """Best Kunisky-family value achievable at size n (pad a smaller 2^k
    instance with zero columns/rows: disc unchanged, still normalized)."""
    k = 0
    while 2 ** (k + 1) <= n:
        k += 1
    return kunisky_delta_lower_bound(k)


def hadamard(n: int) -> np.ndarray | None:
    if n & (n - 1) == 0 and n >= 1:
        H = np.array([[1.0]])
        while H.shape[0] < n:
            H = np.block([[H, H], [H, -H]])
        return H / n ** 0.5
    return None


def seed_matrices(n: int, rng: np.random.Generator) -> list[np.ndarray]:
    seeds = []
    # padded Kunisky
    k = 0
    while 2 ** (k + 1) <= n:
        k += 1
    K = kunisky_float_matrix(k)
    P = np.zeros((n, n))
    P[: K.shape[0], : K.shape[1]] = K
    seeds.append(P)
    H = hadamard(n)
    if H is not None:
        seeds.append(H)
    seeds.append(np.eye(n))
    while len(seeds) < SEEDS:
        seeds.append(project_columns(rng.standard_normal((n, n))))
    return seeds[:SEEDS]


def main() -> None:
    results: dict = {"kunisky_family": [], "per_size": []}

    print("=== Stage 1: Kunisky family, exact Delta vs delta ===")
    for k in range(1, 5):
        A, B, scale = kunisky_normalized_int_pair(k)
        t0 = time.time()
        r = exact_disc_sqrt2(A, B, scale)
        delta = kunisky_delta_lower_bound(k)
        gap = r["disc_float"] - delta
        assert abs(gap) < 1e-9, f"Delta != delta at k={k}?! gap={gap}"
        print(f"k={k} (n={2**k:3d}): Delta = {r['disc_float']:.6f} "
              f"= delta = {delta:.6f}  exact {r['disc_exact_str']} "
              f"[{time.time()-t0:.1f}s, {r['n_sign_vectors_checked']} eps]")
        results["kunisky_family"].append(
            {"k": k, "n": 2 ** k, "delta": delta,
             "Delta_exact": r["disc_exact_str"],
             "Delta_float": r["disc_float"]})
    print("Delta = delta on the whole family: the all-ones coloring achieves")
    print("max = delta (every row l1-norm equals delta), and Prop 2.3 forces")
    print(">= delta.  Kunisky's finite lower bounds are already exact.\n")

    print(f"=== Stage 2+3: per-size search, n = 2..{N_MAX} ===")
    rng = np.random.default_rng(2026)
    for n in range(2, N_MAX + 1):
        base = baseline_at(n)
        t0 = time.time()
        best_val, best_V = -1.0, None
        for si, V0 in enumerate(seed_matrices(n, rng)):
            V, val = softmaxmin_ascent(V0, iters=ITERS, seed=si)
            if val > best_val:
                best_val, best_V = val, V
        entry = {"n": n, "kunisky_baseline": base,
                 "search_best_float": best_val,
                 "search_seconds": round(time.time() - t0, 1)}
        line = (f"n={n:2d}: baseline {base:.6f}  search {best_val:.6f}  "
                f"[{entry['search_seconds']}s]")
        if best_val > base + 1e-4:
            VF = rationalize(best_V, max_den=64)
            assert column_norms_ok(VF)
            cert = exact_disc_fraction(VF)
            entry["certified"] = {
                "disc_exact": cert["disc_exact_str"],
                "disc_float": cert["disc_float"],
                "matrix_num": [[f.numerator for f in row] for row in VF],
                "matrix_den": [[f.denominator for f in row] for row in VF],
            }
            beat = cert["disc_float"] > base + 1e-9
            line += (f"  -> certified {cert['disc_float']:.6f} "
                     f"({'BEATS baseline' if beat else 'does not survive'})")
        print(line)
        results["per_size"].append(entry)

    with open("komlos_results.json", "w") as f:
        json.dump(results, f, indent=1)
    print("\nwrote komlos_results.json")


if __name__ == "__main__":
    main()
