"""Q4: reproduce the paper's 0.008 ms LAPJV figure at N=50 and measure scaling.

Table 5 of arXiv:2609.01807 reports 0.008 ms for the LAPJV assignment at
N=50 on CPU, and argues the O(N^3) solver is 'effectively free'.
scipy.optimize.linear_sum_assignment is a Jonker-Volgenant shortest-augmenting-
path implementation, the same family.
"""
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
SEED = 20260908


def bench(N, K, kind, reps=200, rng=None):
    rng = rng or np.random.default_rng(SEED)
    Ms = []
    for _ in range(reps):
        if kind == "random":
            Ms.append(rng.normal(size=(N, K)))
        elif kind == "rank1":
            a = rng.normal(size=N)
            b = np.sort(rng.uniform(0.5, 2.0, size=K))[::-1]
            Ms.append(np.outer(a, b))
        elif kind == "banded":
            base = -np.abs(np.subtract.outer(np.arange(N), np.arange(K))).astype(float)
            Ms.append(base + 4.0 * rng.normal(size=(N, K)))
    for M in Ms[:5]:
        linear_sum_assignment(M, maximize=True)  # warm
    t0 = time.perf_counter()
    for M in Ms:
        linear_sum_assignment(M, maximize=True)
    return (time.perf_counter() - t0) / reps * 1e3  # ms


def main():
    rows = []
    print(f"{'N':>6} {'K':>6} {'kind':>8} {'ms':>10} {'us':>10}")
    for kind in ["random", "rank1", "banded"]:
        for N in [10, 25, 50, 100, 150, 250, 500, 1000, 2000]:
            reps = 200 if N <= 250 else (40 if N <= 1000 else 10)
            ms = bench(N, N, kind, reps)
            rows.append({"N": N, "K": N, "kind": kind, "ms": ms})
            print(f"{N:6d} {N:6d} {kind:>8} {ms:10.4f} {ms*1000:10.1f}")
    print()
    for K in [1, 5, 10, 25, 50]:
        ms = bench(50, K, "banded", 200)
        rows.append({"N": 50, "K": K, "kind": "banded", "ms": ms})
        print(f"{50:6d} {K:6d} {'banded':>8} {ms:10.4f} {ms*1000:10.1f}")

    # empirical scaling exponent on random, N in [50, 1000]
    sub = [r for r in rows if r["kind"] == "random" and 50 <= r["N"] <= 1000]
    x = np.log([r["N"] for r in sub]); y = np.log([r["ms"] for r in sub])
    slope = float(np.polyfit(x, y, 1)[0])
    print(f"\nempirical exponent on random N=50..1000: {slope:.2f}  (paper cites O(N^3), LAPJV O(N^2) average)")
    (HERE / "out").mkdir(exist_ok=True)
    with open(HERE / "out" / "solver_bench.json", "w") as fh:
        json.dump({"rows": rows, "exponent_random_50_1000": slope}, fh, indent=2)


if __name__ == "__main__":
    main()
