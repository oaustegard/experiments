"""Independent verification of every record claimed in the JSON outputs.

Re-derives each certificate from the stored data alone, through code paths
as disjoint as practical from the search pipelines:

* Komlós per-size records: rebuild the rational matrix from stored
  numerators/denominators, exactly re-check column norms <= 1, and
  recompute disc by DIRECT Fraction enumeration over all 2^{n-1} sign
  classes (no numpy screening, no shared engine code).
* Beck-Fiala witnesses: rebuild each witness system from its set list,
  re-check max degree <= t, and recompute disc by direct enumeration;
  additionally confirm disc >= D via the SAT UNSAT certificate.

Exits non-zero on any mismatch.
"""

from __future__ import annotations

import json
import sys
from fractions import Fraction
from itertools import product

import numpy as np

from beck_fiala import sat_disc_geq


def frac_disc_direct(V: list[list[Fraction]]) -> Fraction:
    """Pure-Fraction exhaustive discrepancy (slow, independent)."""
    m, n = len(V), len(V[0])
    best: Fraction | None = None
    for eps in product((1, -1), repeat=n - 1):
        e = (1,) + eps
        mx = Fraction(0)
        for i in range(m):
            v = sum(V[i][j] * e[j] for j in range(n))
            if v < 0:
                v = -v
            if v > mx:
                mx = v
        if best is None or mx < best:
            best = mx
    return best


def verify_komlos() -> int:
    with open("komlos_results.json") as f:
        R = json.load(f)
    failures = 0
    for entry in R["per_size"]:
        cert = entry.get("certified")
        if not cert:
            continue
        n = entry["n"]
        num, den = cert["matrix_num"], cert["matrix_den"]
        V = [[Fraction(num[i][j], den[i][j]) for j in range(len(num[0]))]
             for i in range(len(num))]
        norms_ok = all(
            sum(V[i][j] ** 2 for i in range(len(V))) <= 1
            for j in range(len(V[0])))
        d = frac_disc_direct(V)
        claimed = Fraction(cert["disc_exact"].replace("(", "").split("/")[0]
                           if "(" in cert["disc_exact"] else
                           cert["disc_exact"])
        ok = norms_ok and d == claimed
        print(f"komlos n={n}: norms_ok={norms_ok} disc={float(d):.6f} "
              f"claimed={float(claimed):.6f} {'OK' if ok else 'FAIL'}")
        failures += 0 if ok else 1
    return failures


def verify_bf() -> int:
    with open("bf_results.json") as f:
        rows = json.load(f)
    failures = 0
    for row in rows:
        sets = row.get("witness_sets")
        if not sets:
            continue
        t, n, D = row["t"], row["n"], row["D"]
        M = np.zeros((len(sets), n), dtype=np.int64)
        for i, s in enumerate(sets):
            for j in s.strip("{}").split(","):
                M[i, int(j)] = 1
        deg_ok = M.sum(axis=0).max() <= t
        # direct enumeration
        best = None
        for eps in product((1, -1), repeat=n - 1):
            e = np.array((1,) + eps)
            mx = int(np.abs(M @ e).max())
            best = mx if best is None else min(best, mx)
        sat_ok = sat_disc_geq(M, D)
        ok = deg_ok and best >= D and sat_ok
        print(f"bf t={t} n={n}: deg_ok={deg_ok} disc={best} claimed>={D} "
              f"sat_cert={sat_ok} {'OK' if ok else 'FAIL'}")
        failures += 0 if ok else 1
    return failures


if __name__ == "__main__":
    f = verify_komlos() + verify_bf()
    print("ALL CERTIFICATES VERIFIED" if f == 0 else f"{f} FAILURES")
    sys.exit(1 if f else 0)
