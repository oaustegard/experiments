"""Paired tests over results.json: sign test on per-issue recall differences
and a paired bootstrap 95% CI on the mean difference, n=59. Prints a markdown
table of every arm vs rg and every arm vs a chosen reference arm."""
from __future__ import annotations

import json
import sys
from math import comb

import numpy as np

from common import HERE

RNG = np.random.default_rng(0)


def sign_test(d: np.ndarray) -> tuple[int, int, float]:
    w, l = int((d > 0).sum()), int((d < 0).sum())
    n = w + l
    if n == 0:
        return w, l, 1.0
    k = min(w, l)
    return w, l, min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n)


def boot_ci(d: np.ndarray, reps: int = 20000) -> tuple[float, float]:
    idx = RNG.integers(0, len(d), size=(reps, len(d)))
    m = d[idx].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def vec(res: dict, arm: str, key: str) -> np.ndarray:
    rows = sorted(res[arm]["rows"], key=lambda r: r["issue"])
    return np.array([r[key] for r in rows])


def row(label: str, a: np.ndarray, b: np.ndarray) -> str:
    d = a - b
    w, l, p = sign_test(d)
    lo, hi = boot_ci(d)
    return f"| {label} | {d.mean():+.3f} | [{lo:+.3f}, {hi:+.3f}] | {w}/{l} | {p:.3f} |"


if __name__ == "__main__":
    res = json.load(open(HERE / "results.json"))
    ref = sys.argv[1] if len(sys.argv) > 1 else "potion-code-16M-v2"
    arms = [a for a in res if a != ref]
    print("| comparison | Δ | 95% CI | w/l | p |\n|---|---|---|---|---|")
    for k in ("r5", "r10"):
        for a in res:
            print(row(f"{a} dense vs rg ({k})", vec(res, a, f"dense_{k}"), vec(res, a, f"rg_{k}")))
        for a in res:
            print(row(f"{a} RRF vs rg ({k})", vec(res, a, f"rrf_{k}"), vec(res, a, f"rg_{k}")))
        for a in arms:
            print(row(f"{a} vs {ref} dense ({k})", vec(res, a, f"dense_{k}"), vec(res, ref, f"dense_{k}")))
