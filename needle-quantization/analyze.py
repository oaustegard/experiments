#!/usr/bin/env python3
"""Rank the bit-width arms against the shipped spec and test every contrast.

    python3 analyze.py        # tables + paired tests, writes analysis.json

Paired exact McNemar over the same 54 routable queries, the test the three
sibling experiments use, so a quantization cost is comparable to a schema-arm
difference or a pruning cut.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_quant import ARMS  # noqa: E402

BYTES_PER_PARAM = {"2": 2, "3": 3, "4": 4, "1.58": 2}  # ternary packs as 2-bit crumbs


def load(arm: str) -> dict:
    return json.loads((HERE / f"results_quant_{arm}.json").read_text())


def size_mb(arm: str) -> float:
    return json.loads((HERE / f"size_{arm}.json").read_text())["mb"]


def routable(res: dict) -> dict[str, bool]:
    return {r["id"]: r["tool_ok"] for r in res["rows"] if r["expected"]}


def mcnemar(a: dict[str, bool], b: dict[str, bool]) -> tuple[int, int, float]:
    ids = sorted(set(a) & set(b))
    a_only = sum(1 for i in ids if a[i] and not b[i])
    b_only = sum(1 for i in ids if b[i] and not a[i])
    n = a_only + b_only
    if n == 0:
        return a_only, b_only, 1.0
    k = min(a_only, b_only)
    return a_only, b_only, min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n))


def main() -> int:
    base = load("shipped")
    base_acc = base["summary"]["tool_acc_routable"]
    base_mb = size_mb("shipped")

    rows = []
    for arm, spec in ARMS.items():
        res = load(arm)
        acc = res["summary"]["tool_acc_routable"]
        ao, bo, p = mcnemar(routable(base), routable(res))
        rows.append({"arm": arm, "spec": spec, "mb": size_mb(arm), "routable": acc,
                     "delta": round(acc - base_acc, 4),
                     "mb_delta_pct": round(100 * (size_mb(arm) / base_mb - 1), 1),
                     "refusal": res["summary"]["refusal_acc"],
                     "args": res["summary"]["args_acc_routable"],
                     "shipped_only": ao, "arm_only": bo, "p": round(p, 4)})

    print(f"control `shipped` = {base_acc:.3f} routable at {base_mb:.2f} MB\n")
    print(f"{'arm':13} {'MB':>6} {'vs MB':>7} {'routable':>9} {'delta':>7} "
          f"{'ship-only':>10} {'arm-only':>9} {'p':>8}  spec")
    for r in sorted(rows, key=lambda r: -r["routable"]):
        star = " *" if r["p"] < 0.05 else ""
        print(f"{r['arm']:13} {r['mb']:6.2f} {r['mb_delta_pct']:+6.1f}% {r['routable']:9.3f} "
              f"{r['delta']:+7.3f} {r['shipped_only']:10} {r['arm_only']:9} {r['p']:8.4f}{star}  {r['spec']}")

    usable = [r for r in rows if "1.58" not in r["spec"] or r["arm"] == "engram-tern"]
    lo, hi = min(r["routable"] for r in usable), max(r["routable"] for r in usable)
    print(f"\nArms with no ternary in the bulk ({len(usable)} of {len(rows)}):")
    print(f"  routable spans {lo:.3f} to {hi:.3f} — {hi - lo:.3f}, "
          f"{(hi - lo) / (1 / 54):.1f} queries of 54")
    print(f"  bytes span {min(r['mb'] for r in usable):.2f} to "
          f"{max(r['mb'] for r in usable):.2f} MB")
    print(f"  significant against shipped: "
          f"{sum(1 for r in usable if r['p'] < 0.05)} of {len(usable)}")

    best_small = min((r for r in usable if r["routable"] >= base_acc), key=lambda r: r["mb"])
    print(f"\nSmallest arm that is not worse than shipped: {best_small['arm']} "
          f"({best_small['mb']:.2f} MB, {best_small['mb_delta_pct']:+.1f}%, "
          f"{best_small['routable']:.3f}, p={best_small['p']})")

    (HERE / "analysis.json").write_text(json.dumps(
        {"control": {"routable": base_acc, "mb": base_mb}, "arms": rows}, indent=1) + "\n")
    print("\nwrote analysis.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
