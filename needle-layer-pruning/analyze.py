#!/usr/bin/env python3
"""Rank the pruned arms against the unpruned control and test the survivors.

    python3 analyze.py            # tables + paired tests, writes analysis.json

Paired exact McNemar over the same 54 routable queries, the test the two sibling
experiments use, so a pruning cost is comparable to a schema-arm difference.
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENGRAM = (2, 15)


def load(label: str) -> dict:
    return json.loads((HERE / f"results_{label}.json").read_text())


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


def destroys_site(start: int, count: int) -> tuple:
    return tuple(l for l in ENGRAM if start <= l < start + count)


def arms() -> list[tuple[int, int, str]]:
    out = []
    for p in sorted(HERE.glob("results_cut*.json")):
        m = re.match(r"results_cut(\d+)_at(\d+)\.json", p.name)
        if m:
            out.append((int(m.group(1)), int(m.group(2)), p.stem[len("results_"):]))
    return sorted(out)


def main() -> int:
    control = load("control")
    base = control["summary"]["tool_acc_routable"]
    cbits = routable(control)
    print(f"control: {control['layers']} layers, routable {base:.3f}\n")

    rows = []
    for count, start, label in arms():
        res = load(label)
        acc = res["summary"]["tool_acc_routable"]
        lost = destroys_site(start, count)
        ao, bo, p = mcnemar(cbits, routable(res))
        rows.append({"count": count, "start": start, "label": label, "routable": acc,
                     "delta": round(acc - base, 4), "layers": res["layers"],
                     "engram_lost": list(lost),
                     "refusal": res["summary"]["refusal_acc"],
                     "control_only": ao, "arm_only": bo, "p": round(p, 4)})

    for count in sorted({r["count"] for r in rows}):
        print(f"== cut of {count} layers (27 -> {27 - count})\n")
        print(f"{'cut':<12} {'routable':>9} {'delta':>8} {'refusal':>8} {'engram lost':>12} "
              f"{'ctl-only':>9} {'arm-only':>9} {'p':>8}")
        for r in sorted((x for x in rows if x["count"] == count), key=lambda x: -x["routable"]):
            lost = ",".join(map(str, r["engram_lost"])) or "-"
            star = " *" if r["p"] < 0.05 else ""
            print(f"[{r['start']:2},{r['start'] + count:2})     {r['routable']:9.3f} {r['delta']:+8.3f} "
                  f"{r['refusal']:8.3f} {lost:>12} {r['control_only']:9} {r['arm_only']:9} "
                  f"{r['p']:8.4f}{star}")
        print()

    four = [r for r in rows if r["count"] == 4]
    if four:
        clean = [r for r in four if not r["engram_lost"]]
        print(f"4-layer cuts: {len(four)} positions, {len(clean)} of them leaving both Engram sites")
        print(f"  best overall      {max(four, key=lambda r: r['routable'])['label']} "
              f"{max(r['routable'] for r in four):.3f}")
        print(f"  best Engram-clean {max(clean, key=lambda r: r['routable'])['label']} "
              f"{max(r['routable'] for r in clean):.3f}")
        within = [r for r in four if r["delta"] >= -0.05]
        print(f"  within 0.05 of control: {len(within)} of {len(four)} "
              f"({', '.join(r['label'] for r in within) or 'none'})")
        dead = [r for r in four if r["routable"] <= 0.05]
        print(f"  effectively dead (<=0.05): {len(dead)} of {len(four)}")

    (HERE / "analysis.json").write_text(json.dumps(
        {"control_routable": base, "arms": rows}, indent=1))
    print("\nwrote analysis.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
