#!/usr/bin/env python3
"""Score the naming variants against PREREG.md and test the contrasts.

    python3 analyze.py            # tables + paired tests, writes analysis.json

Paired exact McNemar over the same 54 routable queries, same test needle-bsky
used, so the numbers are directly comparable to that writeup's contrast table.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from names import VARIANTS  # noqa: E402

MODES = ("flat", "oracle")


def load(variant: str, mode: str) -> dict:
    return json.loads((HERE / f"results_{variant}_{mode}.json").read_text())


def routable(res: dict) -> dict[str, bool]:
    return {r["id"]: r["tool_ok"] for r in res["rows"] if r["expected"]}


def mcnemar(a: dict[str, bool], b: dict[str, bool]) -> tuple[int, int, float]:
    """Exact two-sided McNemar. b_only/a_only are the discordant pairs."""
    ids = sorted(set(a) & set(b))
    a_only = sum(1 for i in ids if a[i] and not b[i])
    b_only = sum(1 for i in ids if b[i] and not a[i])
    n = a_only + b_only
    if n == 0:
        return a_only, b_only, 1.0
    k = min(a_only, b_only)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return a_only, b_only, min(1.0, 2 * tail)


def per_category(res: dict) -> dict[str, float]:
    cats: dict[str, list[bool]] = {}
    for r in res["rows"]:
        cats.setdefault(r["cat"], []).append(r["tool_ok"])
    return {k: round(sum(v) / len(v), 3) for k, v in sorted(cats.items())}


def main() -> int:
    results = {(v, m): load(v, m) for v in VARIANTS for m in MODES}

    print("== routable top-1 (54 queries), by variant and catalogue size\n")
    print(f"{'variant':13} {'flat 18':>9} {'oracle 5':>9} {'retrieval cost':>15}")
    for v in VARIANTS:
        f = results[(v, "flat")]["summary"]["tool_acc_routable"]
        o = results[(v, "oracle")]["summary"]["tool_acc_routable"]
        print(f"{v:13} {f:9.3f} {o:9.3f} {o - f:15.3f}")

    print("\n== paired exact McNemar, routable queries only\n")
    contrasts = [
        ("names-only", "desc-only", "flat", "H1: names vs descriptions, 18 declared"),
        ("names-only", "desc-only", "oracle", "H1: names vs descriptions, 5 declared"),
        ("canon", "names-only", "flat", "cost of deleting every description"),
        ("canon", "desc-only", "flat", "cost of deleting every name"),
        ("canon", "adversarial", "flat", "cost of rotating names onto neighbours"),
        ("canon", "separated", "flat", "gain from rule-written names"),
        ("canon", "separated", "oracle", "gain from rule-written names, 5 declared"),
        ("canon", "neither", "flat", "both channels removed"),
    ]
    print(f"{'contrast':52} {'mode':7} {'A':>3} {'B':>3} {'p':>8}")
    tests = []
    for a, b, mode, label in contrasts:
        ao, bo, p = mcnemar(routable(results[(a, mode)]), routable(results[(b, mode)]))
        star = " *" if p < 0.05 else ""
        print(f"{label + f'  ({a} vs {b})':52} {mode:7} {ao:3} {bo:3} {p:8.4f}{star}")
        tests.append({"a": a, "b": b, "mode": mode, "label": label,
                      "a_only": ao, "b_only": bo, "p": round(p, 4)})

    print("\n== per-category top-1, flat 18\n")
    cats = sorted({r["cat"] for r in results[("canon", "flat")]["rows"]})
    print(f"{'category':14} " + " ".join(f"{v[:9]:>10}" for v in VARIANTS))
    tables = {v: per_category(results[(v, "flat")]) for v in VARIANTS}
    for c in cats:
        print(f"{c:14} " + " ".join(f"{tables[v].get(c, float('nan')):10.3f}" for v in VARIANTS))

    print("\n== per-category top-1, oracle 5\n")
    otab = {v: per_category(results[(v, "oracle")]) for v in VARIANTS}
    print(f"{'category':14} " + " ".join(f"{v[:9]:>10}" for v in VARIANTS))
    for c in cats:
        print(f"{c:14} " + " ".join(f"{otab[v].get(c, float('nan')):10.3f}" for v in VARIANTS))

    out = {
        "routable": {f"{v}_{m}": results[(v, m)]["summary"]["tool_acc_routable"]
                     for v in VARIANTS for m in MODES},
        "args": {f"{v}_{m}": results[(v, m)]["summary"]["args_acc_routable"]
                 for v in VARIANTS for m in MODES},
        "refusal": {f"{v}_{m}": results[(v, m)]["summary"]["refusal_acc"]
                    for v in VARIANTS for m in MODES},
        "median_latency_ms": {f"{v}_{m}": results[(v, m)]["summary"]["median_latency_ms"]
                              for v in VARIANTS for m in MODES},
        "mcnemar": tests,
        "per_category_flat": tables,
        "per_category_oracle": otab,
    }
    (HERE / "analysis.json").write_text(json.dumps(out, indent=1))
    print("\nwrote analysis.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
