#!/usr/bin/env python3
"""Paired comparisons and per-category tables over the four results_*.json.

Every arm sees the same 62 queries, so each contrast is paired: McNemar's exact
test on the discordant pairs, which is the right test here and does not assume
the arms are independent samples.

    python3 analyze.py            # tables to stdout
    python3 analyze.py --json     # machine-readable, used by recheck.py
"""

from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARMS = ["auto", "auto-min", "tuned", "tuned-min"]


def load(arm: str) -> dict:
    return json.loads((HERE / f"results_{arm}.json").read_text())


def mcnemar_exact(a_rows: list[dict], b_rows: list[dict], key: str = "tool_ok") -> dict:
    """Two-sided exact McNemar. a and b are same-order row lists."""
    b_only = sum(1 for x, y in zip(a_rows, b_rows) if not x[key] and y[key])
    a_only = sum(1 for x, y in zip(a_rows, b_rows) if x[key] and not y[key])
    n = a_only + b_only
    if n == 0:
        return {"a_only": 0, "b_only": 0, "n_discordant": 0, "p": 1.0}
    k = min(a_only, b_only)
    p = min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) / (2**n))
    return {"a_only": a_only, "b_only": b_only, "n_discordant": n, "p": round(p, 5)}


def per_category(rows: list[dict]) -> dict:
    cats: dict[str, list] = {}
    for r in rows:
        cats.setdefault(r["cat"], []).append(r)
    return {
        c: {"n": len(rs), "tool": round(sum(x["tool_ok"] for x in rs) / len(rs), 3)}
        for c, rs in sorted(cats.items())
    }


def confusion(rows: list[dict]) -> list[tuple]:
    out = []
    for r in rows:
        if not r["tool_ok"]:
            out.append((r["id"], r["query"][:52], r["expected"], r["got"], r["confidence"]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    R = {arm: load(arm) for arm in ARMS}
    rows = {arm: R[arm]["rows"] for arm in ARMS}

    contrasts = [
        ("auto", "tuned", "description authorship, full arity"),
        ("auto-min", "tuned-min", "description authorship, minimal arity"),
        ("auto", "auto-min", "arity reduction, auto descriptions"),
        ("tuned", "tuned-min", "arity reduction, tuned descriptions"),
    ]
    out = {
        "summary": {arm: R[arm]["summary"] for arm in ARMS},
        "contrasts": [],
        "per_category": {arm: per_category(rows[arm]) for arm in ARMS},
        "gate_sweep": {arm: R[arm]["gate_sweep"] for arm in ARMS},
        "deterministic": {arm: R[arm]["deterministic"] for arm in ARMS},
    }
    for x, y, label in contrasts:
        m = mcnemar_exact(rows[x], rows[y], "tool_ok")
        m.update({"a": x, "b": y, "label": label, "metric": "tool_ok"})
        out["contrasts"].append(m)

    if a.json:
        print(json.dumps(out, indent=1))
        return 0

    print("arm         tools  tool_acc  routable  refuse  args   invented  medms")
    for arm in ARMS:
        s = R[arm]["summary"]
        print(
            f"{arm:11} {R[arm]['n_tools']:5d}  {s['tool_acc']:.3f}     {s['tool_acc_routable']:.3f}     "
            f"{s['refusal_acc']:.3f}   {s['args_acc_routable']:.3f}  {s['invented_rate']:.3f}     "
            f"{s['median_latency_ms']:.0f}"
        )

    print("\nPaired McNemar on tool_ok (n=62):")
    for c in out["contrasts"]:
        print(
            f"  {c['a']:9} vs {c['b']:10} {c['label']:38} "
            f"{c['a']}-only {c['a_only']:2d} / {c['b']}-only {c['b_only']:2d}  p={c['p']}"
        )

    print("\nPer-category tool accuracy:")
    cats = sorted({c for arm in ARMS for c in out["per_category"][arm]})
    print(f"  {'category':13} {'n':>2}  " + "  ".join(f"{a:>9}" for a in ARMS))
    for c in cats:
        n = out["per_category"][ARMS[0]][c]["n"]
        cells = "  ".join(f"{out['per_category'][a].get(c, {}).get('tool', float('nan')):9.3f}" for a in ARMS)
        print(f"  {c:13} {n:2d}  {cells}")

    print("\nErrors, tuned-min:")
    for e in confusion(rows["tuned-min"]):
        print(f"  {e[0]:11} {e[1]:54} want {e[2]} got {e[3]} conf {e[4]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
