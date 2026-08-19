#!/usr/bin/env python3
"""Paired McNemar between arms on the same queries. Exact binomial, two-sided."""

from __future__ import annotations

import json
from math import comb
from pathlib import Path

from handwritten import HandRouter
from router import Router

HERE = Path(__file__).resolve().parent


def exact_p(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def arm(name: str):
    return HandRouter() if name == "hand" else Router(HERE / f"rules_{name}.json")


def main() -> int:
    splits = {
        "family B (held-out)": HERE / "data" / "family_b.jsonl",
        "wild (hand-authored)": HERE / "wild.jsonl",
    }
    contrasts = [("hand", "schema"), ("hand", "overlap"), ("hand", "laplace8"),
                 ("schema", "overlap")]
    out = {}
    hdr = f"{'split':<22}{'contrast':<24}{'A only':>8}{'B only':>8}{'both':>7}{'p':>10}"
    print(hdr)
    print("-" * len(hdr))
    for sname, path in splits.items():
        rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
        on = [r for r in rows if r.get("label")]
        for a_name, b_name in contrasts:
            ra, rb = arm(a_name), arm(b_name)
            b = c = both = 0
            for r in on:
                oa = ra.route(r["query"]) == r["label"]
                ob = rb.route(r["query"]) == r["label"]
                both += oa and ob
                b += oa and not ob
                c += ob and not oa
            p = exact_p(b, c)
            tag = f"{a_name} vs {b_name}"
            ps = f"{p:.2e}" if p < 1e-4 else f"{p:.4f}"
            print(f"{sname:<22}{tag:<24}{b:>8}{c:>8}{both:>7}{ps:>10}")
            out.setdefault(sname, {})[tag] = {"a_only": b, "b_only": c, "both": both,
                                              "p": p, "n": len(on)}
    (HERE / "results_mcnemar.json").write_text(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
