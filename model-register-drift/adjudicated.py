#!/usr/bin/env python3
"""Tabulate adjudication.json against the sample word counts.

The hand count is the score; score.py and structure.py are the shortlists that
fed it. Word counts come from the same body-prose extraction score.py uses, so
the two tables are comparable.
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

from score import prose_of

HERE = Path(__file__).parent
DATA = json.loads((HERE / "adjudication.json").read_text())
FAMILY = set(DATA["_family_aphorism_verdict"])


def main() -> int:
    rows = {}
    for stem, specimens in DATA["samples"].items():
        words = len(prose_of((HERE / "samples" / f"{stem}.md").read_text()).split())
        in_family = sum(1 for s in specimens if FAMILY & set(s["e"]))
        model, run = stem.rsplit("-", 1)
        rows.setdefault(model, []).append({
            "run": run, "words": words, "n": len(specimens),
            "per_1k": round(len(specimens) * 1000 / words, 1),
            "family_pct": round(100 * in_family / len(specimens)),
        })

    print(f"{'model':12} {'runs':5} {'per 1k':16} {'mean':6} {'spread':7} {'family':6}")
    print("-" * 60)
    order = sorted(rows.items(),
                   key=lambda kv: -st.mean(r["per_1k"] for r in kv[1]))
    for model, runs in order:
        vals = [r["per_1k"] for r in runs]
        fam = round(st.mean(r["family_pct"] for r in runs))
        spread = round(max(vals) - min(vals), 1) if len(vals) > 1 else None
        print(f"{model:12} {len(vals):<5} {', '.join(f'{v}' for v in vals):16} "
              f"{st.mean(vals):<6.1f} {str(spread) if spread is not None else '—':7} {fam}%")

    paired = [r for runs in rows.values() if len(runs) > 1 for r in runs]
    if len(paired) >= 4:
        spreads = [round(max(v) - min(v), 1) for v in
                   ([r["per_1k"] for r in runs] for runs in rows.values() if len(runs) > 1)]
        print(f"\nwithin-model spread, models with 2 runs: {spreads} "
              f"(mean {st.mean(spreads):.1f} per 1k)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
