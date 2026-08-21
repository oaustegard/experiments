#!/usr/bin/env python3
"""Convert the published NL2SH benchmark's test split into this repo's eval schema.

`westenfelder/NL2SH-ALFA` (arXiv:2502.06858) is the external set issue #52 names
as the target worth being measured against — 300 requests, each with **two**
acceptable gold commands, a difficulty label, and paths under `/testbed` chosen
so the commands actually run in a container. That last property is what the
cyber corpus lacks: with a fixture built from the gold commands, functional
equivalence decides most of these rows instead of most of them coming back
INCONCLUSIVE.

The output carries the same field names as `cyber_nl.json`, so `run_gen.py`
reads it with no changes. `names_utility` is computed here the way `gen_nl.py`
labels it — does the request contain the gold utility as a word — because these
requests were not written under an instruction to avoid it, and a leak-free
slice has to be identified rather than assumed.

    python3 alfa_prep.py --csv /path/to/test.csv --out data/alfa_test.json
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "nl2sh-retrieval"))

import pleias_gate as G  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=HERE / "data" / "alfa_test.json")
    a = ap.parse_args()

    rows = []
    for r in csv.DictReader(a.csv.open()):
        gold = (r["bash"] or "").strip()
        util = G.gold_utility(gold)
        if not util:
            continue
        nl = (r["nl"] or "").strip()
        rows.append({
            "tier": f"alfa-d{r.get('difficulty', '')}",
            "cmd": gold,
            "alt_cmd": (r.get("bash2") or "").strip(),
            "utility": util,
            "freq": 0,
            "nl": nl,
            "names_utility": bool(re.search(rf"\b{re.escape(util)}\b", nl)),
        })

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rows, indent=1) + "\n")
    leak = sum(r["names_utility"] for r in rows)
    print(f"{len(rows)} rows -> {a.out}  ({leak} name their utility, "
          f"{len(rows) - leak} leak-free)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
