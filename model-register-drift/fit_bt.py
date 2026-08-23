#!/usr/bin/env python3
"""Bradley-Terry fit over the blind pairwise verdicts.

Each verdict says which of two samples reads more like an LLM wrote it. The
fitted strength is on a logit scale: higher means judged more machine-written
more often. Fitted by MM iteration with a light prior so an undefeated sample
does not run to infinity.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
verdicts = json.loads((HERE / "verdicts.json").read_text())
key = json.loads((HERE / "blind_key.json").read_text())

labels = sorted({x for v in verdicts for x in v[:2]})
wins = defaultdict(float)
games = defaultdict(list)
for a, b, w in verdicts:
    wins[w] += 1
    games[a].append(b)
    games[b].append(a)

# MM algorithm with a 0.5-win prior against a virtual average opponent.
p = {l: 1.0 for l in labels}
for _ in range(2000):
    new = {}
    for l in labels:
        denom = sum(1.0 / (p[l] + p[o]) for o in games[l]) + 1.0 / (p[l] + 1.0)
        new[l] = (wins[l] + 0.5) / denom
    norm = math.exp(sum(math.log(v) for v in new.values()) / len(new))
    p = {l: v / norm for l, v in new.items()}

rows = []
for l in labels:
    model, run = key[l].rsplit("-", 1)
    rows.append({
        "label": l, "sample": key[l], "model": model,
        "wins": int(wins[l]), "n": len(games[l]),
        "strength": round(math.log(p[l]), 2),
    })
rows.sort(key=lambda r: -r["strength"])

print(f"{'sample':14} {'model':12} {'W/N':6} {'strength':8}")
print("-" * 46)
for r in rows:
    print(f"{r['sample']:14} {r['model']:12} {r['wins']}/{r['n']:<4} {r['strength']:+.2f}")

print("\nby model (mean strength over its samples)")
agg = defaultdict(list)
for r in rows:
    agg[r["model"]].append(r["strength"])
for m, vs in sorted(agg.items(), key=lambda kv: -sum(kv[1]) / len(kv[1])):
    mean = sum(vs) / len(vs)
    detail = ", ".join(f"{v:+.2f}" for v in vs)
    print(f"  {m:12} {mean:+.2f}   ({detail})")

Path(HERE / "bt_fit.json").write_text(json.dumps(rows, indent=2) + "\n")
