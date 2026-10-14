#!/usr/bin/env python3
"""Precision/recall of each nearness predictor against replayed conflicts, per window.

    python3 score.py replay.json
"""
import json
import sys

rows = json.load(open(sys.argv[1]))
preds = {
    "file": lambda r: r["file"],
    "hunk:3": lambda r: r["gap"] is not None and r["gap"] <= 3,
    "hunk:10": lambda r: r["gap"] is not None and r["gap"] <= 10,
    "hunk:30": lambda r: r["gap"] is not None and r["gap"] <= 30,
    "func": lambda r: r["func"],
    "func|hunk:3": lambda r: r["func"] or (r["gap"] is not None and r["gap"] <= 3),
}
print(f"{'w':>2} {'n':>4} {'confl':>5} | " + " | ".join(f"{k:>18}" for k in preds))
print(f"{'':>2} {'':>4} {'':>5} | " + " | ".join(f"{'flag% P R':>18}" for _ in preds))
for w in sorted({r["w"] for r in rows}):
    rs = [r for r in rows if r["w"] == w]
    c = sum(r["conflict"] for r in rs)
    cells = []
    for f in preds.values():
        flag = [r for r in rs if f(r)]
        tp = sum(r["conflict"] for r in flag)
        cells.append(f"{100*len(flag)/len(rs):5.1f} {tp/max(len(flag),1):5.2f} {tp/max(c,1):5.2f}")
    print(f"{w:>2} {len(rs):>4} {c:>5} | " + " | ".join(f"{x:>18}" for x in cells))
missed = [r for r in rows if r["conflict"] and not r["file"]]
print("conflicts with no shared file:", len(missed), [m["commit"] for m in missed[:5]])
