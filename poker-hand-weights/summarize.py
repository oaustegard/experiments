#!/usr/bin/env python3
"""Markdown tables from results/headtohead.json: the network's result per game and opponent, and the
existing players against the same baselines for scale. Cells are big blinds per 100 hands for the row's
player, ± the 95% interval (1.96 standard errors); a big blind is 20 chips.

  python3 poker-hand-weights/summarize.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = json.loads((HERE / "results" / "headtohead.json").read_text())
if (HERE / "results" / "reference.json").exists():
    R.update(json.loads((HERE / "results" / "reference.json").read_text()))
GAMES = [("holdem", "nl", 6, "Hold'em no-limit, 6 seats"), ("holdem", "nl", 2, "Hold'em no-limit, heads-up"),
         ("holdem", "fl", 6, "Hold'em fixed-limit"), ("omaha", "pl", 6, "Omaha pot-limit"),
         ("omaha8", "fl", 6, "Omaha Hi-Lo"), ("stud7", "fl", 6, "Seven-Card Stud"), ("stud8", "fl", 6, "Stud Hi-Lo"),
         ("razz", "fl", 6, "Razz"), ("stud5", "fl", 6, "Five-Card Stud"), ("draw5", "fl", 6, "Five-Card Draw")]


def cell(k):
    r = R.get(k)
    if not r:
        return "—"
    return f"{r['bb100']:+.1f} ± {r['ci95']:.1f}"


def table(rows, a, opps):
    out = ["| game | " + " | ".join(f"vs {o}" for o in opps) + " |", "|---|" + "---|" * len(opps)]
    for g, s, n, label in rows:
        cells = [cell(f"{g}-{s}-{n}-{a}-vs-{o}") for o in opps]
        if any(c != "—" for c in cells):
            out.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(out)


print("Hand-wired network, held-out deals (bb/100 for the network, ± 95% interval):\n")
print(table(GAMES, "net", ["tag", "v1", "caller", "random"]))
print("\nReference, same deals: the tight-aggressive bot and v1 against the baselines:\n")
for a, opps in (("tag", ["v1", "caller", "random"]), ("v1", ["caller", "random"])):
    print(f"{a}:\n")
    print(table(GAMES, a, opps))
    print()
