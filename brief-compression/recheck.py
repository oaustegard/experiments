#!/usr/bin/env python3
"""Check RESULTS.md's numbers against data/. Under five seconds, no network."""
import json, sys
from pathlib import Path
from collections import defaultdict

D = Path(__file__).parent / "data"
A_CHECKS = 9; B_CHECKS = 13

def cells(task):
    sc = json.loads((D / f"task_{task}_scores.json").read_text())
    br = json.loads((D / f"task_{task}_briefs.json").read_text())
    us = json.loads((D / f"task_{task}_usage.json").read_text())
    out = defaultdict(list)
    for run, s in sc.items():
        out[(s["model"], s["style"])].append((s, br[run], us[run]))
    return out

def succ(v): return sum(s["success"] for s, _, _ in v)
def mean(v, f): return sum(f(x) for x in v) / len(v)

fails = []
def expect(cond, msg):
    if not cond: fails.append(msg)

A = cells("a"); B = cells("b")
expect(sum(len(v) for v in A.values()) == 36, "Task A has 36 runs")
expect(sum(len(v) for v in B.values()) == 60, "Task B has 60 runs")
expect(all(len(v) == 6 for v in A.values()), "Task A 6 per cell")
expect(all(len(v) == 10 for v in B.values()), "Task B 10 per cell")
expect(sum(succ(v) for v in A.values()) == 33, "Task A 33/36 success")

want_b = {("haiku", "prose"): 10, ("haiku", "structured"): 9, ("haiku", "telegraphic"): 8,
          ("sonnet", "prose"): 10, ("sonnet", "structured"): 2, ("sonnet", "telegraphic"): 6}
for k, n in want_b.items():
    expect(succ(B[k]) == n, f"Task B {k} success {succ(B[k])} != {n}")

# structured Sonnet: 8 top_user misses, and success 10/10 with that check excluded
ss = B[("sonnet", "structured")]
expect(sum(not s["top_user_ok"] for s, _, _ in ss) == 8, "8 top_user misses in sonnet structured")
expect(all(all(s[c] for c in s if c not in ("top_user_ok", "success", "n_pass") and isinstance(s[c], bool)) for s, _, _ in ss),
       "sonnet structured passes everything except top_user")

# Svc_ops mechanism: every sonnet telegraphic rows_kept miss equals the case-insensitive count
import csv
st = B[("sonnet", "telegraphic")]
expect(sum(not s["rows_kept_ok"] for s, _, _ in st) == 4, "4 rows_kept misses in sonnet telegraphic")

# cost table
def cost(k):
    v = B[k]
    return (round(mean(v, lambda x: x[2]["tokens"])), round(mean(v, lambda x: x[2]["tool_uses"]), 1))
expect(cost(("haiku", "prose"))[0] == 38959, "haiku prose sub_tokens 38,959")
expect(cost(("haiku", "telegraphic"))[0] == 45225, "haiku telegraphic sub_tokens 45,225")
expect(cost(("sonnet", "prose"))[0] == 50217, "sonnet prose sub_tokens 50,217")
expect(cost(("sonnet", "telegraphic"))[0] == 51736, "sonnet telegraphic sub_tokens 51,736")
expect(cost(("haiku", "telegraphic"))[1] == 4.1 and cost(("haiku", "prose"))[1] == 2.0, "haiku tool calls 4.1 vs 2.0")
expect(cost(("sonnet", "telegraphic"))[1] == 2.9 and cost(("sonnet", "prose"))[1] == 1.0, "sonnet tool calls 2.9 vs 1.0")

# brief token ordering holds in every cell
for task, C in (("a", A), ("b", B)):
    for m in ("haiku", "sonnet"):
        p, s_, t = (mean(C[(m, st_)], lambda x: x[1]["brief_o200k"]) for st_ in ("prose", "structured", "telegraphic"))
        expect(p > s_ > t, f"task {task} {m}: brief tokens prose > structured > telegraphic")

# Fisher pooled telegraphic vs prose
try:
    from scipy.stats import fisher_exact
    tp = succ(B[("haiku", "telegraphic")]) + succ(B[("sonnet", "telegraphic")])
    p = fisher_exact([[tp, 20 - tp], [20, 0]])[1]
    expect(abs(p - 0.0202) < 0.001, f"pooled Fisher p {p:.4f} != 0.020")
except ImportError:
    print("scipy missing; Fisher check skipped")

if fails:
    print("RECHECK FAILED"); [print("  -", f) for f in fails]; sys.exit(1)
print("recheck ok")
