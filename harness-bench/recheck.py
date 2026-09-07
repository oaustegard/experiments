#!/usr/bin/env python3
"""Sub-minute fixture: every count RESULTS.md states must match results/*.json."""
import json, math, re, sys
from pathlib import Path

R = Path(__file__).parent
arms = {a: json.loads((R / f"results/{a}.json").read_text())
        for a in ("oneshot", "agentic", "toolloop")}
opus = {a: json.loads((R / f"results/{a}.json").read_text())
        for a in ("opus-oneshot", "opus-toolloop")}
full = json.loads((R / "results/full-oneshot.json").read_text())
prose = (R / "RESULTS.md").read_text()
fail = []


def check(cond, msg):
    print(f"{'ok  ' if cond else 'FAIL'} {msg}")
    if not cond:
        fail.append(msg)


tot = {a: sum(v["passed"] for v in d.values()) for a, d in arms.items()}
check(tot == {"oneshot": 3, "agentic": 8, "toolloop": 11}, f"arm totals {tot}")
for a, n in tot.items():
    check(f"**{n}/12**" in prose, f"RESULTS.md states {n}/12 for {a}")

check(all(len(d) == 12 for d in arms.values()), "12 tasks graded in every arm")
strays = {f"{a}:{k}": v["stray"] for a, d in arms.items() for k, v in d.items() if v["stray"]}
check(not strays, f"no files touched outside the solution set ({strays or 'none'})")


for a, n in (("opus-oneshot", 29), ("opus-toolloop", 30)):
    got = sum(v["passed"] for v in opus[a].values())
    check(got == n and len(opus[a]) == 30, f"{a} scores {got}/30 (RESULTS.md says {n}/30)")
    check(f"**{n}/30**" in prose, f"RESULTS.md states {n}/30 for {a}")
ostray = {f"{a}:{k}": v["stray"] for a, d in opus.items() for k, v in d.items() if v["stray"]}
check(not ostray, f"no strays in the opus arms ({ostray or 'none'})")


fk = sum(v["passed"] for v in full.values())
check(len(full) == 203, f"full arm graded {len(full)} tasks (RESULTS.md says 203)")
check(fk == 182, f"full arm scores {fk}/203 (RESULTS.md says 182/203)")
check("**182/203 = 0.897**" in prose, "RESULTS.md states 182/203 = 0.897")
fstray = {k: v["stray"] for k, v in full.items() if v["stray"]}
check(not fstray, f"no strays in the full arm ({fstray or 'none'})")
bylang = {}
for k, v in full.items():
    l = k.split("/")[0]
    p_, t_ = bylang.get(l, (0, 0))
    bylang[l] = (p_ + v["passed"], t_ + 1)
for l, (p_, t_) in sorted(bylang.items()):
    check(f"{p_}/{t_} = {p_ / t_:.3f}" in prose, f"RESULTS.md states {l} {p_}/{t_}")


def mcnemar(x, y):
    n01 = sum(1 for k in x if not x[k]["passed"] and y[k]["passed"])
    n10 = sum(1 for k in x if x[k]["passed"] and not y[k]["passed"])
    n = n01 + n10
    p = min(1.0, sum(math.comb(n, i) for i in range(min(n01, n10) + 1)) / 2 ** n * 2) if n else 1.0
    return n01, n10, p


for a, b, want in (("oneshot", "agentic", (5, 0, 0.0625)),
                   ("agentic", "toolloop", (3, 0, 0.2500)),
                   ("oneshot", "toolloop", (8, 0, 0.0078))):
    n01, n10, p = mcnemar(arms[a], arms[b])
    check((n01, n10) == want[:2] and abs(p - want[2]) < 5e-5,
          f"{a}->{b}: +{n01} -{n10} p={p:.4f} (RESULTS.md says +{want[0]} -{want[1]} p={want[2]})")

# the per-task table in RESULTS.md must agree with the JSON, cell by cell
for line in prose.splitlines():
    m = re.match(r"\| (python|go|rust)/([\w-]+) \| (.+?) \| (.+?) \| (.+?) \|$", line)
    if not m:
        continue
    key = f"{m.group(1)}/{m.group(2)}"
    for arm, cell in zip(("oneshot", "agentic", "toolloop"), m.groups()[2:]):
        stated = "PASS" in cell.upper()
        check(arms[arm][key]["passed"] == stated, f"table cell {key} / {arm}")

print(f"\n{'PASS' if not fail else str(len(fail)) + ' MISMATCH'}")
sys.exit(1 if fail else 0)
