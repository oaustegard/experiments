#!/usr/bin/env python3
"""Bradley-Terry over all three judging arms, aggregated by model."""
from __future__ import annotations
import json, math
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
key = json.loads((HERE / "blind_key.json").read_text())
labels = sorted(key)

def fit(vs):
    wins = defaultdict(float); games = defaultdict(list)
    for a, b, w in vs:
        wins[w] += 1; games[a].append(b); games[b].append(a)
    p = {l: 1.0 for l in labels}
    for _ in range(2000):
        new = {l: (wins[l] + 0.5) / (sum(1.0/(p[l]+p[o]) for o in games[l]) + 1.0/(p[l]+1.0))
               for l in labels}
        n = math.exp(sum(math.log(v) for v in new.values()) / len(new))
        p = {l: v/n for l, v in new.items()}
    return {l: math.log(p[l]) for l in labels}, wins

r2 = json.loads((HERE / "verdicts_round2.json").read_text())
arms = {
    "sonnet/AI": json.loads((HERE / "verdicts.json").read_text()),
    "opus/AI": r2["ai"],
    "opus/STAGING": r2["staging"],
}
fits = {name: fit(v) for name, v in arms.items()}

print("per-model mean strength (higher = judged worse on that axis)\n")
print(f"{'model':12} {'sonnet/AI':>11} {'opus/AI':>9} {'opus/STAGING':>13}")
print("-" * 50)
agg = defaultdict(dict)
for name, (f, _) in fits.items():
    by_model = defaultdict(list)
    for l in labels:
        by_model[key[l].rsplit("-", 1)[0]].append(f[l])
    for m, vs in by_model.items():
        agg[m][name] = sum(vs) / len(vs)
for m in sorted(agg, key=lambda m: -agg[m]["opus/STAGING"]):
    a = agg[m]
    print(f"{m:12} {a['sonnet/AI']:+11.2f} {a['opus/AI']:+9.2f} {a['opus/STAGING']:+13.2f}")

s, ai = r2["staging"], r2["ai"]
disagree = sum(1 for x, y in zip(s, ai) if x[2] != y[2])
print(f"\nwithin-judge disagreement between the two questions: {disagree}/{len(s)} pairs")

a1 = {l: fits["sonnet/AI"][0][l] for l in labels}
a2 = {l: fits["opus/AI"][0][l] for l in labels}
a3 = {l: fits["opus/STAGING"][0][l] for l in labels}
def corr(x, y):
    xs = [x[l] for l in labels]; ys = [y[l] for l in labels]
    mx, my = sum(xs)/len(xs), sum(ys)/len(ys)
    num = sum((a-mx)*(b-my) for a, b in zip(xs, ys))
    den = math.sqrt(sum((a-mx)**2 for a in xs) * sum((b-my)**2 for b in ys))
    return num/den if den else 0.0
print(f"corr(sonnet/AI, opus/AI)       = {corr(a1,a2):+.2f}   judge-model effect, question held fixed")
print(f"corr(opus/AI,   opus/STAGING)  = {corr(a2,a3):+.2f}   question effect, judge held fixed")
