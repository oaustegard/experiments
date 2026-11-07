#!/usr/bin/env python3
"""Bootstrap CI for AUROC(Jev B) - AUROC(category prior), per-category gold tic rates, and
Jev B AUROC on all 300 items against each judge. Writes results/ci.json."""
import json, random, collections
from pathlib import Path
from sklearn.metrics import roc_auc_score
D = Path(__file__).resolve().parent / "data"
L = lambda p: [json.loads(l) for l in open(p)]
items = {i["id"]: i for i in L(D / "items.jsonl")}
o = {r["id"]: r["label"] == "tic" for b in range(4) for r in L(D / f"judges/opus_{b}.jsonl")}
s = {r["id"]: r["label"] == "tic" for b in range(4) for r in L(D / f"judges/sonnet_{b}.jsonl")}
gold = {i: o[i] for i in o if o[i] == s[i]}
B = [{r["id"]: r["p"] for r in L(D / f"jev_B_{k}.jsonl")} for k in (1, 2)]
pb = {i: (B[0][i] + B[1][i]) / 2 for i in items}
ids = list(gold)
def prior(i, pool):
    same = [gold[j] for j in pool if j != i and items[j]["category"] == items[i]["category"]]
    return sum(same) / len(same) if same else sum(gold.values()) / len(gold)
pr = {i: prior(i, ids) for i in ids}
rng = random.Random(0); diffs = []
for _ in range(2000):
    smp = [rng.choice(ids) for _ in ids]
    y = [gold[i] for i in smp]
    if len(set(y)) < 2: continue
    diffs.append(roc_auc_score(y, [pb[i] for i in smp]) - roc_auc_score(y, [pr[i] for i in smp]))
diffs.sort()
cat = collections.defaultdict(list)
for i in ids: cat[items[i]["category"]].append(gold[i])
out = {"auroc_diff_jevB_minus_prior": {"point": round(roc_auc_score([gold[i] for i in ids], [pb[i] for i in ids]) - roc_auc_score([gold[i] for i in ids], [pr[i] for i in ids]), 3),
        "ci95": [round(diffs[int(.025 * len(diffs))], 3), round(diffs[int(.975 * len(diffs))], 3)]},
       "gold_tic_rate_by_category": {c: f"{sum(v)}/{len(v)}" for c, v in sorted(cat.items(), key=lambda kv: -len(kv[1])) if len(v) >= 5}}
(D.parent / "results/ci.json").write_text(json.dumps(out, indent=1) + "\n"); print(json.dumps(out, indent=1))
