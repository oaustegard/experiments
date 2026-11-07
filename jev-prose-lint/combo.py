#!/usr/bin/env python3
"""Post-hoc arm (not pre-registered): Jev arm B combined with the hit's category prior.

5-fold CV logistic regression on [logit P_jev_B, logit smoothed category tic rate from the
training folds]. Writes results/combo.json.
"""
import json
import math
import random
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
D = HERE / "data"


def load(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def logit(x):
    x = min(1 - 1e-3, max(1e-3, x))
    return math.log(x / (1 - x))


def coverage(p, y, budget=0.05):
    best = 0.0
    cuts = sorted(set(p))
    for lo in cuts:
        for hi in cuts:
            if hi <= lo:
                continue
            st = [(x >= hi, t) for x, t in zip(p, y) if x <= lo or x >= hi]
            if st and sum(a != b for a, b in st) / len(st) <= budget:
                best = max(best, len(st) / len(p))
    return best


def main():
    items = {i["id"]: i for i in load(D / "items.jsonl")}
    o = {r["id"]: r["label"] == "tic" for b in range(4) for r in load(D / f"judges/opus_{b}.jsonl")}
    s = {r["id"]: r["label"] == "tic" for b in range(4) for r in load(D / f"judges/sonnet_{b}.jsonl")}
    gold = {i: o[i] for i in o if o[i] == s[i]}
    runs = [{r["id"]: r["p"] for r in load(D / f"jev_B_{k}.jsonl")} for k in (1, 2)]
    pb = {i: (runs[0][i] + runs[1][i]) / 2 for i in items}
    ids = list(gold)
    random.Random(1).shuffle(ids)

    def prior(i, pool):
        same = [gold[j] for j in pool if j != i and items[j]["category"] == items[i]["category"]]
        return (sum(same) + 1) / (len(same) + 2)

    pred = {}
    for f in range(5):
        test = set(ids[f::5])
        train = [i for i in ids if i not in test]
        m = LogisticRegression().fit([[logit(pb[i]), logit(prior(i, train))] for i in train],
                                     [gold[i] for i in train])
        for i in test:
            pred[i] = float(m.predict_proba([[logit(pb[i]), logit(prior(i, train))]])[0][1])
    y = [gold[i] for i in ids]
    p = [pred[i] for i in ids]
    pr = [prior(i, ids) for i in ids]
    out = {"n": len(ids),
           "combined": {"auroc": round(roc_auc_score(y, p), 3),
                        "accuracy": round(sum((x >= 0.5) == t for x, t in zip(p, y)) / len(y), 3),
                        "coverage_at_5pct": round(coverage(p, y), 3)},
           "prior_only_smoothed": {"auroc": round(roc_auc_score(y, pr), 3)}}
    (HERE / "results/combo.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
