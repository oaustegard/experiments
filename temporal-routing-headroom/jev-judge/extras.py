#!/usr/bin/env python3
"""Exploratory checks run after the test split was read (labelled as such in RESULTS.md):
band base rate, a diff-size baseline, and the logistic combination with leave-one-task-out CV."""
import glob, json, sys
import numpy as np
from sklearn.linear_model import LogisticRegression
import analyze as A

scores = sys.argv[1]
lab = A.labels()
rows = []
for f in sorted(glob.glob(scores + "/*.json")):
    d = json.load(open(f)); y, a = lab[d["id"]]
    size = sum(1 for l in a["diff"].splitlines() if l[:1] in "+-" and l[:3] not in ("+++", "---"))
    rows.append(dict(task=d["task"], y=y, s={q: A.score(d["P"], q) for q in A.QS}, size=size,
                     tests_green=bool(a["visible_green"])))
out = {}
y = [r["y"] for r in rows]
for q in A.QS:
    s = [r["s"][q] for r in rows]
    inband = [A.BAND[0] <= v <= A.BAND[1] for v in s]
    wrong = [(v >= .5) != l for v, l in zip(s, y)]
    out[f"{q}_band_base_rate"] = sum(inband) / len(s)
    out[f"{q}_band_rate_correct"] = sum(b for b, w in zip(inband, wrong) if not w) / max(1, sum(not w for w in wrong))
    out[f"{q}_band_rate_wrong"] = sum(b for b, w in zip(inband, wrong) if w) / max(1, sum(wrong))
neg = [{"task": r["task"], "y": r["y"], "s": {"size": -r["size"]}} for r in rows]
out["diff_size_auroc"] = A.auroc([-r["size"] for r in rows], y)
out["diff_size_within_task_auroc"] = A.within_task_auroc(neg, "size")
out["tests_green_constant"] = len({r["tests_green"] for r in rows}) == 1
X = np.array([[r["s"][q] for q in A.QS] for r in rows]); Y = np.array(y)
tasks = np.array([r["task"] for r in rows]); pred = np.zeros(len(rows))
for t in set(tasks):
    m = tasks == t
    clf = LogisticRegression(C=1.0, max_iter=1000).fit(X[~m], Y[~m]); pred[m] = clf.predict_proba(X[m])[:, 1]
out["logit_Q1-4_loto_auroc"] = A.auroc(list(pred), y)
out["logit_Q1-4_loto_within_task_auroc"] = A.within_task_auroc(
    [{"task": r["task"], "y": r["y"], "s": {"p": p}} for r, p in zip(rows, pred)], "p")
out["logit_coefs_all_data"] = dict(zip(A.QS, LogisticRegression(C=1.0, max_iter=1000).fit(X, Y).coef_[0].round(2).tolist()))
for k, v in out.items():
    print(k, round(v, 3) if isinstance(v, float) else v)
json.dump(out, open("results/extras_test.json", "w"), indent=1)
