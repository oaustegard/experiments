#!/usr/bin/env python3
"""Calibration of each Jev question against the held-out suite, with task-clustered bootstrap CIs.

Score s = P(pass): P(Q) for Q1, Q3, Q4 and 1 - P(Q2) (Q2 asks about a defect left undone).
A judgement is wrong when s >= 0.5 disagrees with the held-out label. Mid-band = s in [0.3, 0.9].

  python3 analyze.py --scores data/jev/v1/dev --out results/v1_dev.json [--png results/v1_dev.png]
"""
import argparse
import glob
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
QS = ["Q1", "Q2", "Q3", "Q4"]
BAND = (0.3, 0.9)


def labels():
    out = {}
    for f in glob.glob(str(HERE / "data" / "attempts" / "*.json")):
        a = json.load(open(f))
        if "arm" in a:
            out[f"{a['arm']}_r{a['replicate']}_{a['task']}"] = (bool(a["heldout_pass"]), a)
    return out


def score(P, q):
    return None if P[q] is None else (1 - P[q] if q == "Q2" else P[q])


def auroc(s, y):
    pos = [v for v, l in zip(s, y) if l]
    neg = [v for v, l in zip(s, y) if not l]
    if not pos or not neg:
        return None
    w = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return w / (len(pos) * len(neg))


def within_task_auroc(rows, q):
    """AUROC over pass/fail pairs drawn from the same task only: judging the attempt, not the task."""
    w = n = 0
    by = {}
    for r in rows:
        by.setdefault(r["task"], []).append(r)
    for rs in by.values():
        for p in (r for r in rs if r["y"]):
            for f in (r for r in rs if not r["y"]):
                w += (p["s"][q] > f["s"][q]) + 0.5 * (p["s"][q] == f["s"][q])
                n += 1
    return w / n if n else None


def ece(s, y, bins=10):
    n, tot = len(s), 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, v in enumerate(s) if lo <= v < hi or (b == bins - 1 and v == 1.0)]
        if idx:
            tot += len(idx) / n * abs(sum(s[i] for i in idx) / len(idx) - sum(y[i] for i in idx) / len(idx))
    return tot


def reliability(s, y, bins=10):
    rows = []
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, v in enumerate(s) if lo <= v < hi or (b == bins - 1 and v == 1.0)]
        rows.append(dict(lo=lo, hi=hi, n=len(idx),
                         mean_s=sum(s[i] for i in idx) / len(idx) if idx else None,
                         frac_pass=sum(y[i] for i in idx) / len(idx) if idx else None))
    return rows


def stats(s, y):
    tp = sum(1 for v, l in zip(s, y) if l and v >= 0.5)
    tn = sum(1 for v, l in zip(s, y) if not l and v < 0.5)
    npos, nneg = sum(y), len(y) - sum(y)
    wrong = [v for v, l in zip(s, y) if (v >= 0.5) != l]
    mid = [v for v in wrong if BAND[0] <= v <= BAND[1]]
    return dict(n=len(y), n_pass=npos, n_fail=nneg, auroc=auroc(s, y), ece=ece(s, y),
                tpr=tp / npos if npos else None, tnr=tn / nneg if nneg else None,
                n_wrong=len(wrong), frac_wrong_midband=len(mid) / len(wrong) if wrong else None)


def boot(rows, q, reps=2000, seed=0):
    """Resample tasks with replacement; every attempt of a drawn task comes along."""
    rng = random.Random(seed)
    by = {}
    for r in rows:
        by.setdefault(r["task"], []).append(r)
    tasks = sorted(by)
    acc = {k: [] for k in ("auroc", "tpr", "tnr", "ece", "frac_wrong_midband")}
    for _ in range(reps):
        draw = [r for t in (rng.choice(tasks) for _ in tasks) for r in by[t]]
        st = stats([r["s"][q] for r in draw], [r["y"] for r in draw])
        for k in acc:
            if st[k] is not None:
                acc[k].append(st[k])
    ci = {}
    for k, v in acc.items():
        v.sort()
        ci[k] = [v[int(0.025 * len(v))], v[int(0.975 * len(v)) - 1]] if len(v) > 20 else None
    return ci


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--png")
    a = ap.parse_args()
    lab = labels()
    rows = []
    for f in sorted(glob.glob(a.scores + "/*.json")):
        d = json.load(open(f))
        if any(v is None for v in d["P"].values()):
            continue
        rows.append(dict(id=d["id"], task=d["task"], y=lab[d["id"]][0], s={q: score(d["P"], q) for q in QS}))
    out = {"n_attempts": len(rows), "n_tasks": len({r["task"] for r in rows}), "questions": {}}
    for q in QS:
        s, y = [r["s"][q] for r in rows], [r["y"] for r in rows]
        out["questions"][q] = dict(stats(s, y), within_task_auroc=within_task_auroc(rows, q), ci=boot(rows, q), reliability=reliability(s, y))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"n={out['n_attempts']} attempts, {out['n_tasks']} tasks, {sum(not r['y'] for r in rows)} failures")
    for q in QS:
        v = out["questions"][q]
        f = lambda x: "  - " if x is None else f"{x:.2f}"
        print(f"{q} AUROC {f(v['auroc'])} TPR {f(v['tpr'])} TNR {f(v['tnr'])} ECE {f(v['ece'])} "
              f"within-task AUROC {f(v['within_task_auroc'])} wrong {v['n_wrong']}/{v['n']} midband {f(v['frac_wrong_midband'])}")
    if a.png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 4, figsize=(14, 3.4), sharey=True)
        for ax, q in zip(axs, QS):
            rel = [r for r in out["questions"][q]["reliability"] if r["n"]]
            ax.plot([0, 1], [0, 1], "--", color="grey")
            ax.scatter([r["mean_s"] for r in rel], [r["frac_pass"] for r in rel],
                       s=[8 + 6 * r["n"] for r in rel])
            ax.axvspan(*BAND, alpha=0.08)
            ax.set_title(q + " (Q2 inverted)" if q == "Q2" else q)
            ax.set_xlabel("P(pass) from Jev")
        axs[0].set_ylabel("held-out pass rate")
        fig.tight_layout()
        fig.savefig(a.png, dpi=140)


if __name__ == "__main__":
    main()
