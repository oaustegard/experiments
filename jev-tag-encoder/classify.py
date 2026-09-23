"""Step 3: arXiv multi-label (24 labels), Jev zero-shot vs dense + logistic probes on 50/200/500 labels.

Arms, all scored on the same 600 test papers:
  jev_zs@0.5          label score = max prob over the label's mapped tags, threshold 0.5
  jev_zs@oracle-glob  one threshold chosen on the test set (upper bound)
  jev_zs@oracle-lab   per-label thresholds chosen on the test set (upper bound)
  dense+probe@k       gemini-embedding-2, one logistic probe per label, k labelled train papers
  jev256+probe@k      same probe on the 256 Jev probabilities (does the vector carry more than the map?)
Probes: class_weight=balanced, C picked per arm and k by 3-fold CV on the train sample, 5 seeds of k.
F1 is at 0.5 for every arm; micro/macro-AP is reported too, since a fixed threshold orders arms
partly by calibration (sparseup-tag-probe ERRORS #5).
"""
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

import jev
from common import LABEL_NAMES, ap_scores, arxiv, boot, ci, f1s, fmt, label_scores, save
from embed import EMB

SIZES = [50, 200, 500]
SEEDS = range(5)
CS = [0.01, 0.1, 1.0, 10.0]


def fit_predict(Xtr, Ytr, Xte, C):
    """One probe per label; a label with no positives (or no negatives) in train predicts its constant."""
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    S = np.zeros((len(Xte), Ytr.shape[1]))
    for j in range(Ytr.shape[1]):
        y = Ytr[:, j]
        if y.all() or not y.any():
            S[:, j] = float(y.any())
            continue
        m = LogisticRegression(C=C, class_weight="balanced", max_iter=2000).fit(Xtr, y)
        S[:, j] = m.predict_proba(Xte)[:, 1]
    return S


def pick_c(X, Y, seed):
    best = None
    for C in CS:
        pred = np.zeros_like(Y, dtype=bool)
        for tr, va in KFold(3, shuffle=True, random_state=seed).split(X):
            pred[va] = fit_predict(X[tr], Y[tr], X[va], C) >= 0.5
        f = f1s(Y, pred)[0]
        if best is None or f > best[0]:
            best = (f, C)
    return best[1]


def main():
    te_ids, Yte, _ = arxiv("test")
    tr_ids, Ytr, _ = arxiv("train")
    all_ids, _, _ = arxiv()
    pos = {i: k for k, i in enumerate(all_ids)}
    E = np.load(EMB / "arxiv.npy")
    Ete, Etr = E[[pos[i] for i in te_ids]], E[[pos[i] for i in tr_ids]]

    Pte = jev.matrix("arxiv", "about", te_ids)
    ok = ~np.isnan(Pte).any(1)
    print(f"test papers with Jev vectors: {ok.sum()}/{len(ok)}")
    Yte, Ete, Pte = Yte[ok], Ete[ok], Pte[ok]
    n = len(Yte)
    Ste = label_scores(Pte)

    arms = {}  # name -> list of score matrices (one per seed); thresholded at thr
    thr = {}
    arms["jev_zs@0.5"] = [Ste]
    thr["jev_zs@0.5"] = 0.5
    grid = np.round(np.arange(0.05, 0.96, 0.05), 2)
    g = max(grid, key=lambda t: f1s(Yte, Ste >= t)[0])
    arms["jev_zs@oracle-glob"] = [Ste]
    thr["jev_zs@oracle-glob"] = g
    per = np.array([max(grid, key=lambda t: f1s(Yte[:, [j]], Ste[:, [j]] >= t)[0]) for j in range(Yte.shape[1])])
    arms["jev_zs@oracle-lab"] = [Ste]
    thr["jev_zs@oracle-lab"] = per
    print(f"oracle global threshold {g}; per-label thresholds {dict(zip(LABEL_NAMES, per.tolist()))}")

    Ptr = jev.matrix("arxiv", "about", tr_ids)
    have_tr = ~np.isnan(Ptr).any(1)
    feats = {"dense": (Etr, Ete)}
    if have_tr.mean() > 0.95:
        feats["jev256"] = (Ptr, Pte)
    else:
        print(f"jev256+probe skipped: only {have_tr.sum()}/{len(have_tr)} train papers encoded", file=sys.stderr)
    chosen_c = {}
    for fname, (Xtr_all, Xte) in feats.items():
        pool = np.where(have_tr)[0] if fname == "jev256" else np.arange(len(Ytr))
        for k in SIZES:
            name = f"{fname}+probe@{k}"
            arms[name], chosen_c[name] = [], []
            for s in SEEDS:
                idx = np.random.default_rng(1000 * k + s).choice(pool, k, replace=False)
                C = pick_c(Xtr_all[idx], Ytr[idx], s)
                chosen_c[name].append(C)
                arms[name].append(fit_predict(Xtr_all[idx], Ytr[idx], Xte, C))
            thr[name] = 0.5
            print(f"{name}: C per seed {chosen_c[name]}", flush=True)

    def stat(name):
        Ss, t = arms[name], thr[name]
        return lambda i: np.mean([f1s(Yte[i], S[i] >= t) + ap_scores(Yte[i], S[i]) for S in Ss], axis=0)

    out = {}
    rows = []
    for name in arms:
        point = stat(name)(np.arange(n))
        bs = boot(stat(name), n, seed=1)
        c = ci(bs)
        out[name] = {"micro_f1": point[0], "macro_f1": point[1], "micro_ap": point[2], "macro_ap": point[3],
                     "ci": c, "_boot": bs}
        seed_sd = np.std([f1s(Yte, S >= thr[name])[0] for S in arms[name]]) if len(arms[name]) > 1 else 0.0
        out[name]["seed_sd_micro_f1"] = float(seed_sd)
        rows.append(f"| {name} | {fmt(point[0], c[0])} | {fmt(point[1], c[1])} | {fmt(point[2], c[2])} | "
                    f"{fmt(point[3], c[3])} |")
    print("| arm | micro-F1 | macro-F1 | micro-AP | macro-AP |\n|---|---|---|---|---|")
    print("\n".join(rows))

    # paired differences vs the zero-shot arm (same bootstrap resamples: seed=1 for all)
    diffs = {}
    for name in arms:
        if name.startswith("jev_zs"):
            continue
        d = out["jev_zs@0.5"]["_boot"] - out[name]["_boot"]
        diffs[name] = {"micro_f1": [float(out["jev_zs@0.5"]["micro_f1"] - out[name]["micro_f1"])] + ci(d)[0],
                       "micro_ap": [float(out["jev_zs@0.5"]["micro_ap"] - out[name]["micro_ap"])] + ci(d)[2]}
        print(f"jev_zs@0.5 - {name}: micro-F1 {fmt(diffs[name]['micro_f1'][0], diffs[name]['micro_f1'][1:])}  "
              f"micro-AP {fmt(diffs[name]['micro_ap'][0], diffs[name]['micro_ap'][1:])}")

    # per-label zero-shot F1 (which mappings work)
    tp = (Yte & (Ste >= 0.5)).sum(0)
    fp = (~Yte & (Ste >= 0.5)).sum(0)
    fn = (Yte & ~(Ste >= 0.5)).sum(0)
    per_label = {lab: {"n_pos": int(Yte[:, j].sum()),
                       "f1": round(float(2 * tp[j] / max(1, 2 * tp[j] + fp[j] + fn[j])), 3),
                       "precision": round(float(tp[j] / max(1, tp[j] + fp[j])), 3),
                       "recall": round(float(tp[j] / max(1, tp[j] + fn[j])), 3)}
                 for j, lab in enumerate(LABEL_NAMES)}
    for lab, v in sorted(per_label.items(), key=lambda x: x[1]["f1"]):
        print(f"  {lab:<18} n={v['n_pos']:<3} F1 {v['f1']:.2f}  P {v['precision']:.2f}  R {v['recall']:.2f}")

    for v in out.values():
        v.pop("_boot")
    save("step3_classify", {"n_test": n, "arms": out, "diffs_vs_zs": diffs, "per_label_zs": per_label,
                            "oracle_global_thr": float(g), "chosen_C": chosen_c})


if __name__ == "__main__":
    main()
