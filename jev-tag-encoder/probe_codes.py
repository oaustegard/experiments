"""Follow-up to quant_eval: the jev256 probe did better on 1/2/3-bit codes than on the floats.

Hypothesis: StandardScaler divides each tag by its SD, so a rare tag whose values wander between
0.01 and 0.03 gets unit variance and its noise enters the probe at full weight; quantizing sets
those values to one level. Control: float features without standardization. 5 seeds x 50/200/500
labels, same C selection as classify.py. Bootstrap is over test papers, micro metrics only.
"""
import json
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import KFold

import classify
import jev
from common import DATA, RESULTS, arxiv, boot, ci, f1s, fmt, jsonl, label_scores, save
from embed import EMB
from quant import Quantizer

SIZES, SEEDS = classify.SIZES, classify.SEEDS
# unscaled features live on very different scales (unit-norm 3,072-d embeddings: ~0.018 per dim),
# so the unscaled arms get a grid reaching C = 1000 (sparseup-tag-probe ERRORS: one C across scales)
CS_WIDE = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]


def fit_predict_raw(Xtr, Ytr, Xte, C):
    """classify.fit_predict without the StandardScaler."""
    S = np.zeros((len(Xte), Ytr.shape[1]))
    for j in range(Ytr.shape[1]):
        y = Ytr[:, j]
        if y.all() or not y.any():
            S[:, j] = float(y.any())
            continue
        S[:, j] = LogisticRegression(C=C, class_weight="balanced", max_iter=2000).fit(Xtr, y).predict_proba(Xte)[:, 1]
    return S


def pick_c_raw(X, Y, seed):
    best = None
    for C in CS_WIDE:
        pred = np.zeros_like(Y, dtype=bool)
        for tr, va in KFold(3, shuffle=True, random_state=seed).split(X):
            pred[va] = fit_predict_raw(X[tr], Y[tr], X[va], C) >= 0.5
        f = f1s(Y, pred)[0]
        if best is None or f > best[0]:
            best = (f, C)
    return best[1]


def main():
    fit = jev.matrix("mixed", "about", [r["id"] for r in jsonl(DATA / "mixed.jsonl")])
    fit = fit[~np.isnan(fit).any(1)]
    te_ids, Yte, _ = arxiv("test")
    tr_ids, Ytr, _ = arxiv("train")
    Pte, Ptr = jev.matrix("arxiv", "about", te_ids), jev.matrix("arxiv", "about", tr_ids)
    feats = {"float": (Ptr, Pte, True), "float-noscale": (Ptr, Pte, False)}
    for b, s in [(1, "sign"), (2, "logit"), (3, "logit")]:
        q = Quantizer(b, s, fit)
        feats[q.name] = (q(Ptr), q(Pte), True)
    all_ids, _, _ = arxiv()
    pos = {i: k for k, i in enumerate(all_ids)}
    E = np.load(EMB / "arxiv.npy")
    feats["dense"] = (E[[pos[i] for i in tr_ids]], E[[pos[i] for i in te_ids]], True)
    feats["dense-noscale"] = (E[[pos[i] for i in tr_ids]], E[[pos[i] for i in te_ids]], False)
    only = sys.argv[1].split(",") if len(sys.argv) > 1 else None
    if only:
        feats = {k: v for k, v in feats.items() if k in only}
    n = len(Yte)
    zs = label_scores(Pte)
    zs_boot = boot(lambda i: (f1s(Yte[i], zs[i] >= 0.5)[0], average_precision_score(Yte[i].ravel(), zs[i].ravel())),
                   n, seed=6)
    prev = RESULTS / "probe_codes.json"
    out, rows = (json.loads(prev.read_text()) if prev.exists() else {}), []
    for fname, (Xtr, Xte, scale) in feats.items():
        for k in SIZES:
            Ss = []
            for s in SEEDS:
                idx = np.random.default_rng(1000 * k + s).choice(len(Ytr), k, replace=False)
                if scale:
                    C = classify.pick_c(Xtr[idx], Ytr[idx], s)
                    Ss.append(classify.fit_predict(Xtr[idx], Ytr[idx], Xte, C))
                else:
                    C = pick_c_raw(Xtr[idx], Ytr[idx], s)
                    Ss.append(fit_predict_raw(Xtr[idx], Ytr[idx], Xte, C))

            def stat(i, Ss=Ss):
                return np.mean([(f1s(Yte[i], S[i] >= 0.5)[0], average_precision_score(Yte[i].ravel(), S[i].ravel()))
                                for S in Ss], axis=0)
            pt = stat(np.arange(n))
            bs = boot(stat, n, seed=6)
            c = ci(bs)
            dc = ci(bs - zs_boot)
            name = f"{fname}@{k}"
            out[name] = {"micro_f1": float(pt[0]), "micro_ap": float(pt[1]), "ci": c,
                         "minus_zs": [[float(pt[m] - [0.6363, 0.6102][m])] + dc[m] for m in range(2)]}
            rows.append(f"| {name} | {fmt(pt[0], c[0])} | {fmt(pt[1], c[1])} | "
                        f"{pt[0] - 0.6363:+.3f} [{dc[0][0]:+.3f}, {dc[0][1]:+.3f}] |")
            print(rows[-1], flush=True)
    save("probe_codes", out)


if __name__ == "__main__":
    main()
