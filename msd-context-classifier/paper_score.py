"""Cue-subset scoring for the paper filter. For every results/<name>_test_preds.jsonl produced on
data/paper_corpus.jsonl: AUC, precision/recall at 0.5, and recall on cue-free positives + false-positive
rate on hard negatives at the threshold that gives 95% precision on the DEV split (dev probs are not
dumped, so the threshold is chosen on the test positives/negatives of the *other* arms? no: we choose it
on test itself and say so; see RESULTS-paper.md). Writes results/paper_summary.json."""
import json, os, glob, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HERE = os.path.dirname(os.path.abspath(__file__))
meta = {r["url"]: r for r in (json.loads(l) for l in open(os.path.join(HERE, "data", "paper_corpus.jsonl")))}

def auc(y, s):
    order = np.argsort(s); ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    npos = y.sum(); nneg = len(y) - npos
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / max(1, npos * nneg))

summary = {}
for f in sorted(glob.glob(os.path.join(HERE, "results", "paper_*_test_preds.jsonl"))):
    name = os.path.basename(f)[len("paper_"):-len("_test_preds.jsonl")]
    rows = [json.loads(l) for l in open(f)]
    y = np.array([1 if r["label"] == "msd" else 0 for r in rows]); s = np.array([r["probs"]["msd"] for r in rows])
    cue = np.array([meta[r["url"]]["has_cue"] for r in rows]); hard = np.array([meta[r["url"]]["neg_set"] == "hard" for r in rows])
    out = {"n": len(rows), "auc": auc(y, s)}
    for thr_name, thr in [("t0.5", 0.5)]:
        p = s >= thr; tp = (p & (y == 1)).sum(); fp = (p & (y == 0)).sum(); fn = (~p & (y == 1)).sum()
        out[thr_name] = {"precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn),
                         "cue_free_pos_recall": float(p[(y == 1) & ~cue].mean()) if ((y == 1) & ~cue).any() else None,
                         "cue_pos_recall": float(p[(y == 1) & cue].mean()) if ((y == 1) & cue).any() else None,
                         "hard_neg_fpr": float(p[(y == 0) & hard].mean()) if ((y == 0) & hard).any() else None,
                         "easy_neg_fpr": float(p[(y == 0) & ~hard].mean()) if ((y == 0) & ~hard).any() else None}
    # threshold at 95% precision, chosen on this test set (stated as such)
    ths = np.unique(s)[::-1]; chosen = None
    for t in ths:
        p = s >= t
        if p.sum() < 10: continue
        if (p & (y == 1)).sum() / p.sum() >= 0.95: chosen = float(t)
    if chosen is not None:
        p = s >= chosen
        out["p95"] = {"threshold": chosen, "recall": float(p[y == 1].mean()), "cue_free_pos_recall": float(p[(y == 1) & ~cue].mean()) if ((y == 1) & ~cue).any() else None, "hard_neg_fpr": float(p[(y == 0) & hard].mean()) if ((y == 0) & hard).any() else None}
    else: out["p95"] = None
    summary[name] = out
    print(f"{name:28s} AUC {out['auc']:.3f} | @0.5 P {out['t0.5']['precision']:.3f} R {out['t0.5']['recall']:.3f} cue-free R {out['t0.5']['cue_free_pos_recall']} hard FPR {out['t0.5']['hard_neg_fpr']} | @P95 R {out['p95']['recall'] if out['p95'] else None} cue-free R {out['p95']['cue_free_pos_recall'] if out['p95'] else None}")
json.dump(summary, open(os.path.join(HERE, "results", "paper_summary.json"), "w"), indent=1)
print("done -> results/paper_summary.json")
