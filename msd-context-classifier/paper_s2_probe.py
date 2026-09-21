"""Logistic-regression probe on Semantic Scholar's precomputed SPECTER2 embeddings (768-d, title+abstract).
Usage: python3 paper_s2_probe.py --corpus data/paper_corpus.jsonl --name paper_s2probe [--emb data/papers/s2_specter2.jsonl]
Same split as train.py (split_rows on the corpus rows); rows without an embedding are dropped and counted.
Writes results/<name>.json and results/<name>_test_preds.jsonl in train.py's format so paper_score.py scores it."""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import load_corpus, split_rows, evaluate, macro_f1, softmax, log
from sklearn.linear_model import LogisticRegression

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(); ap.add_argument("--corpus", required=True); ap.add_argument("--name", required=True)
ap.add_argument("--emb", default=os.path.join(HERE, "data", "papers", "s2_specter2.jsonl")); ap.add_argument("--knn", type=int, default=0, help="also report a k-NN-to-positives centroid/cosine baseline")
a = ap.parse_args()
E = {}
for l in open(a.emb):
    r = json.loads(l); E[f"pmid:{r['pmid']}"] = np.array(r["vector"], dtype=np.float32)
rows = load_corpus(a.corpus); n0 = len(rows); rows = [r for r in rows if r["url"] in E]
log(f"{n0} rows, {len(rows)} with a SPECTER2 embedding ({n0-len(rows)} dropped)")
labels = sorted({r["label"] for r in rows}); lid = {l: i for i, l in enumerate(labels)}
tr, dv, te = split_rows(rows)
X = lambda rs: np.stack([E[r["url"]] for r in rs]); Y = lambda rs: np.array([lid[r["label"]] for r in rs])
Xtr, Xdv, Xte = X(tr), X(dv), X(te); ytr, ydv, yte = Y(tr), Y(dv), Y(te)
mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6; Xtr, Xdv, Xte = (Xtr - mu) / sd, (Xdv - mu) / sd, (Xte - mu) / sd
t0 = time.time(); best = None
for C in (0.01, 0.03, 0.1, 0.3, 1.0):
    clf = LogisticRegression(C=C, max_iter=5000, class_weight="balanced").fit(Xtr, ytr)
    f = macro_f1(clf.predict(Xdv), ydv, len(labels))
    if best is None or f > best[0]: best = (f, C, clf)
f, C, clf = best; log(f"best C={C} dev macroF1={f:.3f} ({time.time()-t0:.0f}s)")
ldv, lte = clf.decision_function(Xdv), clf.decision_function(Xte)
ldv = np.stack([-ldv, ldv], 1); lte = np.stack([-lte, lte], 1)
res = evaluate(a.name, ldv, ydv, lte, yte, len(labels), labels, {"arm": "s2_specter2_probe", "C": C, "n_dropped_no_embedding": n0 - len(rows), "train_seconds": time.time() - t0})
res["labels"] = labels; res["split"] = {"train": len(tr), "dev": len(dv), "test": len(te)}
if a.knn:
    # cosine to the mean of the training positives: the zero-parameter "find more like these" baseline
    P = np.stack([E[r["url"]] for r in tr if r["label"] == "msd"]); c = P.mean(0); c /= np.linalg.norm(c)
    Xraw = np.stack([E[r["url"]] for r in te]); s = (Xraw / np.linalg.norm(Xraw, axis=1, keepdims=True)) @ c
    order = np.argsort(s); ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1); y = (yte == lid["msd"]).astype(int)
    auc = float((ranks[y == 1].sum() - y.sum() * (y.sum() + 1) / 2) / max(1, y.sum() * (len(y) - y.sum())))
    res["centroid_cosine_auc"] = auc; log(f"centroid-cosine AUC {auc:.3f}")
    # k-NN vote over the training set (cosine), k=a.knn if >1 else 10: the "more like these" tool with a labelled seed set
    k = a.knn if a.knn > 1 else 10
    Tr = np.stack([E[r["url"]] for r in tr]); Tr = Tr / np.linalg.norm(Tr, axis=1, keepdims=True); ytr_pos = (ytr == lid["msd"]).astype(float)
    Q = Xraw / np.linalg.norm(Xraw, axis=1, keepdims=True); S = Q @ Tr.T
    nn = np.argsort(-S, axis=1)[:, :k]; s_knn = ytr_pos[nn].mean(1)
    order = np.argsort(s_knn, kind="stable"); ranks = np.empty(len(s_knn)); ranks[order] = np.arange(1, len(s_knn) + 1)
    auc_knn = float((ranks[y == 1].sum() - y.sum() * (y.sum() + 1) / 2) / max(1, y.sum() * (len(y) - y.sum())))
    res["knn_cosine_auc"] = auc_knn; res["knn_k"] = k; log(f"{k}-NN vote AUC {auc_knn:.3f}")
os.makedirs(os.path.join(HERE, "results"), exist_ok=True); json.dump(res, open(os.path.join(HERE, "results", f"{a.name}.json"), "w"), indent=1)
with open(os.path.join(HERE, "results", f"{a.name}_test_preds.jsonl"), "w") as f_:
    for r, lg in zip(te, softmax(lte, res["temperature"])):
        f_.write(json.dumps({"url": r["url"], "label": r["label"], "pred": labels[int(lg.argmax())], "probs": {labels[i]: float(lg[i]) for i in range(len(labels))}}) + "\n")
log(f"acc {res['accuracy']:.3f} macroF1 {res['macro_f1']:.3f} per-label " + str({k: round(v['f1'], 3) for k, v in res['per_label_f1'].items()}))
print(f"done -> results/{a.name}.json")
