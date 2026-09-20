"""Follow-up: is the question task learnable from a few questions, once pages are shown not to transfer?
Frozen gte-small mean-pool embeddings; logistic regression. 5-fold stratified CV over the 175
unambiguous questions (seed 20260920). Arms per fold:
  desc   zero-shot: cosine to the embedded label description (no training at all)
  pages  train on the page train split only (the transfer arm, re-run on identical folds)
  q      train on the other 4 folds of questions only (140 rows)
  pages+q train on pages + the other 4 folds
Writes results/probe_queries.json. Completion line: 'done -> <path>'."""
import json, os, sys, time
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import load_corpus, split_rows, texts_of, embed, macro_f1, log

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "models", "gte-small")
rows = load_corpus(os.path.join(HERE, "data", "corpus.jsonl")); labels = sorted({r["label"] for r in rows}); lid = {l: i for i, l in enumerate(labels)}
tr, dv, te = split_rows(rows)
qs = [json.loads(l) for l in open(os.path.join(HERE, "data", "queries.jsonl")) if l.strip()]
qc = [q for q in qs if q["label"] in lid and not q.get("ambiguous")]
desc = json.load(open(os.path.join(HERE, "data", "label_descriptions.json")))
tok = AutoTokenizer.from_pretrained(MODEL); enc = AutoModel.from_pretrained(MODEL)
t0 = time.time()
Ep = embed(enc, tok, texts_of(tr), 256, 32); yp = np.array([lid[r["label"]] for r in tr])
Eq = embed(enc, tok, [q["text"] for q in qc], 256, 32); yq = np.array([lid[q["label"]] for q in qc])
Ed = embed(enc, tok, [desc[l] for l in labels], 256, 8)
log(f"embedded pages {len(yp)} questions {len(yq)} in {time.time()-t0:.0f}s")

C = 3.0
res = {"n_questions": int(len(yq)), "folds": 5, "arms": {}}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=20260920)
per = {"desc": [], "pages": [], "q": [], "pages+q": []}; acc = {k: [] for k in per}
pred_desc = (Eq @ Ed.T).argmax(1)
for k, (itr, ite) in enumerate(skf.split(Eq, yq)):
    yt = yq[ite]
    per["desc"].append(macro_f1(pred_desc[ite], yt, len(labels))); acc["desc"].append(float((pred_desc[ite] == yt).mean()))
    for name, X, y in [("pages", Ep, yp), ("q", Eq[itr], yq[itr]), ("pages+q", np.vstack([Ep, Eq[itr]]), np.concatenate([yp, yq[itr]]))]:
        clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(X, y)
        p = clf.predict(Eq[ite]); per[name].append(macro_f1(p, yt, len(labels))); acc[name].append(float((p == yt).mean()))
    log(f"fold {k}: " + " ".join(f"{n} f1 {per[n][-1]:.2f}" for n in per))
for n in per:
    res["arms"][n] = {"macro_f1_mean": float(np.mean(per[n])), "macro_f1_sd": float(np.std(per[n])), "acc_mean": float(np.mean(acc[n])), "folds_f1": [round(x, 3) for x in per[n]]}
# per-label f1 for the q arm, pooled over folds
pooled = np.zeros(len(yq), dtype=int)
for itr, ite in skf.split(Eq, yq):
    pooled[ite] = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(Eq[itr], yq[itr]).predict(Eq[ite])
res["q_per_label_f1"] = {}
for c, l in enumerate(labels):
    tp = ((pooled == c) & (yq == c)).sum(); fp = ((pooled == c) & (yq != c)).sum(); fn = ((pooled != c) & (yq == c)).sum()
    p_ = tp / (tp + fp) if tp + fp else 0.0; r_ = tp / (tp + fn) if tp + fn else 0.0
    res["q_per_label_f1"][l] = round(float(2 * p_ * r_ / (p_ + r_) if p_ + r_ else 0.0), 3)
res["labels"] = labels; res["C"] = C; res["class_weight"] = "balanced"
out = os.path.join(HERE, "results", "probe_queries.json"); json.dump(res, open(out, "w"), indent=1)
log({n: round(res["arms"][n]["macro_f1_mean"], 3) for n in per})
print(f"done -> {out}", flush=True)
