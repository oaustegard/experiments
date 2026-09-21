"""The modern-population test: PMC full-text MSD papers (positives, ~9.8k, mostly 2019+) vs their own PubMed
similar-article neighbours (negatives), SPECTER2 + logistic regression, split by PMID hash. Also: recall of the
PMC-trained detector on the old curated bibliography, and the bibliography-trained detector on PMC positives.
Reports AUC, recall at fixed neighbour-FPR (1%, 5%, 10%), and FPR at fixed recall (90/95/98). Writes results/paper_pmc_probe.json."""
import json, os, sys, hashlib
import numpy as np
from sklearn.linear_model import LogisticRegression
HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(HERE, "data", "papers")
P = lambda f: [json.loads(l) for l in open(os.path.join(D, f)) if l.strip()]
E = {}
for l in open(os.path.join(D, "s2_specter2.jsonl")):
    r = json.loads(l); E[str(r["pmid"])] = np.array(r["vector"], dtype=np.float32)
pos = [r for r in P("pmc_positives.jsonl") if str(r["pmid"]) in E]; neg = [r for r in P("neighborspmc.jsonl") if str(r["pmid"]) in E]
bib = [r for r in P("positives.jsonl") if str(r["pmid"]) in E]
h = lambda r: int(hashlib.sha1(("20260921" + str(r["pmid"])).encode()).hexdigest()[:8], 16) % 100
split = lambda rs: ([r for r in rs if h(r) < 70], [r for r in rs if 70 <= h(r) < 85], [r for r in rs if h(r) >= 85])
ptr, pdv, pte = split(pos); ntr, ndv, nte = split(neg)
print(f"PMC positives {len(pos)} ({len(ptr)}/{len(pdv)}/{len(pte)}), neighbour negatives {len(neg)} ({len(ntr)}/{len(ndv)}/{len(nte)}), bibliography positives {len(bib)}")
X = lambda rs: np.stack([E[str(r["pmid"])] for r in rs])
Xtr = np.vstack([X(ptr), X(ntr)]); ytr = np.array([1] * len(ptr) + [0] * len(ntr)); m, s = Xtr.mean(0), Xtr.std(0) + 1e-6
Xdv = np.vstack([X(pdv), X(ndv)]); ydv = np.array([1] * len(pdv) + [0] * len(ndv)); Xte = np.vstack([X(pte), X(nte)]); yte = np.array([1] * len(pte) + [0] * len(nte))
def auc(y, sc):
    o = np.argsort(sc, kind="stable"); rk = np.empty(len(sc)); rk[o] = np.arange(1, len(sc) + 1); n1 = y.sum(); return float((rk[y == 1].sum() - n1 * (n1 + 1) / 2) / max(1, n1 * (len(y) - n1)))
best = None
for C in (0.01, 0.03, 0.1, 0.3, 1.0):
    clf = LogisticRegression(C=C, max_iter=5000, class_weight="balanced").fit((Xtr - m) / s, ytr); a = auc(ydv, clf.decision_function((Xdv - m) / s))
    if best is None or a > best[0]: best = (a, C, clf)
a_dv, C, clf = best; sc = clf.decision_function((Xte - m) / s); res = {"C": C, "dev_auc": a_dv, "test_auc": auc(yte, sc), "n_test_pos": int(len(pte)), "n_test_neg": int(len(nte))}
print(f"PMC-trained: C={C} dev AUC {a_dv:.3f} TEST AUC {res['test_auc']:.3f}")
negs = np.sort(sc[yte == 0])[::-1]; poss = np.sort(sc[yte == 1])[::-1]
for fpr in (0.01, 0.05, 0.10):
    thr = negs[int(fpr * len(negs))]; res[f"recall_at_fpr{int(fpr*100)}"] = float((sc[yte == 1] >= thr).mean()); print(f"  recall at neighbour-FPR {fpr:.0%}: {res[f'recall_at_fpr{int(fpr*100)}']:.3f}")
for rc in (0.90, 0.95, 0.98):
    thr = poss[int(np.ceil(rc * len(poss))) - 1]; res[f"fpr_at_recall{int(rc*100)}"] = float((sc[yte == 0] >= thr).mean()); print(f"  neighbour-FPR at recall {rc:.0%}: {res[f'fpr_at_recall{int(rc*100)}']:.3f}")
# cross-population: the PMC-trained detector on the old curated bibliography (all of it is unseen by this model)
sb = clf.decision_function((X(bib) - m) / s); thr50 = 0.0
res["bibliography_recall_at_thr0"] = float((sb >= 0).mean()); print(f"PMC-trained detector on the {len(bib)} curated bibliography papers: recall at the 0.5 point {res['bibliography_recall_at_thr0']:.3f}")
# abstract-cue subset of the PMC test positives
cue = np.array([bool(r["has_cue"]) for r in pte]); res["test_pos_cue_rate"] = float(cue.mean()); res["cue_free_recall_at_fpr5"] = float((sc[yte == 1][~cue] >= negs[int(0.05 * len(negs))]).mean())
print(f"cue rate among PMC test positives {cue.mean():.3f}; cue-free recall at 5% FPR {res['cue_free_recall_at_fpr5']:.3f}")
# --- the teammate's objection: nonlinear classifiers over the same vectors, same split
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import time
Ztr, Zte = (Xtr - m) / s, (Xte - m) / s
arms = {}
def report(name, sc_te, t):
    a = auc(yte, sc_te); ng = np.sort(sc_te[yte == 0])[::-1]; ps = np.sort(sc_te[yte == 1])[::-1]
    arms[name] = {"auc": a, "recall_at_fpr5": float((sc_te[yte == 1] >= ng[int(0.05 * len(ng))]).mean()), "fpr_at_recall95": float((sc_te[yte == 0] >= ps[int(np.ceil(0.95 * len(ps))) - 1]).mean()), "seconds": t}
    print(f"  {name:28s} AUC {a:.3f} | recall @5% FPR {arms[name]['recall_at_fpr5']:.3f} | FPR @95% recall {arms[name]['fpr_at_recall95']:.3f} ({t:.0f}s)")
print("nonlinear arms:")
report("LR (hyperplane)", sc, 0.0)
t0 = time.time(); U = lambda A: A / np.linalg.norm(A, axis=1, keepdims=True); Ctr, Cte = U(Xtr - m), U(Xte - m); S = Cte @ Ctr.T; nn = np.argsort(-S, axis=1)[:, :10]; report("10-NN vote, centred cosine", ytr[nn].mean(1) + 1e-6 * S.max(1), time.time() - t0)
t0 = time.time(); svm = SVC(kernel="rbf", C=3.0, gamma="scale", class_weight="balanced").fit(Ztr, ytr); report("RBF-kernel SVM", svm.decision_function(Zte), time.time() - t0)
t0 = time.time(); mlp = MLPClassifier(hidden_layer_sizes=(256,), early_stopping=True, max_iter=200, random_state=0).fit(Ztr, ytr); report("MLP 256 hidden", mlp.predict_proba(Zte)[:, 1], time.time() - t0)
res["nonlinear_arms"] = arms
# --- temporal holdout: train on <=2023, test on 2024+ (both classes), LR
yr = lambda r: int(str(r.get("year") or 0)[:4])
tr_rows = [(r, 1) for r in pos if yr(r) <= 2023] + [(r, 0) for r in neg if yr(r) <= 2023]; te_rows = [(r, 1) for r in pos if yr(r) >= 2024] + [(r, 0) for r in neg if yr(r) >= 2024]
if len(te_rows) > 100 and sum(y for _, y in te_rows) > 20:
    Xa = np.stack([E[str(r["pmid"])] for r, _ in tr_rows]); ya = np.array([y for _, y in tr_rows]); Xb = np.stack([E[str(r["pmid"])] for r, _ in te_rows]); yb = np.array([y for _, y in te_rows])
    ma, sa = Xa.mean(0), Xa.std(0) + 1e-6; clf_t = LogisticRegression(C=C, max_iter=5000, class_weight="balanced").fit((Xa - ma) / sa, ya); sct = clf_t.decision_function((Xb - ma) / sa)
    res["temporal_holdout"] = {"train_n": len(tr_rows), "test_n": len(te_rows), "test_pos": int(yb.sum()), "auc": auc(yb, sct)}
    print(f"temporal holdout (train <=2023 n={len(tr_rows)}, test 2024+ n={len(te_rows)}, {int(yb.sum())} pos): LR AUC {res['temporal_holdout']['auc']:.3f}")
json.dump(res, open(os.path.join(HERE, "results", "paper_pmc_probe.json"), "w"), indent=1); print("done -> results/paper_pmc_probe.json")
