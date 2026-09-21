"""Greedy-recall operating points on SPECTER2 vectors, with and without remex-style normalisation.
Arms (same split as paper_s2_probe.py): centroid cosine raw vs centred; 10-NN vote raw vs centred;
LR on raw float; LR on 1-bit sign codes (centred; centred + random rotation).
Reports AUC and hard/easy-negative FPR at test recall 0.90 / 0.95 / 0.98 / 0.99.
Writes results/paper_s2_greedy.json."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import load_corpus, split_rows
from sklearn.linear_model import LogisticRegression

HERE = os.path.dirname(os.path.abspath(__file__))
corpus = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "data", "paper_corpus.jsonl")
E = {}
for l in open(os.path.join(HERE, "data", "papers", "s2_specter2.jsonl")):
    r = json.loads(l); E[f"pmid:{r['pmid']}"] = np.array(r["vector"], dtype=np.float32)
rows = [r for r in load_corpus(corpus) if r["url"] in E]; tr, dv, te = split_rows(rows)
X = lambda rs: np.stack([E[r["url"]] for r in rs]); Y = lambda rs: np.array([1 if r["label"] == "msd" else 0 for r in rs])
Xtr, Xte, ytr, yte = X(tr), X(te), Y(tr), Y(te)
hard = np.array([r["neg_set"] in ("hard", "neighbor") for r in te]); easy = np.array([r["neg_set"] == "easy" for r in te])
unit = lambda A: A / np.linalg.norm(A, axis=1, keepdims=True)
mu = Xtr.mean(0)
rng = np.random.RandomState(20260921); Q, _ = np.linalg.qr(rng.randn(768, 768))

def auc(y, s):
    order = np.argsort(s, kind="stable"); ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    npos = y.sum(); return float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / max(1, npos * (len(y) - npos)))

def at_recall(y, s, target):
    pos = np.sort(s[y == 1])[::-1]; k = int(np.ceil(target * len(pos))); thr = pos[min(k, len(pos)) - 1]
    p = s >= thr
    return {"threshold": float(thr), "recall": float(p[y == 1].mean()), "hard_fpr": float(p[hard].mean()), "easy_fpr": float(p[easy].mean()), "precision": float(p[y == 1].sum() / max(1, p.sum()))}

def score_arm(name, s):
    out = {"auc": auc(yte, s), **{f"r{int(t*100)}": at_recall(yte, s, t) for t in (0.90, 0.95, 0.98, 0.99)}}
    print(f"{name:34s} AUC {out['auc']:.3f} | hard-FPR @R90 {out['r90']['hard_fpr']:.3f} @R95 {out['r95']['hard_fpr']:.3f} @R98 {out['r98']['hard_fpr']:.3f} @R99 {out['r99']['hard_fpr']:.3f} | easy-FPR @R98 {out['r98']['easy_fpr']:.3f}")
    return out

res = {}
# similarity-based
c_raw = unit(Xtr[ytr == 1].mean(0, keepdims=True)); res["centroid_raw"] = score_arm("centroid cosine, raw", (unit(Xte) @ c_raw.T)[:, 0])
c_ctr = unit((Xtr[ytr == 1] - mu).mean(0, keepdims=True)); res["centroid_centred"] = score_arm("centroid cosine, centred", (unit(Xte - mu) @ c_ctr.T)[:, 0])
for tag, A, B in [("raw", unit(Xtr), unit(Xte)), ("centred", unit(Xtr - mu), unit(Xte - mu))]:
    S = B @ A.T; nn = np.argsort(-S, axis=1)[:, :10]; res[f"knn10_{tag}"] = score_arm(f"10-NN vote, {tag}", ytr[nn].mean(1))
# classifier-based
def lr(A, B, C=0.3):
    m, s = A.mean(0), A.std(0) + 1e-6; clf = LogisticRegression(C=C, max_iter=5000, class_weight="balanced").fit((A - m) / s, ytr); return clf.decision_function((B - m) / s)
res["lr_float"] = score_arm("LR, float32 (768 x 4 B)", lr(Xtr, Xte))
res["lr_1bit_centred"] = score_arm("LR, 1-bit sign(centred) (96 B)", lr(np.sign(Xtr - mu), np.sign(Xte - mu)))
res["lr_1bit_centred_rot"] = score_arm("LR, 1-bit sign(rot(centred)) (96 B)", lr(np.sign((Xtr - mu) @ Q), np.sign((Xte - mu) @ Q)))
res["lr_1bit_raw"] = score_arm("LR, 1-bit sign(raw)", lr(np.sign(Xtr), np.sign(Xte)))
res["n_test"] = int(len(te)); res["n_test_pos"] = int(yte.sum()); res["corpus"] = corpus
json.dump(res, open(os.path.join(HERE, "results", "paper_s2_greedy.json"), "w"), indent=1); print("done -> results/paper_s2_greedy.json")
