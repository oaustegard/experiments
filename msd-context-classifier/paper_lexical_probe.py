"""Lexical baselines for the paper filter, on the same populations and PMID-hash split as paper_pmc_probe.py:
TF-IDF (1-2 grams) + logistic regression, and BM25 max-score against the training positives (query = test title+abstract),
against SPECTER2 + LR on the identical rows. Populations: modern (PMC positives vs their PubMed neighbours) and
bibliography (curated positives vs neighbours; vs hard query negatives). Writes results/paper_lexical_probe.json."""
import json, os, hashlib, re, collections
import numpy as np, scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(HERE, "data", "papers")
P = lambda f: [json.loads(l) for l in open(os.path.join(D, f)) if l.strip()]
E = {}
for l in open(os.path.join(D, "s2_specter2.jsonl")):
    r = json.loads(l); E[str(r["pmid"])] = np.array(r["vector"], dtype=np.float32)
h = lambda r: int(hashlib.sha1(("20260921" + str(r["pmid"])).encode()).hexdigest()[:8], 16) % 100
txt = lambda r: (r.get("title") or "") + " " + (r.get("abstract") or "")
def auc(y, sc):
    o = np.argsort(sc, kind="stable"); rk = np.empty(len(sc)); rk[o] = np.arange(1, len(sc) + 1); n1 = y.sum(); return float((rk[y == 1].sum() - n1 * (n1 + 1) / 2) / max(1, n1 * (len(y) - n1)))
def r5(y, sc): ng = np.sort(sc[y == 0])[::-1]; return float((sc[y == 1] >= ng[int(0.05 * len(ng))]).mean())
def f95(y, sc): ps = np.sort(sc[y == 1])[::-1]; return float((sc[y == 0] >= ps[int(np.ceil(0.95 * len(ps))) - 1]).mean())
def bm25_max(train_docs, test_docs, k1=1.2, b=0.75):
    cv = CountVectorizer(token_pattern=r"(?u)\b\w[\w\-]+\b", lowercase=True, min_df=2); Xtr = cv.fit_transform(train_docs).tocsc().astype(np.float32); Xte = cv.transform(test_docs)
    N = Xtr.shape[0]; dl = np.asarray(Xtr.sum(1)).ravel(); avg = dl.mean(); df = np.asarray((Xtr > 0).sum(0)).ravel(); idf = np.log((N - df + 0.5) / (df + 0.5) + 1)
    Xtr = Xtr.tocoo(); w = Xtr.data * (k1 + 1) / (Xtr.data + k1 * (1 - b + b * dl[Xtr.row] / avg)) * idf[Xtr.col]
    W = sp.csr_matrix((w, (Xtr.row, Xtr.col)), shape=Xtr.shape)  # BM25 weight of each train doc's terms
    Q = (Xte > 0).astype(np.float32)  # query = set of test terms
    S = (Q @ W.T).toarray(); return S.max(1), np.sort(S, 1)[:, -5:].mean(1)
def run(name, pos, neg):
    pos = [r for r in pos if str(r["pmid"]) in E]; neg = [r for r in neg if str(r["pmid"]) in E]
    tr = [(r, 1) for r in pos if h(r) < 70] + [(r, 0) for r in neg if h(r) < 70]; te = [(r, 1) for r in pos if h(r) >= 85] + [(r, 0) for r in neg if h(r) >= 85]
    ytr = np.array([y for _, y in tr]); yte = np.array([y for _, y in te]); out = {"n_train": len(tr), "n_test": len(te), "test_pos": int(yte.sum()), "arms": {}}
    print(f"\n{name}: train {len(tr)}, test {len(te)} ({int(yte.sum())} pos)")
    def rep(nm, sc): out["arms"][nm] = {"auc": auc(yte, sc), "recall_at_fpr5": r5(yte, sc), "fpr_at_recall95": f95(yte, sc)}; print(f"  {nm:34s} AUC {out['arms'][nm]['auc']:.3f} | recall@5%FPR {out['arms'][nm]['recall_at_fpr5']:.3f} | FPR@95%recall {out['arms'][nm]['fpr_at_recall95']:.3f}")
    X = lambda rows: np.stack([E[str(r["pmid"])] for r, _ in rows]); Xtr, Xte = X(tr), X(te); m, s = Xtr.mean(0), Xtr.std(0) + 1e-6
    best = max(((auc(ytr, c.decision_function((Xtr - m) / s)), C, c) for C in (0.03, 0.1, 0.3) for c in [LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit((Xtr - m) / s, ytr)]), key=lambda t: t[0])
    rep("SPECTER2 + LR", best[2].decision_function((Xte - m) / s))
    dtr = [txt(r) for r, _ in tr]; dte = [txt(r) for r, _ in te]
    for ng, nm in ((1, 1), (1, 2)):
        tv = TfidfVectorizer(ngram_range=(ng, nm), min_df=2, sublinear_tf=True, token_pattern=r"(?u)\b\w[\w\-]+\b"); Ttr = tv.fit_transform(dtr); Tte = tv.transform(dte)
        bst = max(((auc(ytr, c.decision_function(Ttr)), c) for C in (0.3, 1, 3, 10) for c in [LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(Ttr, ytr)]), key=lambda t: t[0])
        rep(f"TF-IDF {ng}-{nm}gram + LR ({Ttr.shape[1]} feats)", bst[1].decision_function(Tte))
    trpos = [d for d, (_, y) in zip(dtr, tr) if y == 1]; mx, top5 = bm25_max(trpos, dte)
    rep("BM25 max vs training positives", mx); rep("BM25 mean of top-5", top5)
    # both: rank-average of SPECTER2 LR and TF-IDF LR
    a = best[2].decision_function((Xte - m) / s); b_ = bst[1].decision_function(Tte); ra = lambda v: np.argsort(np.argsort(v)) / len(v); rep("rank-average SPECTER2 + TF-IDF", ra(a) + ra(b_))
    # cue-free subset
    from paper_corpus import has_cue
    cf = np.array([not has_cue(r.get("title"), r.get("abstract")) for r, _ in te]); sc = best[2].decision_function((Xte - m) / s); sc2 = bst[1].decision_function(Tte)
    out["cue_free_test_pos"] = int((cf & (yte == 1)).sum()); out["cue_free_auc"] = {"SPECTER2": auc(yte[cf], sc[cf]), "TF-IDF": auc(yte[cf], sc2[cf])}
    print(f"  cue-free rows only ({int((cf & (yte==1)).sum())} pos): SPECTER2 AUC {out['cue_free_auc']['SPECTER2']:.3f}, TF-IDF AUC {out['cue_free_auc']['TF-IDF']:.3f}")
    return out
res = {"modern: PMC positives vs PubMed neighbours": run("modern population", P("pmc_positives.jsonl"), P("neighborspmc.jsonl")),
       "bibliography vs PubMed neighbours": run("bibliography vs neighbours", P("positives.jsonl"), P("neighbors.jsonl")),
       "bibliography vs hard query negatives": run("bibliography vs hard negatives", P("positives.jsonl"), P("hard.jsonl"))}
json.dump(res, open(os.path.join(HERE, "results", "paper_lexical_probe.json"), "w"), indent=1); print("done -> results/paper_lexical_probe.json")
