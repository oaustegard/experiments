"""Citation-graph features for the modern population (PLAN-paper.md addendum 2).
Positives = data/papers/pmc_positives.jsonl, negatives = neighborspmc.jsonl, split 70/15/15 by PMID hash (as paper_pmc_probe.py).
Graph sources: s2_graph.jsonl (Semantic Scholar: cites, refs, authors, venue) and pubmed_links.jsonl (PubMed: refs, citedin).
Every "known MSD" set is built from TRAINING positives only. Arms: scalar graph features alone; sparse bags (refs, citers, authors);
SPECTER2 alone; SPECTER2 + scalars; everything. Plus a temporal holdout for the combined model. Writes results/paper_graph_probe.json."""
import json, os, sys, hashlib, time, collections
import numpy as np, scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(HERE, "data", "papers")
P = lambda f: [json.loads(l) for l in open(os.path.join(D, f)) if l.strip()]
G = {r["pmid"]: r for r in P("s2_graph.jsonl") if not r.get("missing")}
L = {r["pmid"]: r for r in P("pubmed_links.jsonl")}
E = {}
for l in open(os.path.join(D, "s2_specter2.jsonl")):
    r = json.loads(l); E[str(r["pmid"])] = np.array(r["vector"], dtype=np.float32)
pos = [r for r in P("pmc_positives.jsonl") if str(r["pmid"]) in E]; neg = [r for r in P("neighborspmc.jsonl") if str(r["pmid"]) in E]
h = lambda r: int(hashlib.sha1(("20260921" + str(r["pmid"])).encode()).hexdigest()[:8], 16) % 100
split = lambda rs: ([r for r in rs if h(r) < 70], [r for r in rs if 70 <= h(r) < 85], [r for r in rs if h(r) >= 85])
ptr, pdv, pte = split(pos); ntr, ndv, nte = split(neg)
tr = [(r, 1) for r in ptr] + [(r, 0) for r in ntr]; dv = [(r, 1) for r in pdv] + [(r, 0) for r in ndv]; te = [(r, 1) for r in pte] + [(r, 0) for r in nte]
ytr, ydv, yte = (np.array([y for _, y in s]) for s in (tr, dv, te))
pm = lambda r: str(r["pmid"])
# --- known-MSD sets from TRAINING positives only
known_pmid = {pm(r) for r in ptr}; known_s2 = {G[pm(r)]["s2_id"] for r in ptr if pm(r) in G and G[pm(r)].get("s2_id")}
known_authors = collections.Counter(a for r in ptr if pm(r) in G for a in G[pm(r)]["authors"])
ref_by_pos = collections.Counter(x for r in ptr if pm(r) in G for x in G[pm(r)]["refs"]); canon_s2 = {x for x, _ in ref_by_pos.most_common(100)}
ref_by_pos_pm = collections.Counter(x for r in ptr if pm(r) in L for x in L[pm(r)]["refs"]); canon_pm = {x for x, _ in ref_by_pos_pm.most_common(100)}
cov = {"s2_present": sum(pm(r) in G for r, _ in tr + dv + te), "s2_refs_nonempty": sum(bool(G[pm(r)]["refs"]) for r, _ in tr + dv + te if pm(r) in G), "pm_refs_nonempty": sum(bool(L.get(pm(r), {}).get("refs")) for r, _ in tr + dv + te),
       "s2_cites_nonempty": sum(bool(G[pm(r)]["cites"]) for r, _ in tr + dv + te if pm(r) in G), "pm_citedin_nonempty": sum(bool(L.get(pm(r), {}).get("citedin")) for r, _ in tr + dv + te), "n": len(tr + dv + te)}
print("coverage:", cov)

def scalars(r):
    g = G.get(pm(r), {}); l = L.get(pm(r), {}); s2refs = g.get("refs") or []; s2cites = g.get("cites") or []; auth = g.get("authors") or []; pmrefs = l.get("refs") or []; pmcit = l.get("citedin") or []
    f = [len(s2refs), len(s2cites), len(auth), len(pmrefs), len(pmcit),
         sum(x in known_s2 for x in s2refs), sum(x in known_s2 for x in s2refs) / max(1, len(s2refs)),
         sum(x in known_s2 for x in s2cites), sum(x in known_s2 for x in s2cites) / max(1, len(s2cites)),
         sum(x in known_pmid for x in pmrefs), sum(x in known_pmid for x in pmrefs) / max(1, len(pmrefs)),
         sum(x in known_pmid for x in pmcit), sum(x in known_pmid for x in pmcit) / max(1, len(pmcit)),
         sum(known_authors.get(a, 0) > 0 for a in auth), sum(known_authors.get(a, 0) > 0 for a in auth) / max(1, len(auth)), max([known_authors.get(a, 0) for a in auth] + [0]),
         sum(x in canon_s2 for x in s2refs), sum(x in canon_pm for x in pmrefs)]
    return f
NAMES = ["n_s2refs", "n_s2cites", "n_authors", "n_pmrefs", "n_pmcitedin", "refs_to_known", "frac_refs_known", "citers_known", "frac_citers_known", "pmrefs_to_known", "frac_pmrefs_known", "pmcitedin_known", "frac_pmcitedin_known", "authors_known", "frac_authors_known", "max_author_msd_papers", "refs_to_canon_s2", "refs_to_canon_pm"]
S = lambda rows: np.array([scalars(r) for r, _ in rows], dtype=np.float32)
Str, Sdv, Ste = S(tr), S(dv), S(te); Str_l, Sdv_l, Ste_l = np.log1p(Str), np.log1p(Sdv), np.log1p(Ste)
def auc(y, s):
    o = np.argsort(s, kind="stable"); rk = np.empty(len(s)); rk[o] = np.arange(1, len(s) + 1); n1 = y.sum(); return float((rk[y == 1].sum() - n1 * (n1 + 1) / 2) / max(1, n1 * (len(y) - n1)))
def r5(y, s):
    ng = np.sort(s[y == 0])[::-1]; return float((s[y == 1] >= ng[int(0.05 * len(ng))]).mean())
res = {"coverage": cov, "arms": {}}
def fit_lr(Xa, Xb, Cs=(0.01, 0.03, 0.1, 0.3, 1.0), Xd=None):
    best = None
    for C in Cs:
        clf = LogisticRegression(C=C, max_iter=5000, class_weight="balanced").fit(Xa, ytr); a = auc(ydv, clf.decision_function(Xd if Xd is not None else Xa[:0]))
        if best is None or a > best[0]: best = (a, C, clf)
    return best[2]
def arm(name, Xa, Xd, Xb, note=""):
    clf = fit_lr(Xa, Xb, Xd=Xd); s = clf.decision_function(Xb); res["arms"][name] = {"auc": auc(yte, s), "recall_at_fpr5": r5(yte, s), "note": note}
    print(f"{name:40s} AUC {res['arms'][name]['auc']:.3f}  recall@5%FPR {res['arms'][name]['recall_at_fpr5']:.3f} {note}"); return s
# single scalar features
print("single scalar features (AUC on test):")
for j, n in enumerate(NAMES):
    a = auc(yte, Ste[:, j]); res["arms"][f"scalar:{n}"] = {"auc": a}; print(f"  {n:24s} {a:.3f}")
m, sd = Str_l.mean(0), Str_l.std(0) + 1e-6; Z = lambda A: (A - m) / sd
s_scal = arm("all graph scalars (LR)", Z(Str_l), Z(Sdv_l), Z(Ste_l))
# sparse bags
def bag(key, src, min_df=2):
    vocab = collections.Counter(x for r, _ in tr for x in (src(r) or [])); keep = {x: i for i, x in enumerate(v for v, c in vocab.items() if c >= min_df)}
    def M(rows):
        rows_i, cols = [], []
        for i, (r, _) in enumerate(rows):
            for x in set(src(r) or []):
                if x in keep: rows_i.append(i); cols.append(keep[x])
        return sp.csr_matrix((np.ones(len(rows_i), dtype=np.float32), (rows_i, cols)), shape=(len(rows), len(keep)))
    return M(tr), M(dv), M(te), len(keep)
bags = {}
for name, src in [("bag of S2 references", lambda r: G.get(pm(r), {}).get("refs")), ("bag of S2 citers", lambda r: G.get(pm(r), {}).get("cites")), ("bag of PubMed references", lambda r: L.get(pm(r), {}).get("refs")), ("bag of PubMed citers", lambda r: L.get(pm(r), {}).get("citedin")), ("bag of authors", lambda r: G.get(pm(r), {}).get("authors")), ("venue", lambda r: [G.get(pm(r), {}).get("venue") or "?"])]:
    Btr, Bdv, Bte, V = bag(name, src); bags[name] = (Btr, Bdv, Bte)
    if V == 0: print(f"{name}: empty"); continue
    arm(name, Btr, Bdv, Bte, note=f"({V} columns)")
# SPECTER2 alone and combined
X = lambda rows: np.stack([E[pm(r)] for r, _ in rows]); Xtr, Xdv, Xte = X(tr), X(dv), X(te); mx, sx = Xtr.mean(0), Xtr.std(0) + 1e-6; ZX = lambda A: (A - mx) / sx
s_emb = arm("SPECTER2 alone (LR)", ZX(Xtr), ZX(Xdv), ZX(Xte))
s_comb = arm("SPECTER2 + graph scalars", np.hstack([ZX(Xtr), Z(Str_l)]), np.hstack([ZX(Xdv), Z(Sdv_l)]), np.hstack([ZX(Xte), Z(Ste_l)]))
allB = lambda i: sp.hstack([bags[k][i] for k in ("bag of S2 references", "bag of S2 citers", "bag of PubMed references", "bag of PubMed citers", "bag of authors") if k in bags]).tocsr()
s_all = arm("SPECTER2 + scalars + all bags", sp.hstack([sp.csr_matrix(np.hstack([ZX(Xtr), Z(Str_l)])), allB(0)]).tocsr(), sp.hstack([sp.csr_matrix(np.hstack([ZX(Xdv), Z(Sdv_l)])), allB(1)]).tocsr(), sp.hstack([sp.csr_matrix(np.hstack([ZX(Xte), Z(Ste_l)])), allB(2)]).tocsr())
s_bags = arm("all bags only (no embedding)", allB(0), allB(1), allB(2))
# temporal holdout of SPECTER2 + scalars
yr = lambda r: int(str(r.get("year") or 0)[:4])
A = [(r, y) for r, y in tr + dv + te if yr(r) <= 2023]; B = [(r, y) for r, y in tr + dv + te if yr(r) >= 2024]
if len(B) > 100:
    # rebuild known sets from A's positives only
    kp = {pm(r) for r, y in A if y == 1}; ks = {G[pm(r)]["s2_id"] for r, y in A if y == 1 and pm(r) in G and G[pm(r)].get("s2_id")}; ka = collections.Counter(a for r, y in A if y == 1 and pm(r) in G for a in G[pm(r)]["authors"])
    def sc2(r):
        g = G.get(pm(r), {}); l = L.get(pm(r), {}); refs = g.get("refs") or []; cites = g.get("cites") or []; auth = g.get("authors") or []; pr = l.get("refs") or []; pc = l.get("citedin") or []
        return [len(refs), len(cites), len(auth), len(pr), len(pc), sum(x in ks for x in refs), sum(x in ks for x in refs) / max(1, len(refs)), sum(x in ks for x in cites), sum(x in ks for x in cites) / max(1, len(cites)), sum(x in kp for x in pr), sum(x in kp for x in pr) / max(1, len(pr)), sum(x in kp for x in pc), sum(x in kp for x in pc) / max(1, len(pc)), sum(ka.get(a, 0) > 0 for a in auth), sum(ka.get(a, 0) > 0 for a in auth) / max(1, len(auth)), max([ka.get(a, 0) for a in auth] + [0]), 0, 0]
    SA, SB = np.log1p(np.array([sc2(r) for r, _ in A], dtype=np.float32)), np.log1p(np.array([sc2(r) for r, _ in B], dtype=np.float32)); XA, XB = np.stack([E[pm(r)] for r, _ in A]), np.stack([E[pm(r)] for r, _ in B]); yA, yB = np.array([y for _, y in A]), np.array([y for _, y in B])
    FA, FB = np.hstack([XA, SA]), np.hstack([XB, SB]); mA, sA = FA.mean(0), FA.std(0) + 1e-6
    for nm, (fa, fb) in {"temporal: SPECTER2 + scalars": (FA, FB), "temporal: scalars only": (SA, SB)}.items():
        ma_, sa_ = fa.mean(0), fa.std(0) + 1e-6; clf = LogisticRegression(C=0.1, max_iter=5000, class_weight="balanced").fit((fa - ma_) / sa_, yA); s = clf.decision_function((fb - ma_) / sa_)
        res["arms"][nm] = {"auc": auc(yB, s), "recall_at_fpr5": r5(yB, s), "train_n": len(A), "test_n": len(B)}; print(f"{nm:40s} AUC {res['arms'][nm]['auc']:.3f}  recall@5%FPR {res['arms'][nm]['recall_at_fpr5']:.3f} (train {len(A)}, test {len(B)})")
json.dump(res, open(os.path.join(HERE, "results", "paper_graph_probe.json"), "w"), indent=1); print("done -> results/paper_graph_probe.json")
