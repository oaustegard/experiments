"""Round 3: retrieval in the expanded tag space against text representations.

Queries: the 235 fixture memories whose `refs` cite other fixture memories; relevant =
the cited memories. Corpus: the other 3,456. Arms in PLAN.md round 3.
Writes results/retrieval.json.
"""
import json, math, sys, time
from datetime import datetime
import numpy as np
import scipy.sparse as sp
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from common import DATA, load_fixture

RESULTS = DATA.parent / "results"
ALPHAS = (0.25, 0.5, 1.0)
TOPN, MIN_PAIR, KS = 20, 2, (10, 50)
SEED = 0


def recall_at(ranked, rel, k):
    return len(set(ranked[:k]) & rel) / len(rel)


def mrr(ranked, rel):
    for i, d in enumerate(ranked):
        if d in rel:
            return 1.0 / (i + 1)
    return 0.0


def rank_rows(score_rows, qidx):
    """score_rows: (nq, n) dense scores; the query's own column is masked."""
    out = []
    for r, qi in enumerate(qidx):
        s = np.asarray(score_rows[r]).ravel().astype(np.float64).copy()
        s[qi] = -np.inf
        out.append(np.argsort(-s, kind="stable"))
    return out


def evaluate(rankings, rels):
    m = {f"recall@{k}": float(np.mean([recall_at(r, rel, k) for r, rel in zip(rankings, rels)])) for k in KS}
    m["mrr"] = float(np.mean([mrr(r, rel) for r, rel in zip(rankings, rels)]))
    return m


def per_query(rankings, rels, k=10):
    return np.array([recall_at(r, rel, k) for r, rel in zip(rankings, rels)]), np.array([mrr(r, rel) for r, rel in zip(rankings, rels)])


def boot_diff(a, b, n_boot=1000, seed=SEED):
    rng = np.random.RandomState(seed); n = len(a); d = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.randint(0, n, n); d[i] = a[idx].mean() - b[idx].mean()
    return {"mean": float((a - b).mean()), "lo": float(np.percentile(d, 2.5)), "hi": float(np.percentile(d, 97.5))}


def main():
    t0 = time.time()
    fx = load_fixture(); mems = fx["memories"]; ids = [m["id"] for m in mems]; pos = {m: i for i, m in enumerate(ids)}
    n = len(ids)
    links = json.loads((DATA / "links.json").read_text())
    rel_of = {}
    for q, t in links["pairs"]:
        rel_of.setdefault(q, set()).add(pos[t])
    qids = sorted(rel_of); qidx = [pos[q] for q in qids]; rels = [rel_of[q] for q in qids]
    print(f"{len(qids)} queries, {sum(len(r) for r in rels)} relevant pairs, corpus {n}", file=sys.stderr)

    # ---- tag space
    vocab = sorted({t for m in mems for t in m["tags"]}); vi = {t: j for j, t in enumerate(vocab)}
    rows, cols = zip(*[(i, vi[t]) for i, m in enumerate(mems) for t in m["tags"]])
    T = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(n, len(vocab)))
    df = np.asarray(T.sum(0)).ravel()
    Cm = (T.T @ T).tocsr()  # pair counts, diag = df
    N = float(n)

    def pmi_expansion(tags_idx, loo=False):
        """weights {tag_j: pmi_norm} for the union of top-N co-occurring tags of tags_idx."""
        exp = {}
        for a in tags_idx:
            row = Cm.getrow(a); js, cs = row.indices, row.data.copy()
            dfa = df[a] - (1 if loo else 0); NN = N - (1 if loo else 0)
            for j, c in zip(js, cs):
                if j == a: continue
                dfj = df[j] - (1 if (loo and j in tags_idx) else 0)
                if loo and j in tags_idx: c -= 1
                if c < MIN_PAIR or dfa <= 0 or dfj <= 0: continue
                p = math.log((c * NN) / (dfa * dfj))
                if p > 0: exp[j] = max(exp.get(j, 0.0), p)
        top = sorted(exp.items(), key=lambda x: -x[1])[:TOPN * max(1, len(tags_idx))]
        return dict(top)
    doc_tags = [set(T.getrow(i).indices.tolist()) for i in range(n)]
    all_exp = [pmi_expansion(doc_tags[i]) for i in range(n)]
    pmi_max = max((w for e in all_exp for w in e.values()), default=1.0)
    q_exp = {qi: pmi_expansion(doc_tags[qi], loo=True) for qi in qidx}

    def expanded_matrix(alpha, override=None):
        r, c, v = [], [], []
        for i in range(n):
            e = override.get(i, all_exp[i]) if override else all_exp[i]
            vec = {j: 1.0 for j in doc_tags[i]}
            for j, w in e.items():
                vec[j] = max(vec.get(j, 0.0), alpha * w / pmi_max)
            for j, w in vec.items():
                r.append(i); c.append(j); v.append(w)
        return normalize(sp.csr_matrix((np.array(v, np.float32), (r, c)), shape=(n, len(vocab))))

    arms = {}
    Tn = normalize(T)
    arms["tag-binary"] = rank_rows((Tn[qidx] @ Tn.T).toarray(), qidx)
    for a in ALPHAS:
        Ed = expanded_matrix(a)                       # corpus side: global PMI
        Eq = expanded_matrix(a, override=q_exp)[qidx]  # query side: leave-one-out PMI
        arms[f"tag-expanded a={a}"] = rank_rows((Eq @ Ed.T).toarray(), qidx)
    exp_stats = {"vocab": len(vocab), "mean_tags_per_doc": float(T.sum(1).mean()),
                 "mean_expansion_terms_per_doc": float(np.mean([len(e) for e in all_exp])), "pmi_max": float(pmi_max)}

    # ---- text spaces
    D = np.load(DATA / "dense.npy"); arms["gte-small"] = rank_rows(D[qidx] @ D.T, qidx)
    tj = json.loads((DATA / "texts.json").read_text()); assert tj["ids"] == ids; texts = tj["texts"]
    vw = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2)
    vc = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2)
    X = normalize(hstack([vw.fit_transform(texts), vc.fit_transform(texts)]).tocsr())
    arms["tfidf word+char"] = rank_rows((X[qidx] @ X.T).toarray(), qidx)
    S = sp.load_npz(DATA / "sparse.npz").tocsr()
    arms["sparseup doc-doc"] = rank_rows((S[qidx] @ S.T).toarray(), qidx)
    if (DATA / "sparse_queries.npz").exists():
        Q = sp.load_npz(DATA / "sparse_queries.npz").tocsr(); qorder = json.loads((DATA / "sparse_queries_ids.json").read_text())
        assert qorder == qids
        arms["sparseup query-doc"] = rank_rows((Q @ S.T).toarray(), qidx)

    # ---- fusion and controls
    def rrf(r1, r2, k=60):
        out = []
        for a, b in zip(r1, r2):
            s = np.zeros(n); s[a] += 1 / (k + np.arange(1, n + 1)); s[b] += 1 / (k + np.arange(1, n + 1))
            out.append(np.argsort(-s, kind="stable"))
        return out
    arms["rrf(tag-expanded a=0.5, gte)"] = rrf(arms["tag-expanded a=0.5"], arms["gte-small"])
    arms["rrf(tfidf, gte)"] = rrf(arms["tfidf word+char"], arms["gte-small"])
    rng = np.random.RandomState(SEED)
    arms["random"] = [np.array([j for j in rng.permutation(n) if j != qi]) for qi in qidx]
    def ts(s):
        try: return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
        except Exception: return 0.0
    tt = np.array([ts(links["t"][m]) for m in ids])
    arms["nearest in time"] = rank_rows(-np.abs(tt[qidx][:, None] - tt[None, :]), qidx)

    # ---- metrics
    metrics = {name: evaluate(r, rels) for name, r in arms.items()}
    g10, gm = per_query(arms["gte-small"], rels)
    boots = {}
    for name, r in arms.items():
        if name == "gte-small": continue
        a10, am = per_query(r, rels)
        boots[name] = {"recall@10": boot_diff(a10, g10), "mrr": boot_diff(am, gm)}
    # overlap: relevant hits found by tag-expanded@10 and missed by gte@10
    te = arms["tag-expanded a=0.5"]; ge = arms["gte-small"]
    gte_miss = tag_rescue = tfidf_rescue = 0; jacc = []
    for r_te, r_ge, r_tf, rel in zip(te, ge, arms["tfidf word+char"], rels):
        miss = rel - set(r_ge[:10]); gte_miss += len(miss)
        tag_rescue += len(miss & set(r_te[:10])); tfidf_rescue += len(miss & set(r_tf[:10]))
        jacc.append(len(set(r_te[:10]) & set(r_ge[:10])) / len(set(r_te[:10]) | set(r_ge[:10])))
    overlap = {"gte_misses_at_10": gte_miss, "rescued_by_tag_expanded_at_10": tag_rescue,
               "rescued_by_tfidf_at_10": tfidf_rescue, "mean_top10_jaccard_tag_vs_gte": float(np.mean(jacc))}

    # ---- refutation check from PLAN round 3: recency. Re-score on the half of the relevant
    # pairs whose query-to-cited age gap is above the median.
    gaps = {(pos[q], pos[t]): abs(tt[pos[q]] - tt[pos[t]]) / 86400 for q, t in links["pairs"]}
    med = float(np.median(list(gaps.values())))
    far_rel = {}
    for (qi, ti), g in gaps.items():
        if g > med: far_rel.setdefault(qi, set()).add(ti)
    far_q = sorted(far_rel); far_rels = [far_rel[q] for q in far_q]
    qrow = {qi: r for r, qi in enumerate(qidx)}
    distant = {"median_gap_days": med, "n_queries": len(far_q), "n_pairs": sum(len(r) for r in far_rels), "metrics": {}, "bootstrap_vs_gte": {}}
    g10d, _ = per_query([arms["gte-small"][qrow[q]] for q in far_q], far_rels)
    for name, r in arms.items():
        rr = [r[qrow[q]] for q in far_q]
        distant["metrics"][name] = evaluate(rr, far_rels)
        a10, _ = per_query(rr, far_rels)
        distant["bootstrap_vs_gte"][name] = boot_diff(a10, g10d)

    out = {"n_queries": len(qids), "n_relevant_pairs": sum(len(r) for r in rels), "corpus": n, "distant_half": distant,
           "tag_space": exp_stats, "metrics": metrics, "bootstrap_vs_gte": boots, "overlap": overlap,
           "wall_seconds": round(time.time() - t0)}
    RESULTS.mkdir(exist_ok=True); (RESULTS / "retrieval.json").write_text(json.dumps(out, indent=1))
    print(f"\n{'arm':32s} R@10    R@50    MRR")
    for name, m in metrics.items():
        print(f"{name:32s} {m['recall@10']:.3f}   {m['recall@50']:.3f}   {m['mrr']:.3f}")
    print("\nvs gte-small, recall@10 paired diff [95% CI]:")
    for name, b in boots.items():
        print(f"  {name:30s} {b['recall@10']['mean']:+.3f} [{b['recall@10']['lo']:+.3f}, {b['recall@10']['hi']:+.3f}]")
    print("\noverlap:", json.dumps(overlap)); print("tag space:", json.dumps(exp_stats))
    print(f"\ndistant half (age gap > {med:.1f} d): {distant['n_queries']} queries, {distant['n_pairs']} pairs")
    for name, m in distant["metrics"].items():
        b = distant["bootstrap_vs_gte"][name]
        print(f"  {name:30s} R@10 {m['recall@10']:.3f}  MRR {m['mrr']:.3f}  vs gte {b['mean']:+.3f} [{b['lo']:+.3f}, {b['hi']:+.3f}]")


if __name__ == "__main__":
    main()
