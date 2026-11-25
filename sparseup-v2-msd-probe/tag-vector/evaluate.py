"""Retrieval eval: tag-binary vs BM25 vs word+char TF-IDF vs RRF fusion over the PubMed fixture.

Usage: python3 evaluate.py SPLIT     (SPLIT = dev | test)
Reads data/vocab.json, data/fixture.json; writes results/results_SPLIT.json.
"""
from __future__ import annotations

import json
import re
import sys

import numpy as np
from rank_bm25 import BM25Okapi
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from tagger import Tagger

SPLIT = sys.argv[1] if len(sys.argv) > 1 else "dev"
K, RRF_K, BOOT, SEED = 10, 60, 2000, 20261125
rng = np.random.default_rng(SEED)

vocab = json.load(open("data/vocab.json", encoding="utf-8"))
fx = json.load(open("data/fixture.json", encoding="utf-8"))
docs = fx["docs"]
texts = [d["title"] + " " + d["abstract"] for d in docs]
N = len(docs)
tagger = Tagger(vocab["alias"], set(vocab["block"]), set(vocab.get("case_free", [])))
doc_tags = [set(tagger.tag(t)) for t in texts]
tie = rng.permutation(N).astype(float) * 1e-9          # fixed arbitrary tie-break

# queries: one per (concept, surface form) for concepts in this split
queries = []
for c in fx["concepts"]:
    if c["split"] != SPLIT:
        continue
    rel = {i for i, d in enumerate(docs) if c["mesh"] in d["mesh"]}
    if not rel:
        continue
    for form in ("catalog", "symbol", "mesh"):
        q = c["forms"].get(form)
        if q:
            queries.append({"concept": c["id"], "form": form, "text": q, "rel": rel})

tok = lambda s: re.findall(r"\w+", s.lower())
bm25 = BM25Okapi([tok(t) for t in texts])
wv = TfidfVectorizer(analyzer="word", token_pattern=r"\w+", sublinear_tf=True, ngram_range=(1, 2), min_df=1)
cv = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2)
Dw, Dc = wv.fit_transform(texts), cv.fit_transform(texts)
Dtf = normalize(hstack([Dw, Dc]).tocsr())


def s_tag(q, mode):
    Q = set(tagger.tag(q))
    if not Q:
        return np.zeros(N) + tie, False
    inter = np.array([len(Q & d) for d in doc_tags], float)
    if mode == "dot":
        return inter + tie, True
    union = np.array([len(Q | d) for d in doc_tags], float)
    return inter / union + tie, True


def s_bm25(q):
    return np.asarray(bm25.get_scores(tok(q))) + tie


def s_tfidf(q):
    v = normalize(hstack([wv.transform([q]), cv.transform([q])]).tocsr())
    return (Dtf @ v.T).toarray().ravel() + tie


def ranks(s):
    order = np.argsort(-s, kind="stable")
    r = np.empty(N, int)
    r[order] = np.arange(1, N + 1)
    return r


def rrf(*scores):
    return sum(1.0 / (RRF_K + ranks(s)) for s in scores) + tie


def metrics(s, rel):
    order = np.argsort(-s, kind="stable")
    top = order[:K]
    hits = sum(1 for i in top if i in rel)
    first = next(r for r, i in enumerate(order, 1) if i in rel)
    return hits / min(len(rel), K), 1.0 / first


rows = []
for q in queries:
    st, covered = s_tag(q["text"], "jaccard")
    sd, _ = s_tag(q["text"], "dot")
    sb, sf = s_bm25(q["text"]), s_tfidf(q["text"])
    arms = {"tag_jaccard": st, "tag_dot": sd, "bm25": sb, "tfidf_wc": sf,
            "rrf_tag_bm25": rrf(st, sb), "rrf_tag_tfidf": rrf(st, sf), "rrf_bm25_tfidf": rrf(sb, sf)}
    row = {"concept": q["concept"], "form": q["form"], "text": q["text"], "n_rel": len(q["rel"]),
           "tag_covered": covered, "q_tags": sorted(set(tagger.tag(q["text"])))}
    for a, s in arms.items():
        row[a] = dict(zip(("r10", "mrr"), metrics(s, q["rel"])))
    rows.append(row)

ARMS = ["tag_jaccard", "tag_dot", "bm25", "tfidf_wc", "rrf_tag_bm25", "rrf_tag_tfidf", "rrf_bm25_tfidf"]


def boot(vals):
    vals = np.asarray(vals)
    idx = rng.integers(0, len(vals), (BOOT, len(vals)))
    m = vals[idx].mean(1)
    return [round(float(vals.mean()), 3), round(float(np.percentile(m, 2.5)), 3), round(float(np.percentile(m, 97.5)), 3)]


summary = {}
for form in ("catalog", "symbol", "mesh", "all"):
    R = [r for r in rows if form == "all" or r["form"] == form]
    if not R:
        continue
    out = {"n_queries": len(R), "tag_coverage": round(float(np.mean([r["tag_covered"] for r in R])), 3)}
    for a in ARMS:
        out[a] = {m: boot([r[a][m] for r in R]) for m in ("r10", "mrr")}
    diffs = {}
    for a, b in (("tag_jaccard", "bm25"), ("tag_jaccard", "tfidf_wc"), ("rrf_tag_bm25", "bm25"),
                 ("rrf_tag_tfidf", "tfidf_wc"), ("rrf_tag_tfidf", "rrf_bm25_tfidf")):
        diffs[f"{a}-{b}"] = {m: boot([r[a][m] - r[b][m] for r in R]) for m in ("r10", "mrr")}
    out["paired_diffs"] = diffs
    summary[form] = out

json.dump({"split": SPLIT, "n_docs": N, "n_concepts": len({r['concept'] for r in rows}),
           "doc_tag_mean": round(float(np.mean([len(t) for t in doc_tags])), 2),
           "docs_with_no_tag": int(sum(1 for t in doc_tags if not t)),
           "summary": summary, "rows": rows}, open(f"results/results_{SPLIT}.json", "w"), indent=1, ensure_ascii=False)
for form, o in summary.items():
    print(form, o["n_queries"], "cov", o["tag_coverage"],
          " ".join(f"{a}={o[a]['r10'][0]:.3f}/{o[a]['mrr'][0]:.3f}" for a in ARMS))
