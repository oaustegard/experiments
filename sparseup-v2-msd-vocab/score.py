"""Score the encodes train.py wrote (OUT/<tag>_fx_*.npz, <tag>_sf_*.npz) on the MSD fixture and SciFact.

    python3 score.py OUT [tag ...]      # default: every tag found; -> results/<OUT name>.json

MSD fixture: the frozen ../sparseup-v2-msd-probe/tag-vector/evaluate.py runs unchanged (runpy) for its queries,
tie-break, metrics and baseline arms (tag_jaccard, bm25, tfidf_wc), on the test split; each sparse encode is
scored the same way, alone and fused with the tag vector and TF-IDF (RRF, k = 60). SciFact: nDCG@10 over the
300 test queries, as in ../sparseup-v2/pack/scifact.py (identical doc ids skipped).
"""
from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
TV = HERE.parent / "sparseup-v2-msd-probe" / "tag-vector"
OUT = Path(sys.argv[1])
tags = sys.argv[2:] or sorted({p.name.split("_fx_")[0] for p in OUT.glob("*_fx_docs.npz")})

sys.argv = ["evaluate.py", "test"]
cwd = Path.cwd()
import os

os.chdir(TV)
sys.path.insert(0, str(TV))
E = runpy.run_path(str(TV / "evaluate.py"), run_name="__main__")
os.chdir(cwd)
queries, tie, metrics, rrf = E["queries"], E["tie"], E["metrics"], E["rrf"]
fxq = {q: i for i, q in enumerate(json.loads((OUT / "fx_queries.json").read_text()))}
rng = np.random.default_rng(20261125)


def boot(vals):
    vals = np.asarray(vals)
    m = vals[rng.integers(0, len(vals), (2000, len(vals)))].mean(1)
    return [round(float(vals.mean()), 4), round(float(np.percentile(m, 2.5)), 4), round(float(np.percentile(m, 97.5)), 4)]


base = {q["text"]: {"tag": E["s_tag"](q["text"], "jaccard")[0], "tfidf": E["s_tfidf"](q["text"]), "bm25": E["s_bm25"](q["text"])} for q in queries}
res = {"fixture": {}, "scifact": {}}
per_query = {}
for t in tags:
    D, Q = sp.load_npz(OUT / f"{t}_fx_docs.npz"), sp.load_npz(OUT / f"{t}_fx_queries.npz")
    assert D.shape[0] == len(tie), (D.shape, len(tie))
    rows = []
    for q in queries:
        s = (D @ Q[fxq[q["text"]]].T).toarray().ravel() + tie
        b = base[q["text"]]
        arms = {"alone": s, "rrf_tag": rrf(s, b["tag"]), "rrf_tag_tfidf": rrf(s, b["tag"], b["tfidf"])}
        rows.append({"form": q["form"], "text": q["text"], **{a: metrics(v, q["rel"]) for a, v in arms.items()}})
    per_query[t] = rows
    out = {"doc_nnz": round(D.getnnz() / D.shape[0], 1), "query_nnz": round(Q.getnnz() / Q.shape[0], 1)}
    for form in ("catalog", "symbol", "mesh", "all"):
        R = [r for r in rows if form == "all" or r["form"] == form]
        out[form] = {a: {"r10": boot([r[a][0] for r in R]), "mrr": boot([r[a][1] for r in R])} for a in ("alone", "rrf_tag", "rrf_tag_tfidf")}
    res["fixture"][t] = out

# paired differences against the untouched model (s0_norewrite) where present
ref_t = "s0_norewrite"
if ref_t in per_query:
    for t, rows_t in per_query.items():
        if t == ref_t:
            continue
        d = {}
        for form in ("catalog", "symbol", "mesh", "all"):
            pairs = [(a, b) for a, b in zip(rows_t, per_query[ref_t]) if form == "all" or a["form"] == form]
            d[form] = {"r10": boot([a["alone"][0] - b["alone"][0] for a, b in pairs]),
                       "mrr": boot([a["alone"][1] - b["alone"][1] for a, b in pairs])}
        res["fixture"][t]["vs_untouched"] = d

# SciFact
ids = json.loads((OUT / "scifact_ids.json").read_text()) if (OUT / "scifact_ids.json").exists() else None
if ids:
    qrels = pd.read_csv(HERE.parent / "sparseup-v2" / "pack" / "cache" / "scifact" / "test.tsv", sep="\t", dtype=str)
    rel = {}
    for q, d in zip(qrels["query-id"], qrels["corpus-id"]):
        rel.setdefault(q, set()).add(d)
    doc_ids = np.array(ids["doc_ids"])
    nd = {}
    for t in tags:
        if not (OUT / f"{t}_sf_docs.npz").exists():
            continue
        S = (sp.load_npz(OUT / f"{t}_sf_queries.npz") @ sp.load_npz(OUT / f"{t}_sf_docs.npz").T).toarray()
        vals = []
        for i, q in enumerate(ids["query_ids"]):
            s = S[i].copy()
            s[doc_ids == q] = -np.inf
            top = doc_ids[np.argsort(-s, kind="mergesort")[:10]]
            dcg = sum(1 / np.log2(r + 2) for r, d in enumerate(top) if d in rel[q])
            idcg = sum(1 / np.log2(r + 2) for r in range(min(10, len(rel[q]))))
            vals.append(dcg / idcg)
        nd[t] = vals
        res["scifact"][t] = {"ndcg10": boot(vals)}
    if ref_t in nd:
        for t, v in nd.items():
            if t != ref_t:
                res["scifact"][t]["vs_untouched"] = boot(np.array(v) - np.array(nd[ref_t]))

(HERE / "results").mkdir(exist_ok=True)
(HERE / "results" / f"{OUT.parent.name if OUT.name == 'out' else OUT.name}.json").write_text(json.dumps(res, indent=1))
for t, o in res["fixture"].items():
    print(t, "nnz d/q", o["doc_nnz"], o["query_nnz"], " | ".join(
        f"{f} {o[f]['alone']['r10'][0]:.3f}/{o[f]['alone']['mrr'][0]:.3f} +tag {o[f]['rrf_tag']['r10'][0]:.3f} +tag+tfidf {o[f]['rrf_tag_tfidf']['r10'][0]:.3f}"
        for f in ("catalog", "symbol", "mesh", "all")), "| scifact", res["scifact"].get(t, {}).get("ndcg10"))
