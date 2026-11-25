"""Score the sparseup v2 arms next to the frozen evaluate.py arms, on the same queries, ties and metrics.

    python3 sparseup_eval.py test        # after sparseup_encode.py fp32 and int8; -> results/sparseup_test.json

evaluate.py runs unchanged (runpy) and its scoring functions, queries and tie-break are reused, so every
arm here is scored exactly as in RESULTS.md. Added arms: sparseup_fp32, sparseup_int8 (dot product of
the folded sparse vectors) and RRF (k = 60) fusions of sparseup_fp32 with tag_jaccard, BM25 and TF-IDF.
"""
from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
SPLIT = sys.argv[1] if len(sys.argv) > 1 else "test"
sys.argv = ["evaluate.py", SPLIT]
E = runpy.run_path(str(HERE / "evaluate.py"), run_name="__main__")
queries, tie, metrics, rrf = E["queries"], E["tie"], E["metrics"], E["rrf"]
s_tag, s_bm25, s_tfidf = E["s_tag"], E["s_bm25"], E["s_tfidf"]


def load(variant):
    D = sp.load_npz(HERE / "cache" / f"sparseup_{variant}_docs.npz")
    Q = sp.load_npz(HERE / "cache" / f"sparseup_{variant}_queries.npz")
    qi = {q: i for i, q in enumerate(json.loads((HERE / "cache" / f"sparseup_{variant}_queries.json").read_text()))}
    return lambda text: (D @ Q[qi[text]].T).toarray().ravel() + tie


sup = {v: load(v) for v in ("fp32", "int8")}
rows = []
for q in queries:
    st, _ = s_tag(q["text"], "jaccard")
    sb, sf, s32, s8 = s_bm25(q["text"]), s_tfidf(q["text"]), sup["fp32"](q["text"]), sup["int8"](q["text"])
    arms = {"tag_jaccard": st, "bm25": sb, "tfidf_wc": sf, "sparseup_fp32": s32, "sparseup_int8": s8,
            "rrf_sparseup_tag": rrf(s32, st), "rrf_sparseup_bm25": rrf(s32, sb), "rrf_sparseup_tfidf": rrf(s32, sf),
            "rrf_tag_tfidf": rrf(st, sf), "rrf_sparseup_tag_tfidf": rrf(s32, st, sf)}
    row = {"concept": q["concept"], "form": q["form"], "text": q["text"], "n_rel": len(q["rel"])}
    for a, s in arms.items():
        row[a] = dict(zip(("r10", "mrr"), metrics(s, q["rel"])))
    rows.append(row)

ARMS = list(arms)
rng = np.random.default_rng(20261125)


def boot(vals):
    vals = np.asarray(vals)
    m = vals[rng.integers(0, len(vals), (2000, len(vals)))].mean(1)
    return [round(float(vals.mean()), 3), round(float(np.percentile(m, 2.5)), 3), round(float(np.percentile(m, 97.5)), 3)]


PAIRS = [("sparseup_fp32", "tag_jaccard"), ("sparseup_fp32", "bm25"), ("sparseup_fp32", "tfidf_wc"),
         ("sparseup_int8", "sparseup_fp32"), ("rrf_sparseup_tag", "tag_jaccard"), ("rrf_sparseup_tag", "sparseup_fp32"),
         ("rrf_sparseup_tfidf", "sparseup_fp32"), ("rrf_sparseup_tag_tfidf", "tag_jaccard")]
summary = {}
for form in ("catalog", "symbol", "mesh", "all"):
    R = [r for r in rows if form == "all" or r["form"] == form]
    out = {"n_queries": len(R)}
    for a in ARMS:
        out[a] = {m: boot([r[a][m] for r in R]) for m in ("r10", "mrr")}
    out["paired_diffs"] = {f"{a}-{b}": {m: boot([r[a][m] - r[b][m] for r in R]) for m in ("r10", "mrr")} for a, b in PAIRS}
    summary[form] = out

D32 = sp.load_npz(HERE / "cache" / "sparseup_fp32_docs.npz")
(HERE / "results" / f"sparseup_{SPLIT}.json").write_text(json.dumps(
    {"split": SPLIT, "n_docs": D32.shape[0], "doc_nnz_mean": round(D32.getnnz() / D32.shape[0], 1),
     "summary": summary, "rows": rows}, indent=1, ensure_ascii=False))
for form, o in summary.items():
    print(form, o["n_queries"], " ".join(f"{a}={o[a]['r10'][0]:.3f}/{o[a]['mrr'][0]:.3f}" for a in ARMS))
