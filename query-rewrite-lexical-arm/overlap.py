"""Question-to-gold-document term overlap, per question category.

BM25 wins by term matching, so if the benchmark's generator lifted phrasing from
the gold document into the question, a lexical arm is advantaged in a way that
would not transfer to queries a person actually types. The benchmark's own
`semantic` category is its control for exactly this.
"""
import json, os, sys
import numpy as np
import pyarrow.parquet as pq
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bm25_sparse as BS

DATA = "/home/user/erb-data"
try:
    import Stemmer; ST = Stemmer.Stemmer("english")
except ImportError:
    ST = None
an = BS.Analyzer(ST)

q = pq.read_table(f"{DATA}/questions_test.parquet").to_pydict()
qs = [(i, t, x, list(g)) for i, t, x, g in
      zip(q["question_id"], q["question_type"], q["question"], q["expected_doc_ids"])
      if g is not None and len(g) > 0]
gold_ids = {g for _, _, _, gg in qs for g in gg}

d = pq.read_table(f"{DATA}/documents_test.parquet")
ids = d.column("doc_id").to_pylist()
ti = d.column("title").to_pylist()
co = d.column("content").to_pylist()
want = {i: k for k, i in enumerate(ids) if i in gold_ids}
docs = {i: set(an(f"{ti[k] or ''}\n{co[k] or ''}")) for i, k in want.items()}
del d, ids, ti, co

rows = {}
for qid, qtype, text, gold in qs:
    qt = set(an(text))
    if not qt:
        continue
    best = max((len(qt & docs[g]) / len(qt) for g in gold if g in docs), default=0.0)
    rows.setdefault(qtype, []).append(best)

out = {t: dict(n=len(v), mean_term_coverage=round(100 * float(np.mean(v)), 1),
               median=round(100 * float(np.median(v)), 1))
       for t, v in sorted(rows.items())}
allv = [x for v in rows.values() for x in v]
out["_ALL"] = dict(n=len(allv), mean_term_coverage=round(100 * float(np.mean(allv)), 1),
                   median=round(100 * float(np.median(allv)), 1))
json.dump(out, open("results/overlap.json", "w"), indent=2)
print(f"{'question type':28} {'n':>4} {'mean %':>7} {'median %':>9}")
for t, v in sorted(out.items(), key=lambda x: -x[1]["mean_term_coverage"]):
    print(f"{t:28} {v['n']:4} {v['mean_term_coverage']:7.1f} {v['median']:9.1f}")
