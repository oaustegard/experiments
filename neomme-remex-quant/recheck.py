#!/usr/bin/env python3
"""Sub-minute fixture: the writeup cannot drift from the artifacts.

1. Re-renders the results tables from results.json + results_perquery.npz and
   diffs them against the block embedded in RESULTS.md.
2. Recomputes the fp32 dense nDCG@10 through a deliberately different code path
   (sorted() over Python floats, explicit math.log2 loop) from the cached
   embeddings, and checks it against results.json. Needs data/scifact_enc; skipped
   with a warning when the cache is absent (it is gitignored).
3. Negative control: shuffled qrels must collapse the score, or the scorer cannot
   go red.
Exit non-zero on any mismatch."""
from __future__ import annotations

import json, math, random, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import report  # noqa: E402

ok = True


def check(label, cond, detail=""):
    global ok
    print(("PASS " if cond else "FAIL ") + label + (f"  ({detail})" if detail else ""))
    ok &= bool(cond)


s = (HERE / "RESULTS.md").read_text()
for corpus, (stem, marker) in report.STEMS.items():
    if not (HERE / f"{stem}.json").exists():
        continue
    results, perquery, _ = report.load(corpus)
    n_q = len(next(iter(perquery.values())))
    block = report.render(results, perquery).replace("(300 queries, 5,000 resamples)", f"({n_q:,} queries, 5,000 resamples)")
    embedded = s[s.index(f"<!-- {marker}:start -->") + len(f"<!-- {marker}:start -->"):s.index(f"<!-- {marker}:end -->")].strip()
    check(f"RESULTS.md {marker} == render({stem}.json)", embedded == block.strip())
    check(f"every arm in {stem}.json has per-query scores", set(results) == set(perquery))
results = json.loads((HERE / "results.json").read_text())
check("dense 4x MRL x fp32/remex arms present", all(f"dense remex {b}-bit d={d}" in results for b in (4, 2, 1) for d in (128, 256, 512, 1024)))

data = HERE / "data" / "scifact_enc"
if (data / "dense.npy").exists():
    dense = np.load(data / "dense.npy"); qd = np.load(data / "q_dense.npy")
    doc_ids = json.loads((data / "doc_ids.json").read_text()); qids = json.loads((data / "query_ids.json").read_text())
    qrels = defaultdict(set)
    for _, r in pd.read_csv(HERE / "data" / "scifact" / "qrels" / "test.tsv", sep="\t").iterrows():
        if int(r["score"]) > 0:
            qrels[str(r["query-id"])].add(str(r["corpus-id"]))

    def ndcg_independent(qrels_):
        tot, n = 0.0, 0
        for qi, q in enumerate(qids):
            rel = qrels_[q]
            if not rel & set(doc_ids):
                continue
            scores = (qd[qi] @ dense.T).tolist()
            ranked = sorted(range(len(scores)), key=lambda i: -scores[i])[:10]
            dcg = sum(1.0 / math.log2(r + 2) for r, i in enumerate(ranked) if doc_ids[i] in rel)
            idcg = sum(1.0 / math.log2(r + 2) for r in range(min(len(rel), 10)))
            tot += dcg / idcg; n += 1
        return tot / n
    v = ndcg_independent(qrels); claimed = results["dense fp32 d=1024"]["ndcg10"]
    check("fp32 dense nDCG@10 independent recompute", abs(v - claimed) < 1e-6, f"{v:.6f} vs {claimed:.6f}")
    random.seed(0); ids = list(doc_ids); shuffled = {q: {random.choice(ids) for _ in rel} for q, rel in qrels.items()}
    v0 = ndcg_independent(shuffled)
    check("negative control (shuffled qrels) collapses", v0 < 0.05, f"{v0:.4f}")
    meta = json.loads((data / "meta.json").read_text())
    check("meta.json records n_docs = 5183", meta["n_docs"] == 5183)
else:
    print("SKIP  data/scifact_enc absent (gitignored) — independent recompute not run")

sys.exit(0 if ok else 1)
