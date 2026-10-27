"""Write one dataset's worker inputs into the jail directory.

    python3 prep.py DATASET JAIL_DIR [MAX_QUERIES]

docs.jsonl holds the union of the SPLADE top-100 candidates (title + text),
queries.json the test queries, cands.json the candidate lists. MAX_QUERIES
takes a fixed seeded subsample (ArguAna has 1,373 test queries).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def main(ds, jail, max_q=None):
    p = HERE / "cache" / "data" / ds
    run = json.loads((HERE / "cache" / "runs" / f"{ds}.json").read_text())
    qr = pd.read_parquet(p / "qrels_test.parquet")
    judged = set(qr.loc[qr.score > 0, "query-id"].astype(str))
    qids = sorted(q for q in run if q in judged)  # nDCG is undefined without a positive
    if max_q and len(qids) > int(max_q):
        qids = sorted(np.random.default_rng(0).choice(qids, int(max_q), replace=False).tolist())
    corpus = pd.read_parquet(p / "corpus.parquet")
    text = dict(zip(corpus["_id"].astype(str),
                    (corpus.title.fillna("") + " " + corpus.text.fillna("")).str.strip()))
    queries = pd.read_parquet(p / "queries.parquet")
    qtext = dict(zip(queries["_id"].astype(str), queries.text))
    cands = {q: [d for d, _ in run[q]] for q in qids}
    docs = sorted({d for q in qids for d in cands[q]})
    out = Path(jail) / ds
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "docs.jsonl", "w") as f:
        f.writelines(json.dumps({"id": d, "text": text[d]}) + "\n" for d in docs)
    (out / "queries.json").write_text(json.dumps({q: qtext[q] for q in qids}))
    (out / "cands.json").write_text(json.dumps(cands))
    print(ds, len(qids), "queries", len(docs), "docs", sum(len(c) for c in cands.values()), "pairs")


if __name__ == "__main__":
    main(*sys.argv[1:])
