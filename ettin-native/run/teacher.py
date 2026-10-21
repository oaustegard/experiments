"""Build the distillation text set and embed it with the teacher.

Usage: .venv/bin/python -I run/teacher.py [--teacher Snowflake/snowflake-arctic-embed-m-v1.5]

Texts (out/data/train.jsonl, one {"t", "role"} per line, role q = query prompt, d = none):
  queries  #102 synth_train (23,985) + gpu_more (55,289) + msmarco_train (30,000), 60,000
           more MS MARCO queries, 30,000 NQ questions; any string equal (case-folded) to a
           synth_test or real_test query is dropped
  docs     the 61,686 corpus posts (transductive: the deployed index is distilled on its own
           corpus, the teacher sees the same text), 60,000 MS MARCO passages, 30,000 NQ answers
Teacher vectors: L2-normalized, fp16, out/data/<teacher>.f16.npy, row-aligned with train.jsonl.
Docs truncated to 128 tokens like the task.
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT.parent
D = ROOT / "out" / "data"
TEST = EXP / "eg2-native-small" / "out" / "q"
POSTS = EXP / "eg2-query-student" / "out" / "posts.jsonl"


def build():
    test = {json.loads(l)["q"].strip().lower() for f in ("synth_test", "real_test") for l in (TEST / f"{f}.jsonl").open()}
    rng = random.Random(0)
    qs = []
    for f in ("synth_train", "gpu_more", "msmarco_train"):
        qs += [json.loads(l)["q"] for l in (D / f"{f}.jsonl").open()]
    mq = pq.read_table(D / "msm_query.parquet").column(1).to_pylist()
    nq = pq.read_table(D / "nq.parquet")
    nq_q, nq_a = nq.column(0).to_pylist(), nq.column(1).to_pylist()
    mp = pq.read_table(D / "msm_pass0.parquet").column(1).to_pylist()
    qs += rng.sample(mq, 60000)
    idx = rng.sample(range(len(nq_q)), 30000)
    qs += [nq_q[i] for i in idx]
    seen = set()
    qs = [q for q in qs if q and q.strip().lower() not in test and not (q in seen or seen.add(q))]
    posts = [json.loads(l)["text"] for l in POSTS.open()]
    docs = [p for p in posts if p] + rng.sample(mp, 60000) + [nq_a[i] for i in idx]
    rows = [{"t": q, "role": "q"} for q in qs] + [{"t": d, "role": "d"} for d in docs]
    with (D / "train.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("queries", len(qs), "docs", len(docs), flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="Snowflake/snowflake-arctic-embed-m-v1.5")
    a = ap.parse_args()
    rows = build() if not (D / "train.jsonl").exists() else [json.loads(l) for l in (D / "train.jsonl").open()]
    m = SentenceTransformer(a.teacher, device="mps" if torch.backends.mps.is_available() else "cpu")
    m.max_seq_length = 128
    qp = (m.prompts or {}).get("query", "")
    out = np.zeros((len(rows), m.get_sentence_embedding_dimension()), np.float16)
    for role, p in (("q", qp), ("d", "")):
        ii = [i for i, r in enumerate(rows) if r["role"] == role]
        t0 = time.time()
        E = m.encode([p + rows[i]["t"] for i in ii], batch_size=128, normalize_embeddings=True, show_progress_bar=False)
        out[ii] = E.astype(np.float16)
        print(role, len(ii), f"{time.time() - t0:.0f} s", flush=True)
    np.save(D / f"{a.teacher.split('/')[-1]}.f16.npy", out)


if __name__ == "__main__":
    main()
