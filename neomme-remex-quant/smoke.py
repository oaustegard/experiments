#!/usr/bin/env python3
"""Assemble a partial dataset from whatever checkpoints exist + encode a few judged
queries, so bench.py can be exercised end to end before the full encode finishes.
Writes data/smoke_enc/. Not a result."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import encode as E

HERE = Path(__file__).resolve().parent; OUT = HERE / "data" / "smoke_enc"; OUT.mkdir(parents=True, exist_ok=True)
doc_ids_all, texts = E.corpus_texts()
idx, dense, toks = [], [], []
for ck in sorted(E.CKPT.glob("doc_*.npz")):
    z = np.load(ck); lens = z["lens"]; st = np.concatenate([[0], np.cumsum(lens)])
    for j, i in enumerate(z["idx"]):
        idx.append(int(i)); dense.append(z["dense"][j]); toks.append(z["toks"][st[j]:st[j + 1]])
doc_ids = [doc_ids_all[i] for i in idx]
np.save(OUT / "dense.npy", np.stack(dense)); np.save(OUT / "mv_tokens.npy", np.concatenate(toks))
np.save(OUT / "mv_offsets.npy", np.concatenate([[0], np.cumsum([len(t) for t in toks])]).astype(np.int64)); (OUT / "doc_ids.json").write_text(json.dumps(doc_ids))
qrels = pd.read_csv(E.DATA / "qrels" / "test.tsv", sep="\t"); have = set(doc_ids)
qids_hit = list(dict.fromkeys(str(q) for q, c in zip(qrels["query-id"], qrels["corpus-id"]) if str(c) in have))[:12]
q = pd.read_parquet(E.DATA / "queries.parquet"); q = q[q["_id"].astype(str).isin(qids_hit)]
proc, model = E.load_model(); torch.set_num_threads(2)
d, t, _ = E.encode_batch(proc, model, list(q.text), "query")
np.save(OUT / "q_dense.npy", d); np.save(OUT / "q_mv_tokens.npy", np.concatenate(t)); np.save(OUT / "q_mv_offsets.npy", np.concatenate([[0], np.cumsum([len(x) for x in t])]).astype(np.int64))
(OUT / "query_ids.json").write_text(json.dumps(list(q["_id"].astype(str))))
print(f"smoke set: {len(doc_ids)} docs, {len(q)} queries with a relevant doc present")
