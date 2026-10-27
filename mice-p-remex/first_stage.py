"""SPLADE-v3 (DistilBERT) first stage: top-100 candidates per test query.

The MICE-P paper re-ranks the top-100 of naver/splade-v3-distilbert on the
decontaminated BEIR sets. This reproduces that candidate list on CPU so the
codec arms all re-rank the same documents. One run file per dataset, skipped
when present (checkpoint per unit).

    python3 first_stage.py scifact nfcorpus scidocs arguana fiqa
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
MODEL = CACHE / "models" / "splade-v3-distilbert"
DOC_LEN, Q_LEN, TOPK, BS = 256, 64, 100, 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

torch.set_num_threads(4)

if not MODEL.exists():  # fresh VM (Colab): fetch model and data from the Hub
    from huggingface_hub import snapshot_download
    snapshot_download("naver/splade-v3-distilbert", local_dir=MODEL)
    for ds in sys.argv[1:]:
        snapshot_download(f"lightonai/{ds}-decontaminated", repo_type="dataset",
                          local_dir=CACHE / "data" / ds, allow_patterns=["*.parquet"])


def load(ds):
    p = CACHE / "data" / ds
    corpus = pd.read_parquet(p / "corpus.parquet")
    queries = pd.read_parquet(p / "queries.parquet")
    qrels = pd.read_parquet(p / "qrels_test.parquet")
    corpus["full"] = (corpus.title.fillna("") + " " + corpus.text.fillna("")).str.strip()
    test_q = sorted(set(qrels["query-id"].astype(str)))
    qtext = dict(zip(queries["_id"].astype(str), queries.text))
    return corpus["_id"].astype(str).tolist(), corpus["full"].tolist(), test_q, [qtext[q] for q in test_q]


@torch.no_grad()
def encode(tok, model, texts, max_len):
    order = np.argsort([-len(t) for t in texts])
    rows, cols, vals = [], [], []
    for s in range(0, len(texts), BS):
        idx = order[s:s + BS]
        enc = tok([texts[i] for i in idx], truncation=True, max_length=max_len, padding=True,
                  return_tensors="pt").to(DEVICE)
        logits = model(**enc).logits.float()
        w = torch.log1p(torch.relu(logits)) * enc["attention_mask"].unsqueeze(-1)
        rep = w.max(dim=1).values.cpu()
        nz = rep.nonzero()
        rows.append(torch.as_tensor(idx)[nz[:, 0]].numpy())
        cols.append(nz[:, 1].numpy())
        vals.append(rep[nz[:, 0], nz[:, 1]].numpy())
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                         shape=(len(texts), model.config.vocab_size), dtype=np.float32)


def main(datasets):
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForMaskedLM.from_pretrained(MODEL).eval().to(DEVICE)
    out_dir = CACHE / "runs"
    out_dir.mkdir(exist_ok=True)
    for ds in datasets:
        out = out_dir / f"{ds}.json"
        if out.exists():
            print(ds, "exists, skipping", flush=True)
            continue
        t0 = time.time()
        doc_ids, docs, qids, qtexts = load(ds)
        D = encode(tok, model, docs, DOC_LEN)
        t1 = time.time()
        Q = encode(tok, model, qtexts, Q_LEN)
        S = (Q @ D.T).toarray()
        run = {}
        for i, q in enumerate(qids):
            k = min(TOPK, S.shape[1])
            top = np.argpartition(-S[i], k - 1)[:k]
            top = top[np.argsort(-S[i][top], kind="stable")]
            run[q] = [[doc_ids[j], float(S[i, j])] for j in top]
        tmp = out.with_suffix(".tmp")
        tmp.write_text(json.dumps(run))
        tmp.rename(out)
        print(f"{ds}: {len(docs)} docs in {t1 - t0:.0f}s, {len(qids)} queries, total {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
