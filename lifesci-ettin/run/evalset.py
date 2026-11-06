"""BEIR biomedical test sets (SciFact, NFCorpus, TREC-COVID) from the Hub, and nDCG@10 / recall@100."""
import csv
import math
from functools import lru_cache

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

SETS = ["scifact", "nfcorpus", "trec-covid"]


@lru_cache(None)
def load(name):
    c = pd.read_parquet(hf_hub_download(f"BeIR/{name}", "corpus/corpus-00000-of-00001.parquet", repo_type="dataset"))
    q = pd.read_parquet(hf_hub_download(f"BeIR/{name}", "queries/queries-00000-of-00001.parquet", repo_type="dataset"))
    qr = {}
    with open(hf_hub_download(f"BeIR/{name}-qrels", "test.tsv", repo_type="dataset")) as f:
        r = csv.reader(f, delimiter="\t")
        next(r)
        for qid, did, s in r:
            if int(s) > 0:
                qr.setdefault(str(qid), {})[str(did)] = int(s)
    q = q[q["_id"].astype(str).isin(qr)]
    docs = {"ids": c["_id"].astype(str).tolist(), "title": c["title"].fillna("").tolist(), "text": c["text"].fillna("").tolist()}
    queries = {"ids": q["_id"].astype(str).tolist(), "text": q["text"].tolist()}
    return docs, queries, qr


def score(ranked, qids, qrels, k=10):
    """ranked: list (per query) of doc-id lists, best first."""
    nd, rc = [], []
    for qid, r in zip(qids, ranked):
        rel = qrels[qid]
        dcg = sum((2 ** rel.get(d, 0) - 1) / math.log2(i + 2) for i, d in enumerate(r[:k]))
        ideal = sorted(rel.values(), reverse=True)[:k]
        idcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
        nd.append(dcg / idcg if idcg else 0.0)
        rc.append(len(set(r[:100]) & set(rel)) / len(rel))
    return {"ndcg@10": float(np.mean(nd)), "recall@100": float(np.mean(rc)), "per_query_ndcg": [round(x, 4) for x in nd]}


def dense_rank(Q, Dm, doc_ids, k=100):
    Q = Q / np.linalg.norm(Q, axis=1, keepdims=True)
    Dm = Dm / np.linalg.norm(Dm, axis=1, keepdims=True)
    out = []
    for i in range(0, len(Q), 256):
        s = Q[i:i + 256] @ Dm.T
        top = np.argpartition(-s, min(k, s.shape[1] - 1), axis=1)[:, :k]
        for row, t in zip(s, top):
            out.append([doc_ids[j] for j in t[np.argsort(-row[t])]])
    return out
