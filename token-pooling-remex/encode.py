#!/usr/bin/env python3
"""Encode BEIR SciFact and NFCorpus with the two checkpoints from the Hugging Face
token-pooling post (2026-11-11): the pooling-aware model and its baseline, trained on
the same data with the same recipe except the loss.

Documents are stored UNPOOLED; bench.py applies every pooling setting afterwards.
HierarchicalTokenPooling is the last module of the encode pipeline (after Normalize),
so pooling stored tokens gives the same vectors as encode_document(token_pooling=...).

Writes .work/<model_key>/<corpus>/ (gitignored):
  tokens.npy (n_tokens, 128) float32 unit-norm, offsets.npy (n_docs+1,) int64
  q_tokens.npy, q_offsets.npy     the judged test queries
  doc_ids.json, query_ids.json, qrels.json, meta.json
Resumable per (model, corpus): a finished meta.json is not recomputed.
"""
from __future__ import annotations

import json, sys, time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
WORK = HERE / ".work"
MODELS = {
    "aware": "tomaarsen/mve-ModernBERT-base-token-pooling",
    "baseline": "tomaarsen/mve-ModernBERT-base-token-pooling-baseline",
}
CORPORA = {"scifact": "BeIR/scifact", "nfcorpus": "BeIR/nfcorpus"}
DOC_MAX_LEN = 512


def load_corpus(name):
    from datasets import load_dataset
    hf = CORPORA[name]
    corpus = load_dataset(hf, "corpus", split="corpus")
    queries = load_dataset(hf, "queries", split="queries")
    qrels_ds = load_dataset(hf + "-qrels", split="test")
    qrels = {}
    for r in qrels_ds:
        if int(r["score"]) > 0:
            qrels.setdefault(str(r["query-id"]), {})[str(r["corpus-id"])] = int(r["score"])
    qmap = {str(r["_id"]): r["text"] for r in queries}
    qids = [q for q in qrels if q in qmap]
    docs = [(r["title"] + " " + r["text"]).strip() for r in corpus]
    dids = [str(r["_id"]) for r in corpus]
    return docs, dids, [qmap[q] for q in qids], qids, {q: qrels[q] for q in qids}


def flatten(embs):
    embs = [e.float().cpu().numpy() if torch.is_tensor(e) else np.asarray(e, dtype=np.float32) for e in embs]
    lens = [len(e) for e in embs]
    return np.concatenate(embs).astype(np.float32), np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)


def main(pairs):
    torch.set_num_threads(4)
    from sentence_transformers import MultiVectorEncoder
    for mkey, corpus in pairs:
        out = WORK / mkey / corpus
        if (out / "meta.json").exists():
            print(f"[{mkey}/{corpus}] done, skipping", flush=True)
            continue
        out.mkdir(parents=True, exist_ok=True)
        docs, dids, queries, qids, qrels = load_corpus(corpus)
        json.dump(dids, open(out / "doc_ids.json", "w"))
        json.dump(qids, open(out / "query_ids.json", "w"))
        json.dump(qrels, open(out / "qrels.json", "w"))
        m = MultiVectorEncoder(MODELS[mkey], device="cpu")
        m.max_seq_length = DOC_MAX_LEN
        order = np.argsort([len(d) for d in docs])
        inv = np.empty_like(order); inv[order] = np.arange(len(order))
        t0 = time.time()
        D = m.encode_document([docs[i] for i in order], batch_size=16, show_progress_bar=False)
        D = [D[inv[i]] for i in range(len(docs))]
        t_doc = time.time() - t0
        Q = m.encode_query(queries, batch_size=32, show_progress_bar=False)
        T, off = flatten(D); QT, qoff = flatten(Q)
        np.save(out / "tokens.npy", T); np.save(out / "offsets.npy", off)
        np.save(out / "q_tokens.npy", QT); np.save(out / "q_offsets.npy", qoff)
        meta = dict(model=MODELS[mkey], corpus=corpus, n_docs=len(docs), n_queries=len(qids),
                    n_tokens=int(len(T)), tokens_per_doc=round(len(T) / len(docs), 2),
                    q_tokens_per_query=round(len(QT) / len(qids), 2), dim=int(T.shape[1]),
                    doc_max_len=DOC_MAX_LEN, encode_doc_seconds=round(t_doc, 1),
                    max_abs_norm_dev=float(np.abs(np.linalg.norm(T, axis=1) - 1).max()))
        json.dump(meta, open(out / "meta.json", "w"), indent=1)
        print(f"[{mkey}/{corpus}] {meta}", flush=True)
        del m


if __name__ == "__main__":
    args = sys.argv[1:]
    corpora = [a for a in args if a in CORPORA] or list(CORPORA)
    models = [a for a in args if a in MODELS] or list(MODELS)
    main([(m, c) for c in corpora for m in models])
