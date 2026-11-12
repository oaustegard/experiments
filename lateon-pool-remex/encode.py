#!/usr/bin/env python3
"""Encode BEIR SciFact and NFCorpus with lightonai/LateOn (ModernBERT-base ColBERT,
128-d tokens, not trained for pooling) at the model card's settings: [Q]/[D] prefixes,
query_length 32, document_length 300, document-side punctuation skiplist, no query
expansion. sentence-transformers 6.3 MultiVectorEncoder loads the PyLate checkpoint
natively (residual Dense layers included); the SciFact score is the load check.

Documents are stored UNPOOLED; bench.py applies every pooling setting afterwards.
Checkpoints every CHUNK documents, so a kill loses under ~5 minutes.

Writes .work/<corpus>/ (gitignored):
  tokens.npy (n_tokens, 128) float32 unit-norm, offsets.npy (n_docs+1,) int64
  q_tokens.npy, q_offsets.npy   judged test queries
  doc_ids.json, query_ids.json, qrels.json, meta.json
Prints 'encode done -> <corpus>' per corpus and 'ENCODE ALL DONE' at the end.
"""
from __future__ import annotations

import json, sys, time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
WORK = HERE / ".work"
MODEL = "lightonai/LateOn"
CORPORA = {"scifact": "BeIR/scifact", "nfcorpus": "BeIR/nfcorpus"}
CHUNK = 1000


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


def to_np(e):
    return e.float().cpu().numpy() if torch.is_tensor(e) else np.asarray(e, dtype=np.float32)


def flatten(embs):
    embs = [to_np(e) for e in embs]
    lens = [len(e) for e in embs]
    return np.concatenate(embs).astype(np.float32), np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)


def main(corpora):
    torch.set_num_threads(4)
    from sentence_transformers import MultiVectorEncoder
    m = MultiVectorEncoder(MODEL, device="cpu")
    for corpus in corpora:
        out = WORK / corpus
        if (out / "meta.json").exists():
            print(f"[{corpus}] done, skipping", flush=True)
            continue
        (out / "chunks").mkdir(parents=True, exist_ok=True)
        docs, dids, queries, qids, qrels = load_corpus(corpus)
        json.dump(dids, open(out / "doc_ids.json", "w"))
        json.dump(qids, open(out / "query_ids.json", "w"))
        json.dump(qrels, open(out / "qrels.json", "w"))
        order = np.argsort([len(d) for d in docs])  # length-sorted batches, restored below
        n_chunks = (len(docs) + CHUNK - 1) // CHUNK
        t0 = time.time()
        for c in range(n_chunks):
            f = out / "chunks" / f"{c:04d}.npz"
            if f.exists():
                continue
            idx = order[c * CHUNK:(c + 1) * CHUNK]
            D = m.encode_document([docs[i] for i in idx], batch_size=16, show_progress_bar=False)
            T, off = flatten(D)
            np.savez(f, idx=idx, T=T, off=off)
            print(f"[{corpus}] chunk {c + 1}/{n_chunks} {time.time() - t0:.0f}s", flush=True)
        per_doc = [None] * len(docs)
        for c in range(n_chunks):
            # read each array ONCE: NpzFile re-reads the whole array on every z[...] access,
            # and a slice of each fresh copy kept ~1000 copies alive (13 GB OOM, ERRORS.md #1)
            with np.load(out / "chunks" / f"{c:04d}.npz") as z:
                idx, Tc, offc = z["idx"], z["T"], z["off"]
            for j, i in enumerate(idx):
                per_doc[i] = Tc[offc[j]:offc[j + 1]]
        T, off = flatten(per_doc)
        Q = m.encode_query(queries, batch_size=32, show_progress_bar=False)
        QT, qoff = flatten(Q)
        np.save(out / "tokens.npy", T); np.save(out / "offsets.npy", off)
        np.save(out / "q_tokens.npy", QT); np.save(out / "q_offsets.npy", qoff)
        meta = dict(model=MODEL, corpus=corpus, n_docs=len(docs), n_queries=len(qids),
                    n_tokens=int(len(T)), tokens_per_doc=round(len(T) / len(docs), 2),
                    q_tokens_per_query=round(len(QT) / len(qids), 2), dim=int(T.shape[1]),
                    query_length=m.query_length if hasattr(m, "query_length") else None,
                    encode_doc_seconds=round(time.time() - t0, 1),
                    max_abs_norm_dev=float(np.abs(np.linalg.norm(T, axis=1) - 1).max()))
        json.dump(meta, open(out / "meta.json", "w"), indent=1)
        print(f"[{corpus}] {meta}", flush=True)
        print(f"encode done -> {corpus}", flush=True)
    print("ENCODE ALL DONE", flush=True)


if __name__ == "__main__":
    args = sys.argv[1:]
    main([a for a in args if a in CORPORA] or list(CORPORA))
