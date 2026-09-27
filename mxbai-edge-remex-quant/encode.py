#!/usr/bin/env python3
"""Encode BEIR SciFact and NFCorpus with mxbai-edge-colbert-v0-32m (tokens) and
bge-small-en-v1.5 (single vector, the storage comparator).

Writes .work/<corpus>/:
  mv_tokens.npy    (n_tokens_total, 64) float32, L2-normalised per token (pylate output,
                   document-side punctuation skiplist already applied)
  mv_offsets.npy   (n_docs + 1,) int64; doc i owns rows offsets[i]:offsets[i+1]
  q_mv_tokens.npy, q_mv_offsets.npy   same for the judged test queries
  dense.npy, q_dense.npy               bge-small-en-v1.5, (n, 384) L2-normalised
  doc_ids.json, query_ids.json, qrels.json, meta.json

Resumable per (corpus, model): a finished .npy is not recomputed.
Encode settings are the model card's own (config_sentence_transformers.json):
query_length 48, document_length 512, no query expansion, skiplist on.
"""
from __future__ import annotations

import json, sys, time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
WORK = HERE / ".work"
MODEL = "mixedbread-ai/mxbai-edge-colbert-v0-32m"
DENSE_MODEL = "BAAI/bge-small-en-v1.5"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
CORPORA = {"scifact": "BeIR/scifact", "nfcorpus": "BeIR/nfcorpus"}


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
    lens = [len(e) for e in embs]
    return (np.concatenate(embs).astype(np.float32),
            np.concatenate([[0], np.cumsum(lens)]).astype(np.int64))


def main(names):
    torch.set_num_threads(4)
    for name in names:
        out = WORK / name
        out.mkdir(parents=True, exist_ok=True)
        docs, dids, queries, qids, qrels = load_corpus(name)
        json.dump(dids, open(out / "doc_ids.json", "w"))
        json.dump(qids, open(out / "query_ids.json", "w"))
        json.dump(qrels, open(out / "qrels.json", "w"))
        print(f"[{name}] docs={len(docs)} queries={len(qids)}", flush=True)
        # length-sorted batching, restored to corpus order afterwards
        order = np.argsort([len(d) for d in docs])
        inv = np.empty_like(order); inv[order] = np.arange(len(order))

        if not (out / "mv_offsets.npy").exists():
            from pylate import models
            m = models.ColBERT(model_name_or_path=MODEL, device="cpu")
            meta = dict(model=MODEL, query_length=m.query_length, document_length=m.document_length,
                        do_query_expansion=m.do_query_expansion, skiplist=len(m.skiplist) > 0)
            t0 = time.time()
            D = m.encode([docs[i] for i in order], is_query=False, batch_size=32, convert_to_numpy=True)
            D = [D[inv[i]] for i in range(len(docs))]
            Q = m.encode(queries, is_query=True, batch_size=32, convert_to_numpy=True)
            meta["encode_seconds"] = round(time.time() - t0, 1)
            T, off = flatten(D); QT, qoff = flatten(Q)
            np.save(out / "mv_tokens.npy", T); np.save(out / "q_mv_tokens.npy", QT)
            np.save(out / "q_mv_offsets.npy", qoff); np.save(out / "mv_offsets.npy", off)
            meta.update(n_tokens=int(len(T)), tokens_per_doc=float(len(T) / len(docs)),
                        dim=int(T.shape[1]), max_abs_norm_dev=float(np.abs(np.linalg.norm(T, axis=1) - 1).max()))
            json.dump(meta, open(out / "meta_mv.json", "w"), indent=1)
            print(f"[{name}] mxbai: {meta}", flush=True)
            del m

        if not (out / "q_dense.npy").exists():
            from sentence_transformers import SentenceTransformer
            st = SentenceTransformer(DENSE_MODEL, device="cpu")
            t0 = time.time()
            Dd = st.encode([docs[i] for i in order], batch_size=32, normalize_embeddings=True, convert_to_numpy=True)[inv]
            Qd = st.encode([BGE_QUERY_PREFIX + q for q in queries], batch_size=32, normalize_embeddings=True, convert_to_numpy=True)
            np.save(out / "dense.npy", Dd.astype(np.float32)); np.save(out / "q_dense.npy", Qd.astype(np.float32))
            json.dump(dict(model=DENSE_MODEL, max_seq_length=st.max_seq_length, query_prefix=BGE_QUERY_PREFIX,
                           encode_seconds=round(time.time() - t0, 1)), open(out / "meta_dense.json", "w"), indent=1)
            print(f"[{name}] bge-small done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or list(CORPORA))
