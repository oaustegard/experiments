#!/usr/bin/env python3
"""Encode BEIR SciFact with perplexity-ai/pplx-embed-v2-late-0.6b (128-d tokens).

Writes .work/<corpus>/ in the layout mxbai-edge-remex-quant/bench.py reads:
  mv_tokens.npy    (n_tokens_total, 128) float32, L2-normalised per token (the model's
                   own Normalize module; document-side punctuation skiplist applied)
  mv_offsets.npy   (n_docs + 1,) int64; doc i owns rows offsets[i]:offsets[i+1]
  q_mv_tokens.npy, q_mv_offsets.npy   same for the judged test queries
  doc_ids.json, query_ids.json, qrels.json, meta_mv.json

Encode settings are the model card's own (sentence-transformers MultiVectorEncoder,
encode_query / encode_document prompts "[Q] " / "[D] ", query_length 1024,
document_length 4096, no query expansion). Documents are length-sorted into batches and
checkpointed per chunk under .work/<corpus>/chunks/, so a killed run resumes.
"""
from __future__ import annotations

import json, sys, time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
WORK = HERE / ".work"
MODEL = "perplexity-ai/pplx-embed-v2-late-0.6b"
CORPORA = {"scifact": "BeIR/scifact", "nfcorpus": "BeIR/nfcorpus"}
CHUNK = 128


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


def to_np(x):
    return [e.float().cpu().numpy() if torch.is_tensor(e) else np.asarray(e, dtype=np.float32) for e in x]


def main(names):
    torch.set_num_threads(4)
    from sentence_transformers import MultiVectorEncoder
    m = None
    for name in names:
        out = WORK / name
        (out / "chunks").mkdir(parents=True, exist_ok=True)
        if (out / "mv_offsets.npy").exists():
            print(f"[{name}] already done", flush=True)
            continue
        docs, dids, queries, qids, qrels = load_corpus(name)
        json.dump(dids, open(out / "doc_ids.json", "w"))
        json.dump(qids, open(out / "query_ids.json", "w"))
        json.dump(qrels, open(out / "qrels.json", "w"))
        print(f"[{name}] docs={len(docs)} queries={len(qids)}", flush=True)
        order = np.argsort([len(d) for d in docs], kind="stable")
        n_chunks = (len(order) + CHUNK - 1) // CHUNK
        pending = not (out / "q_mv_offsets.npy").exists() or any(
            not (out / "chunks" / f"{c:04d}.npz").exists() for c in range(n_chunks))
        if m is None and pending:
            m = MultiVectorEncoder(MODEL, device="cpu")
        t0 = time.time()
        if not (out / "q_mv_offsets.npy").exists():
            QT, qoff = flatten(to_np(m.encode_query(queries, batch_size=32)))
            np.save(out / "q_mv_tokens.npy", QT); np.save(out / "q_mv_offsets.npy", qoff)
            print(f"[{name}] queries: {len(QT)} tokens, {time.time() - t0:.0f}s", flush=True)

        for c in range(n_chunks):
            f = out / "chunks" / f"{c:04d}.npz"
            if f.exists():
                continue
            idx = order[c * CHUNK:(c + 1) * CHUNK]
            t1 = time.time()
            E = to_np(m.encode_document([docs[i] for i in idx], batch_size=16))
            T, off = flatten(E)
            np.savez(f, idx=idx, T=T, off=off)
            el = time.time() - t0
            print(f"[{name}] chunk {c + 1}/{n_chunks} {time.time() - t1:.0f}s (elapsed {el / 60:.1f} min)", flush=True)

        per_doc = [None] * len(docs)
        for c in range(n_chunks):
            # read each array once: z["T"] re-reads the whole array on every access, and a
            # per-doc slice of it pins one full copy per doc (OOM-killed the first assembly)
            with np.load(out / "chunks" / f"{c:04d}.npz") as z:
                idx, Tc, offc = z["idx"], z["T"], z["off"]
            for j, i in enumerate(idx):
                per_doc[i] = Tc[offc[j]:offc[j + 1]]
        T, off = flatten(per_doc)
        np.save(out / "mv_tokens.npy", T)
        np.save(out / "mv_offsets.npy", off)  # written last: its presence marks completion
        meta = dict(model=MODEL, encode_seconds_this_run=round(time.time() - t0, 1), n_tokens=int(len(T)),
                    tokens_per_doc=float(len(T) / len(docs)), dim=int(T.shape[1]),
                    max_abs_norm_dev=float(np.abs(np.linalg.norm(T, axis=1) - 1).max()),
                    sentence_transformers=__import__("sentence_transformers").__version__,
                    transformers=__import__("transformers").__version__, torch=torch.__version__)
        json.dump(meta, open(out / "meta_mv.json", "w"), indent=1)
        print(f"[{name}] done: {meta}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ["scifact"])
