#!/usr/bin/env python3
"""Encode BEIR SciFact and NFCorpus with google/embeddinggemma-2, text-only.

Text-only load is the card's own: config_kwargs={"vision_config": None,
"audio_config": None} (270M effective). Prompts are the card's: documents
"title: {title} | text: {text}" ("title: none" when empty), queries the
"Retrieval-query" prompt "task: search result | query: ". float32 compute
(the card warns float16 overflows; bf16 is not faster on this CPU).

Writes .work/<corpus>/:
  dense.npy, q_dense.npy   (n, 768) float32, L2-normalised (the model's Normalize)
  doc_ids.json, query_ids.json, qrels.json, meta.json
Resumable: each 128-doc chunk lands in .work/<corpus>/ckpt/ before assembly.
"""
from __future__ import annotations

import json, sys, time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
WORK = HERE / ".work"
MODEL = "google/embeddinggemma-2"
CORPORA = {"scifact": "BeIR/scifact", "nfcorpus": "BeIR/nfcorpus"}
QUERY_PROMPT = "Retrieval-query"
MAX_LEN = 8192
CHUNK, BATCH = 128, 16


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
    docs = [f"title: {(r['title'] or '').strip() or 'none'} | text: {r['text'].strip()}" for r in corpus]
    dids = [str(r["_id"]) for r in corpus]
    return docs, dids, [qmap[q] for q in qids], qids, {q: qrels[q] for q in qids}


def load_model():
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(MODEL, device="cpu", config_kwargs={"vision_config": None, "audio_config": None},
                            model_kwargs={"torch_dtype": torch.float32})
    m.max_seq_length = MAX_LEN
    return m


def main(names):
    torch.set_num_threads(4)
    m = load_model()
    for name in names:
        out = WORK / name; ck = out / "ckpt"; ck.mkdir(parents=True, exist_ok=True)
        if (out / "meta.json").exists():
            print(f"[{name}] done already"); continue
        docs, dids, queries, qids, qrels = load_corpus(name)
        json.dump(dids, open(out / "doc_ids.json", "w")); json.dump(qids, open(out / "query_ids.json", "w"))
        json.dump(qrels, open(out / "qrels.json", "w"))
        print(f"[{name}] docs={len(docs)} queries={len(qids)}", flush=True)
        order = np.argsort([len(d) for d in docs])[::-1]          # longest first: OOM shows up immediately
        t0 = time.time()
        for c0 in range(0, len(docs), CHUNK):
            f = ck / f"{c0:06d}.npz"
            if f.exists():
                continue
            idx = order[c0:c0 + CHUNK]
            E = m.encode([docs[i] for i in idx], batch_size=BATCH, convert_to_numpy=True, normalize_embeddings=True)
            np.savez(f, idx=idx, E=E.astype(np.float32))
            el = time.time() - t0
            print(f"[{name}] {c0 + len(idx):5d}/{len(docs)}  {el/60:5.1f} min", flush=True)
        D = np.zeros((len(docs), 768), np.float32); seen = np.zeros(len(docs), bool)
        for f in sorted(ck.glob("*.npz")):
            z = np.load(f); D[z["idx"]] = z["E"]; seen[z["idx"]] = True
        assert seen.all()
        Q = m.encode(queries, prompt_name=QUERY_PROMPT, batch_size=32, convert_to_numpy=True, normalize_embeddings=True)
        np.save(out / "dense.npy", D); np.save(out / "q_dense.npy", Q.astype(np.float32))
        import transformers, sentence_transformers
        meta = dict(model=MODEL, load="config_kwargs vision_config=None audio_config=None (text-only, 270M)",
                    doc_template="title: {title|none} | text: {text}", query_prompt=QUERY_PROMPT,
                    query_prefix=m.prompts[QUERY_PROMPT], max_seq_length=MAX_LEN, dtype="float32",
                    n_docs=len(docs), n_queries=len(qids), encode_minutes=round((time.time() - t0) / 60, 1),
                    transformers=transformers.__version__, sentence_transformers=sentence_transformers.__version__,
                    torch=torch.__version__)
        json.dump(meta, open(out / "meta.json", "w"), indent=2)
        print(f"[{name}] done {json.dumps(meta)}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ["scifact", "nfcorpus"])
