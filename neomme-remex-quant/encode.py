#!/usr/bin/env python3
"""Encode BEIR SciFact with NeoMME-260M-Retriever, both heads, one forward pass.

Writes data/scifact_enc/:
  dense.npy        (n_docs, 1024) float32, L2-normalised (the model's own Normalize)
  mv_tokens.npy    (n_tokens_total, 128) float16, L2-normalised per token; padding rows dropped
  mv_offsets.npy   (n_docs + 1,) int64 — doc i owns rows offsets[i]:offsets[i+1]
  q_dense.npy, q_mv_tokens.npy, q_mv_offsets.npy — same for the 300 judged test queries
  doc_ids.json, query_ids.json
  meta.json        encode configuration (METHODS.md: an embedding artifact without
                   its encode settings is write-only)

Resumable: each doc batch lands in checkpoints/ before assembly. Docs are
sorted by character length before batching to cut padding (exact, not approximate:
the model is bidirectional with an attention mask, so batch composition does not
change a row's vectors beyond float noise — asserted by --parity-check).
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "scifact"
OUT = HERE / "data" / "scifact_enc"
CKPT = HERE / "checkpoints"
MODEL = "Hcompany/NeoMME-260M-Retriever"
MAX_LEN = 1024          # tokens; SciFact p95 is ~560 tokens, max ~2300
BATCH = 16


def load_model():
    from transformers import NeoMMEForRetrieval, NeoMMEProcessor
    torch.set_num_threads(max(1, torch.get_num_threads()))
    proc = NeoMMEProcessor.from_pretrained(MODEL)
    model = NeoMMEForRetrieval.from_pretrained(MODEL, dtype=torch.float32).eval()
    return proc, model


def encode_batch(proc, model, texts, task):
    msgs = [[{"role": "user", "content": t}] for t in texts]
    batch = proc.apply_chat_template(
        msgs, task=task, tokenize=True, return_dict=True, return_tensors="pt",
        processor_kwargs={"padding": "longest", "truncation": True, "max_length": MAX_LEN},
    )
    with torch.inference_mode():
        out = model(**batch)
    mask = batch["attention_mask"].bool()
    dense = out.dense_embeddings.float().numpy()
    toks = [out.embeddings[i][mask[i]].float().numpy().astype(np.float16) for i in range(len(texts))]
    return dense, toks, mask.sum(1).tolist()


def corpus_texts():
    c = pd.read_parquet(DATA / "corpus.parquet")
    texts = [(t + " " + x).strip() for t, x in zip(c.title, c.text)]
    return list(c["_id"].astype(str)), texts


def judged_queries():
    q = pd.read_parquet(DATA / "queries.parquet")
    qrels = pd.read_csv(DATA / "qrels" / "test.tsv", sep="\t")
    judged = set(qrels["query-id"].astype(str))
    q = q[q["_id"].astype(str).isin(judged)]
    return list(q["_id"].astype(str)), list(q.text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parity-check", action="store_true", help="encode 8 docs alone and in a padded batch; report max abs diff")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True); CKPT.mkdir(exist_ok=True)
    proc, model = load_model()

    if args.parity_check:
        _, texts = corpus_texts()
        short = [t for t in texts if len(t) < 600][:4]; long_ = [t for t in texts if len(t) > 2000][:4]
        d_alone, t_alone, _ = encode_batch(proc, model, short, "document")
        d_mixed, t_mixed, _ = encode_batch(proc, model, short + long_, "document")
        print("dense max|diff|", float(np.abs(d_alone - d_mixed[:4]).max()))
        print("mv    max|diff|", max(float(np.abs(a.astype(np.float32) - b.astype(np.float32)).max()) for a, b in zip(t_alone, t_mixed[:4])))
        return

    doc_ids, texts = corpus_texts()
    order = np.argsort([len(t) for t in texts])[::-1]      # longest first: OOM shows up immediately
    n = len(texts); t0 = time.time(); done_tokens = 0
    for b0 in range(0, n, BATCH):
        idx = order[b0:b0 + BATCH]
        ck = CKPT / f"doc_{b0:05d}.npz"
        if ck.exists():
            continue
        dense, toks, lens = encode_batch(proc, model, [texts[i] for i in idx], "document")
        np.savez(ck, idx=idx, dense=dense, toks=np.concatenate(toks), lens=np.array([len(t) for t in toks]))
        done_tokens += sum(lens)
        el = time.time() - t0
        print(f"[{b0 + len(idx):5d}/{n}] {el/60:5.1f} min, {(b0 + len(idx))/el:.2f} docs/s, batch max_len {max(lens)}", flush=True)

    # assemble in corpus order
    dense = np.zeros((n, 1024), np.float32); toks = [None] * n
    for ck in sorted(CKPT.glob("doc_*.npz")):
        z = np.load(ck); lens = z["lens"]; starts = np.concatenate([[0], np.cumsum(lens)])
        for j, i in enumerate(z["idx"]):
            dense[i] = z["dense"][j]; toks[i] = z["toks"][starts[j]:starts[j + 1]]
    assert all(t is not None for t in toks)
    offsets = np.concatenate([[0], np.cumsum([len(t) for t in toks])]).astype(np.int64)
    np.save(OUT / "dense.npy", dense); np.save(OUT / "mv_tokens.npy", np.concatenate(toks)); np.save(OUT / "mv_offsets.npy", offsets)
    (OUT / "doc_ids.json").write_text(json.dumps(doc_ids))

    qids, qtexts = judged_queries()
    qd, qt = [], []
    for b0 in range(0, len(qtexts), 32):
        d, t, _ = encode_batch(proc, model, qtexts[b0:b0 + 32], "query"); qd.append(d); qt.extend(t)
    np.save(OUT / "q_dense.npy", np.concatenate(qd)); np.save(OUT / "q_mv_tokens.npy", np.concatenate(qt))
    np.save(OUT / "q_mv_offsets.npy", np.concatenate([[0], np.cumsum([len(t) for t in qt])]).astype(np.int64))
    (OUT / "query_ids.json").write_text(json.dumps(qids))

    import transformers, sentence_transformers
    meta = dict(model=MODEL, heads=["dense_embeddings (1024, mean-pooled, L2-normalised)", "embeddings (128 per token, L2-normalised, MeanMaxSim)"],
                dtype="float32 compute; multi-vector stored float16", max_length_tokens=MAX_LEN, batch=BATCH, padding="longest, length-sorted",
                template="processor.apply_chat_template(task=document|query)", corpus="BeIR/scifact corpus parquet (5183 docs, title + ' ' + text)",
                queries="BeIR/scifact queries restricted to test-qrels judged ids", n_docs=n, n_queries=len(qids), n_doc_tokens=int(offsets[-1]),
                transformers=transformers.__version__, sentence_transformers=sentence_transformers.__version__, torch=torch.__version__,
                wall_minutes=round((time.time() - t0) / 60, 1), threads=torch.get_num_threads())
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2))
    print("done", json.dumps(meta))


if __name__ == "__main__":
    main()
