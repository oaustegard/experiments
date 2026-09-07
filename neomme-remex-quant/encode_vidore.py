#!/usr/bin/env python3
"""Encode a ViDoRe v1 task (page images + text queries) with NeoMME-260M-Retriever, both heads.

Usage: python3 encode_vidore.py docvqa|shift
Reads data/vidore/<task>.parquet (columns: query, image{bytes}, image_filename).
Corpus = unique pages by image_filename; queries = rows with a non-null query;
qrels = each query's own page (ViDoRe v1 protocol, one relevant page per query).

Writes data/vidore_<task>_enc/ in the same layout as encode.py (dense.npy,
mv_tokens.npy float16, mv_offsets.npy, q_*.npy, doc_ids.json, query_ids.json)
plus qrels.tsv (query-id, corpus-id, score) so bench.py can consume it with
--data. Resumable via checkpoints/<task>_*.npz. Pages are batched one at a time
(a 2048-px page is up to 4,096 patches; padding across pages would waste most
of the batch), so no length sorting is needed.
"""
from __future__ import annotations

import io, json, sys, time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import encode as E  # load_model, MODEL

task = sys.argv[1]
SRC = HERE / "data" / "vidore" / f"{task}.parquet"
OUT = HERE / "data" / f"vidore_{task}_enc"; OUT.mkdir(parents=True, exist_ok=True)
CKPT = HERE / "checkpoints"; CKPT.mkdir(exist_ok=True)


def encode_images(proc, model, images):
    msgs = [[{"role": "user", "content": [{"type": "image", "image": im}]}] for im in images]
    batch = proc.apply_chat_template(msgs, task="document", tokenize=True, return_dict=True, return_tensors="pt", processor_kwargs={"padding": "longest"})
    with torch.inference_mode():
        out = model(**batch)
    mask = batch["attention_mask"].bool()
    dense = out.dense_embeddings.float().numpy()
    toks = [out.embeddings[i][mask[i]].float().numpy().astype(np.float16) for i in range(len(images))]
    return dense, toks


def main():
    tbl = pq.read_table(SRC).to_pandas()
    pages = tbl.drop_duplicates("image_filename").reset_index(drop=True)
    doc_ids = list(pages.image_filename)
    qrows = tbl[tbl["query"].notna()].reset_index(drop=True)
    qids = [f"q{i}" for i in range(len(qrows))]
    (OUT / "qrels.tsv").write_text("query-id\tcorpus-id\tscore\n" + "".join(f"{q}\t{fn}\t1\n" for q, fn in zip(qids, qrows.image_filename)))
    (OUT / "doc_ids.json").write_text(json.dumps(doc_ids)); (OUT / "query_ids.json").write_text(json.dumps(qids))
    proc, model = E.load_model()
    n = len(pages); t0 = time.time(); B = 2
    for b0 in range(0, n, B):
        ck = CKPT / f"{task}_{b0:05d}.npz"
        if ck.exists():
            continue
        ims = [Image.open(io.BytesIO(pages.image.iloc[i]["bytes"])).convert("RGB") for i in range(b0, min(n, b0 + B))]
        dense, toks = encode_images(proc, model, ims)
        np.savez(ck, idx=np.arange(b0, b0 + len(ims)), dense=dense, toks=np.concatenate(toks), lens=np.array([len(t) for t in toks]))
        el = time.time() - t0
        print(f"[{b0 + len(ims):4d}/{n}] {el/60:5.1f} min, {(b0 + len(ims))/el:.2f} pages/s, tokens {[len(t) for t in toks]} px {[im.size for im in ims]}", flush=True)
    dense = np.zeros((n, 1024), np.float32); toks = [None] * n
    for ck in sorted(CKPT.glob(f"{task}_*.npz")):
        z = np.load(ck); lens = z["lens"]; st = np.concatenate([[0], np.cumsum(lens)])
        for j, i in enumerate(z["idx"]):
            dense[i] = z["dense"][j]; toks[i] = z["toks"][st[j]:st[j + 1]]
    assert all(t is not None for t in toks)
    off = np.concatenate([[0], np.cumsum([len(t) for t in toks])]).astype(np.int64)
    np.save(OUT / "dense.npy", dense); np.save(OUT / "mv_tokens.npy", np.concatenate(toks)); np.save(OUT / "mv_offsets.npy", off)
    qd, qt = [], []
    for b0 in range(0, len(qrows), 32):
        d, t, _ = E.encode_batch(proc, model, list(qrows["query"].iloc[b0:b0 + 32]), "query"); qd.append(d); qt.extend(t)
    np.save(OUT / "q_dense.npy", np.concatenate(qd)); np.save(OUT / "q_mv_tokens.npy", np.concatenate(qt))
    np.save(OUT / "q_mv_offsets.npy", np.concatenate([[0], np.cumsum([len(t) for t in qt])]).astype(np.int64))
    import transformers
    meta = dict(model=E.MODEL, task=f"vidore/{task}", heads=E.__doc__.split("\n")[0], dtype="float32 compute; multi-vector stored float16", image_max_side=2048, patch_size=32,
                batch=B, n_pages=n, n_queries=len(qids), n_page_tokens=int(off[-1]), tokens_per_page=float(off[-1] / n), transformers=transformers.__version__, torch=torch.__version__,
                wall_minutes=round((time.time() - t0) / 60, 1), threads=torch.get_num_threads(), qrels="one relevant page per query (its source page), ViDoRe v1 protocol")
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2)); print("done", json.dumps(meta))


if __name__ == "__main__":
    main()
