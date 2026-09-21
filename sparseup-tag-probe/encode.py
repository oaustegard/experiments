"""Encode every pinned memory with SPARSEUP (encode_document) and gte-small.

Writes gitignored data/ artifacts, checkpointing per batch so a killed run resumes:
  data/parts/sparse_{i:05d}.npz   scipy CSR rows for that batch (SPARSEUP, weights kept)
  data/parts/dense_{i:05d}.npy    gte-small embeddings for that batch
  data/sparse.npz, data/dense.npy merged at the end
  data/encode_meta.json           truncation counts, dims, timings
  data/topk.json                  per memory: top-64 (token, weight) decoded, for arm 1
"""
import json, sys, time
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import torch
from common import DATA

SPARSE_MODEL = "Linkup-Platform/linkup-sparseup-embed-v1"
DENSE_MODEL = "thenlper/gte-small"
BATCH = 64
TOPK = 64


def main():
    from sentence_transformers import SparseEncoder, SentenceTransformer
    torch.set_num_threads(4)
    t = json.loads((DATA / "texts.json").read_text())
    ids, texts = t["ids"], t["texts"]
    parts = DATA / "parts"; parts.mkdir(exist_ok=True)
    enc = SparseEncoder(SPARSE_MODEL, trust_remote_code=True, device="cpu")
    tok = enc.tokenizer
    max_len = 512
    enc.max_seq_length = max_len
    dense = SentenceTransformer(DENSE_MODEL, device="cpu")
    dense.max_seq_length = 512

    # Truncation census on the document prompt the model prepends.
    prompt = enc.prompts.get("document", "")
    n_tok = [len(tok(prompt + s, add_special_tokens=True)["input_ids"]) for s in texts]
    truncated = int(sum(1 for n in n_tok if n > max_len))
    (DATA / "encode_meta.json").write_text(json.dumps({
        "n": len(ids), "max_seq_length": max_len, "n_truncated": truncated,
        "token_counts_quantiles": {q: int(np.quantile(n_tok, q)) for q in (0.5, 0.75, 0.9, 0.95, 0.99)},
        "sparse_model": SPARSE_MODEL, "dense_model": DENSE_MODEL, "status": "running"}, indent=1))
    print(f"{len(ids)} docs; {truncated} exceed {max_len} tokens", flush=True)

    t0 = time.time()
    for b, i in enumerate(range(0, len(ids), BATCH)):
        sp_path, de_path = parts / f"sparse_{b:05d}.npz", parts / f"dense_{b:05d}.npy"
        if sp_path.exists() and de_path.exists():
            continue
        chunk = texts[i:i + BATCH]
        with torch.inference_mode():
            emb = enc.encode_document(chunk, batch_size=16, convert_to_sparse_tensor=True)
        coo = emb.coalesce()
        idx, val = coo.indices().numpy(), coo.values().numpy().astype(np.float32)
        mat = sp.csr_matrix((val, (idx[0], idx[1])), shape=tuple(coo.shape))
        sp.save_npz(sp_path, mat)
        np.save(de_path, dense.encode(chunk, batch_size=16, normalize_embeddings=True).astype(np.float32))
        done = min(i + BATCH, len(ids))
        el = time.time() - t0
        print(f"batch {b} done {done}/{len(ids)} {el/60:.1f} min elapsed, "
              f"eta {el/done*(len(ids)-done)/60:.1f} min", flush=True)

    n_parts = (len(ids) + BATCH - 1) // BATCH
    S = sp.vstack([sp.load_npz(parts / f"sparse_{b:05d}.npz") for b in range(n_parts)]).tocsr()
    D = np.vstack([np.load(parts / f"dense_{b:05d}.npy") for b in range(n_parts)])
    assert S.shape[0] == len(ids) == D.shape[0]
    sp.save_npz(DATA / "sparse.npz", S)
    np.save(DATA / "dense.npy", D)

    vocab = {v: k for k, v in tok.get_vocab().items()}
    topk = []
    for r in range(S.shape[0]):
        row = S.getrow(r)
        order = np.argsort(-row.data)[:TOPK]
        topk.append([(vocab[int(row.indices[j])], float(row.data[j])) for j in order])
    (DATA / "topk.json").write_text(json.dumps({"ids": ids, "topk": topk}))
    meta = json.loads((DATA / "encode_meta.json").read_text())
    nnz = np.diff(S.indptr)
    meta.update({"status": "done", "sparse_dims": int(S.shape[1]),
                 "active_dims_used": int(len(np.unique(S.indices))),
                 "nnz_per_doc": {"mean": float(nnz.mean()), "median": float(np.median(nnz)),
                                 "min": int(nnz.min()), "max": int(nnz.max())},
                 "dense_dims": int(D.shape[1]), "wall_minutes": round((time.time() - t0) / 60, 1)})
    (DATA / "encode_meta.json").write_text(json.dumps(meta, indent=1))
    print("done", json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
