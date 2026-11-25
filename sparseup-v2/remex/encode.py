"""Step 1: encode BEIR sets with SPARSEUP-v2 at full output, cache CSR fp32.

Usage: python3 encode.py scifact [nfcorpus fiqa arguana] [--doc-max-len 512] [--subsample N]
Docs: title + " " + text via encode_document ([D]); queries via encode_query ([Q]).
No max_active_dims cap. bf16 weights (the card's default), fp32 output.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from datasets import load_dataset
from sentence_transformers import SparseEncoder

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
MODEL = "Linkup-Platform/linkup-sparseup-embed-v2"
HF_NAME = {"scifact": "mteb/scifact", "nfcorpus": "mteb/nfcorpus", "fiqa": "mteb/fiqa", "arguana": "mteb/arguana"}


def load_beir(name):
    ds = HF_NAME[name]
    corpus = load_dataset(ds, "corpus")["corpus"]
    queries = load_dataset(ds, "queries")["queries"]
    qrels = load_dataset(ds, "default")["test"]
    qrel = {}
    for r in qrels:
        if int(r["score"]) > 0 or name == "arguana":
            qrel.setdefault(str(r["query-id"]), {})[str(r["corpus-id"])] = int(r["score"])
    # keep graded scores as given (incl. 0 if present) for nDCG; drop queries without positives later
    qrel_all = {}
    for r in qrels:
        qrel_all.setdefault(str(r["query-id"]), {})[str(r["corpus-id"])] = int(r["score"])
    qids = [q for q in qrel_all if any(v > 0 for v in qrel_all[q].values())]
    qtext = {str(r["_id"]): r["text"] for r in queries}
    qids = [q for q in qids if q in qtext]
    dids = [str(r["_id"]) for r in corpus]
    dtext = [((r["title"] or "") + " " + (r["text"] or "")).strip() for r in corpus]
    return dids, dtext, qids, [qtext[q] for q in qids], qrel_all


def to_csr(t):
    if t.is_sparse:
        t = t.to_dense()
    return sp.csr_matrix(t.float().cpu().numpy(), dtype=np.float32)


def encode_sorted(fn, texts, tok, batch_tokens, max_len):
    lens = np.array([min(len(x), max_len) for x in tok(texts, add_special_tokens=False)["input_ids"]])
    order = np.argsort(-lens)
    out = [None] * len(texts)
    i = 0
    t0 = time.time()
    while i < len(order):
        L = max(int(lens[order[i]]) + 4, 8)
        bs = max(1, min(256, batch_tokens // L))
        idx = order[i : i + bs]
        with torch.inference_mode():
            emb = fn([texts[j] for j in idx], batch_size=len(idx), convert_to_tensor=True, convert_to_sparse_tensor=False, show_progress_bar=False)
        m = to_csr(emb)
        for k, j in enumerate(idx):
            out[j] = m[k]
        i += bs
        if (i // bs) % 20 == 0:
            print(f"  {i}/{len(order)} {time.time()-t0:.0f}s", flush=True)
    return sp.vstack(out).tocsr(), time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="+")
    ap.add_argument("--doc-max-len", type=int, default=512)
    ap.add_argument("--batch-tokens", type=int, default=16384)
    ap.add_argument("--subsample", type=int, default=0, help="keep qrels docs + N random docs (seed 0)")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N docs only")
    a = ap.parse_args()
    torch.set_num_threads(4)
    model = SparseEncoder(MODEL, trust_remote_code=True)
    inner = model[0].model
    inner.config.doc_max_length = a.doc_max_len
    tok = model[0].tokenizer
    CACHE.mkdir(exist_ok=True)
    for name in a.datasets:
        dids, dtext, qids, qtext, qrel = load_beir(name)
        note = {}
        if a.subsample:
            rel = {d for q in qids for d, s in qrel[q].items() if s > 0}
            rng = np.random.default_rng(0)
            pool = [i for i, d in enumerate(dids) if d not in rel]
            keep = sorted({i for i, d in enumerate(dids) if d in rel} | set(rng.choice(pool, a.subsample, replace=False).tolist()))
            note = {"subsampled": True, "n_full": len(dids), "n_kept": len(keep)}
            dids = [dids[i] for i in keep]
            dtext = [dtext[i] for i in keep]
        if a.limit:
            dids, dtext = dids[: a.limit], dtext[: a.limit]
        print(f"{name}: {len(dids)} docs, {len(qids)} queries", flush=True)
        Q, tq = encode_sorted(model.encode_query, qtext, tok, a.batch_tokens, 8192)
        D, td = encode_sorted(model.encode_document, dtext, tok, a.batch_tokens, a.doc_max_len)
        tag = name + (f"_lim{a.limit}" if a.limit else "")
        sp.save_npz(CACHE / f"{tag}_docs.npz", D)
        sp.save_npz(CACHE / f"{tag}_queries.npz", Q)
        meta = {"dataset": name, "hf": HF_NAME[name], "n_docs": len(dids), "n_queries": len(qids),
                "doc_max_len": a.doc_max_len, "doc_wall_s": round(td, 1), "query_wall_s": round(tq, 1),
                "doc_nnz_mean": float(np.diff(D.indptr).mean()), "query_nnz_mean": float(np.diff(Q.indptr).mean()),
                "dtype": str(next(inner.parameters()).dtype), "torch_threads": torch.get_num_threads(), **note}
        (CACHE / f"{tag}_meta.json").write_text(json.dumps({"meta": meta, "dids": dids, "qids": qids, "qrels": qrel}))
        print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
