"""Kaggle T4 job: SPARSEUP-v2 encodes that would take hours on 4 CPUs.

MODE (set below by the pusher via sed) is one of:
  smoke - SciFact first 400 docs + 200 queries in bf16 and fp32, and int8/int4 weight
          quant on the same docs; timings; checks every stage end to end.
  fiqa  - full FiQA corpus + test queries (precision PREC), doc max length 512.
  wq    - step 3: SciFact + NFCorpus docs and queries with fp32 weights, int8
          per-output-row symmetric, and int4 per-row group-32 symmetric fake-quant
          (all nn.Linear weights + the token embedding, which is tied to the decoder).
Outputs: /kaggle/working/<dataset>_<variant>_{docs,queries}.npz (CSR fp32) + meta json.
"""
import json
import subprocess
import sys
import time

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "sentence-transformers==6.3.0",
                "transformers==5.24.1", "datasets"], check=True)

import numpy as np
import scipy.sparse as sp
import torch
from datasets import load_dataset
from sentence_transformers import SparseEncoder

MODE = "smoke"
PREC = "bf16"
OUT = "/kaggle/working"
MODEL = "Linkup-Platform/linkup-sparseup-embed-v2"
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-", flush=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def load_beir(name):
    ds = f"mteb/{name}"
    corpus = load_dataset(ds, "corpus")["corpus"]
    queries = load_dataset(ds, "queries")["queries"]
    qrels = load_dataset(ds, "default")["test"]
    qrel_all = {}
    for r in qrels:
        qrel_all.setdefault(str(r["query-id"]), {})[str(r["corpus-id"])] = int(r["score"])
    qids = [q for q in qrel_all if any(v > 0 for v in qrel_all[q].values())]
    qtext = {str(r["_id"]): r["text"] for r in queries}
    qids = [q for q in qids if q in qtext]
    dids = [str(r["_id"]) for r in corpus]
    dtext = [((r["title"] or "") + " " + (r["text"] or "")).strip() for r in corpus]
    return dids, dtext, qids, [qtext[q] for q in qids], qrel_all


def load_model(dtype):
    m = SparseEncoder(MODEL, trust_remote_code=True, device=DEV, model_kwargs={"dtype": dtype})
    m[0].model.config.doc_max_length = 512
    return m


@torch.no_grad()
def fake_quant(model, scheme):
    """In-place fake quantization of every Linear weight and the token embedding."""
    inner = model[0].model
    n = 0
    for name, mod in inner.named_modules():
        w = None
        if isinstance(mod, (torch.nn.Linear, torch.nn.Embedding)):
            w = mod.weight
        if w is None:
            continue
        W = w.data.float()
        if scheme == "int8":
            s = W.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / 127.0
            Q = torch.clamp(torch.round(W / s), -127, 127) * s
        elif scheme == "int4g32":
            out, inn = W.shape
            pad = (-inn) % 32
            Wp = torch.nn.functional.pad(W, (0, pad)).view(out, -1, 32)
            s = Wp.abs().amax(dim=2, keepdim=True).clamp_min(1e-12) / 7.0
            Q = (torch.clamp(torch.round(Wp / s), -7, 7) * s).view(out, -1)[:, :inn]
        else:
            raise ValueError(scheme)
        w.data.copy_(Q.to(w.dtype))
        n += 1
        print(f"  quant {scheme} {name} {tuple(W.shape)}", flush=True)
    # tied decoder: check it shares storage with the embedding (so it was quantized too)
    dec = [m for nm, m in inner.named_modules() if nm.endswith("decoder")]
    emb = [m for m in inner.modules() if isinstance(m, torch.nn.Embedding)]
    tied = bool(dec and emb and dec[0].weight.data_ptr() == emb[0].weight.data_ptr())
    print(f"  quantized {n} tensors; decoder tied to embedding: {tied}", flush=True)
    return n, tied


def to_csr(t):
    if t.is_sparse:
        t = t.to_dense()
    return sp.csr_matrix(t.float().cpu().numpy(), dtype=np.float32)


def encode_sorted(fn, texts, tok, batch_tokens, max_len, label):
    lens = np.array([min(len(x), max_len) for x in tok(texts, add_special_tokens=False)["input_ids"]])
    order = np.argsort(-lens)
    out = [None] * len(texts)
    i, nb, t0 = 0, 0, time.time()
    while i < len(order):
        L = max(int(lens[order[i]]) + 4, 8)
        bs = max(1, min(256, batch_tokens // L))
        idx = order[i: i + bs]
        with torch.inference_mode():
            emb = fn([texts[j] for j in idx], batch_size=len(idx), convert_to_tensor=True,
                     convert_to_sparse_tensor=False, show_progress_bar=False)
        m = to_csr(emb)
        for k, j in enumerate(idx):
            out[j] = m[k]
        i += bs
        nb += 1
        if nb % 50 == 0:
            print(f"  [{label}] {i}/{len(order)} {time.time()-t0:.0f}s", flush=True)
    return sp.vstack(out).tocsr(), time.time() - t0


def run(model, name, variant, dids, dtext, qids, qtext, qrel, batch_tokens):
    tok = model[0].tokenizer
    Q, tq = encode_sorted(model.encode_query, qtext, tok, batch_tokens, 8192, f"{name}/{variant}/q")
    D, td = encode_sorted(model.encode_document, dtext, tok, batch_tokens, 512, f"{name}/{variant}/d")
    sp.save_npz(f"{OUT}/{name}_{variant}_docs.npz", D)
    sp.save_npz(f"{OUT}/{name}_{variant}_queries.npz", Q)
    meta = {"dataset": name, "variant": variant, "n_docs": len(dids), "n_queries": len(qids),
            "doc_wall_s": round(td, 1), "query_wall_s": round(tq, 1), "device": DEV,
            "gpu": torch.cuda.get_device_name(0) if DEV == "cuda" else None,
            "doc_nnz_mean": float(np.diff(D.indptr).mean()), "query_nnz_mean": float(np.diff(Q.indptr).mean())}
    with open(f"{OUT}/{name}_{variant}_meta.json", "w") as f:
        json.dump({"meta": meta, "dids": dids, "qids": qids, "qrels": qrel}, f)
    print(json.dumps(meta), flush=True)


DT = {"bf16": torch.bfloat16, "fp32": torch.float32}
if MODE == "smoke":
    dids, dtext, qids, qtext, qrel = load_beir("scifact")
    dids, dtext, qids, qtext = dids[:400], dtext[:400], qids[:200], qtext[:200]
    for prec in ["bf16", "fp32"]:
        run(load_model(DT[prec]), "scifact400", prec, dids, dtext, qids, qtext, qrel, 8192)
    for scheme in ["int8", "int4g32"]:
        m = load_model(torch.float32)
        fake_quant(m, scheme)
        run(m, "scifact400", scheme, dids, dtext, qids, qtext, qrel, 8192)
elif MODE == "fiqa":
    dids, dtext, qids, qtext, qrel = load_beir("fiqa")
    run(load_model(DT[PREC]), "fiqa", PREC, dids, dtext, qids, qtext, qrel, 8192)
elif MODE == "wq":
    for name in ["scifact", "nfcorpus"]:
        data = load_beir(name)
        for scheme in ["fp32", "int8", "int4g32"]:
            m = load_model(torch.float32)
            if scheme != "fp32":
                fake_quant(m, scheme)
            run(m, name, scheme, *data, 8192)
            del m
            torch.cuda.empty_cache()
print("DONE", flush=True)
