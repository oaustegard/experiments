"""Encode the fixture's abstracts and every query form with linkup-sparseup-embed-v2.

    python3 sparseup_encode.py fp32 | int8      # -> cache/sparseup_<variant>.npz

Uses the PyTorch reference in ../../sparseup-v2/pack/ref.py (checked against the model's own
modeling_splade.py there: 38 of 47 vectors bit-identical, max |diff| 1.1e-5). int8 = fp32 compute on the
weights read back from the browser pack (../../sparseup-v2/pack/packs/int8, built by pack.py). Documents are
title + " " + abstract with the "[D]" prompt, queries the "[Q]" prompt, both truncated to 512 tokens.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import scipy.sparse as sp
import torch

HERE = Path(__file__).resolve().parent
PACK = HERE.parents[1] / "sparseup-v2" / "pack"
sys.path.insert(0, str(PACK))
from pack import dequant_state_dict
from ref import Reference

variant = sys.argv[1] if len(sys.argv) > 1 else "fp32"
fx = json.loads((HERE / "data" / "fixture.json").read_text(encoding="utf-8"))
docs = [d["title"] + " " + d["abstract"] for d in fx["docs"]]
queries = sorted({q for c in fx["concepts"] for q in c["forms"].values()})

ref = Reference(PACK / "hf", threads=4)
if variant == "int8":
    ref.load_weights(dequant_state_dict(PACK / "packs" / "int8"))


def enc(texts, kind):
    rows, t0 = [], time.time()
    with torch.inference_mode():
        for i, t in enumerate(texts):
            rows.append(sp.csr_matrix(ref.encode_ids(*ref.tokenize(t, kind, 512)).numpy()[None, :]))
            if (i + 1) % 200 == 0:
                print(variant, kind, i + 1, f"{(i + 1) / (time.time() - t0):.1f}/s", flush=True)
    return sp.vstack(rows).tocsr()


Q, D = enc(queries, "query"), enc(docs, "document")
(HERE / "cache").mkdir(exist_ok=True)
sp.save_npz(HERE / "cache" / f"sparseup_{variant}_docs.npz", D)
sp.save_npz(HERE / "cache" / f"sparseup_{variant}_queries.npz", Q)
(HERE / "cache" / f"sparseup_{variant}_queries.json").write_text(json.dumps(queries, ensure_ascii=False))
print(variant, "docs", D.shape, "nnz/doc", D.getnnz() / D.shape[0], "queries", Q.shape, "nnz/q", Q.getnnz() / Q.shape[0])
