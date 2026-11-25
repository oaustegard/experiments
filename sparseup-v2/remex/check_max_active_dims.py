"""P12: Sentence Transformers max_active_dims=k vs top-k truncation of a full encode made with the same
batches (SciFact, 300 docs, batch 4). Also measures how far a bf16 encode moves when batch composition
changes (vs the length-sorted cache)."""
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from sentence_transformers import SparseEncoder

import compress as C
from encode import load_beir

HERE = Path(__file__).resolve().parent
torch.set_num_threads(2)
m = SparseEncoder("Linkup-Platform/linkup-sparseup-embed-v2", trust_remote_code=True)
m[0].model.config.doc_max_length = 512
dids, dtext, *_ = load_beir("scifact")
N = 300
full_cached = sp.load_npz(HERE / "cache" / "scifact_docs.npz").tocsr()[:N]
out = {"n_docs": N}
with torch.inference_mode():
    for k in [None, 32, 64, 128]:
        e = m.encode_document(dtext[:N], batch_size=4, max_active_dims=k, convert_to_tensor=True,
                              convert_to_sparse_tensor=False)
        X = sp.csr_matrix(e.float().numpy())
        X.sort_indices()
        if k is None:
            full_run = X
            cos = np.asarray(X.multiply(full_cached).sum(1)).ravel() / np.sqrt(
                np.asarray(X.multiply(X).sum(1)).ravel() * np.asarray(full_cached.multiply(full_cached).sum(1)).ravel())
            out["batch_effect_vs_cache"] = {"rows_identical_pattern": float(np.mean(
                [np.array_equal(X[i].indices, full_cached[i].indices) for i in range(N)])),
                "max_abs_diff": float(abs(X - full_cached).max()), "cos_mean": float(cos.mean()), "cos_min": float(cos.min())}
            print("batch effect", out["batch_effect_vs_cache"], flush=True)
        ref = C.topk_rows(full_run, k)
        same_pattern = np.mean([np.array_equal(X[i].indices, ref[i].indices) for i in range(N)])
        maxdiff = float(abs(X - ref).max())
        nnz = float(np.diff(X.indptr).mean())
        same_values = np.mean([np.array_equal(np.sort(X[i].data), np.sort(ref[i].data)) for i in range(N)])
        tied = np.mean([np.sum(full_run[i].data == np.sort(full_run[i].data)[::-1][k - 1]) > 1
                        if k is not None and full_run[i].nnz > k else False for i in range(N)])
        out[str(k)] = {"rows_identical_pattern": float(same_pattern), "rows_identical_sorted_values": float(same_values),
                       "rows_with_tie_at_kth_value": float(tied), "max_abs_diff": maxdiff, "nnz_mean": nnz}
        print(k, out[str(k)], flush=True)
(HERE / "results" / "max_active_dims_check.json").write_text(json.dumps(out, indent=1))
