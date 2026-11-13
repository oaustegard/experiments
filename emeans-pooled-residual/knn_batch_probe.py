#!/usr/bin/env python3
"""faiss.knn(x, C, 1) cost per distance when the rows come in batches, relative to one call over all rows.

Explains the d = 3 slowdown in grid.py: the e-means batch law picks small batches at low d, and faiss's
brute-force assignment is far less efficient per distance on a batch than on one full pass there.
Random normal data, 4 threads, 1M rows (300k at d = 128). Writes results_knn.json.
"""
import json
import time
from pathlib import Path

import faiss
import numpy as np

faiss.omp_set_num_threads(4)
rng = np.random.default_rng(0)
out = {}
for d in (3, 32, 128):
    X = rng.standard_normal((1_000_000 if d < 128 else 300_000, d)).astype("float32")
    for k in (256, 4096):
        C = X[:k].copy()
        t = time.perf_counter()
        faiss.knn(X, C, 1)
        full = (time.perf_counter() - t) / (len(X) * k)
        rel = {}
        for B in (512, 4096, 32768):
            reps = max(1, len(X) // B // 8)
            t = time.perf_counter()
            for i in range(reps):
                faiss.knn(X[i * B:(i + 1) * B], C, 1)
            rel[str(B)] = round((time.perf_counter() - t) / (reps * B * k) / full, 1)
        out[f"d{d}_k{k}"] = {"full_pass_ns_per_distance": round(full * 1e9, 3), "batch_over_full": rel}
        print(d, k, out[f"d{d}_k{k}"], flush=True)
json.dump(out, open(Path(__file__).resolve().parent / "results_knn.json", "w"), indent=1)
