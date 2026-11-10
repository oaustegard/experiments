#!/usr/bin/env python3
"""e-means against Lloyd's k-means on one embedding set, from shared initial centroids.

Trains on the first --ntrain rows of docs.npy and holds out the rest for test MSE.
For each (k, seed):
  lloyd   faiss.Kmeans stepped one iteration at a time from the shared init, so the
          centroids after every iteration are available (25 iterations)
  emeans  faiss.contrib.e_means.NumpyEmeans, one fresh run per epoch budget T, since
          its batch size, step size and cosine schedule all depend on T
MSE is the mean squared distance from a point to its nearest centroid, evaluated with
an exact assignment of the final centroids (never the moving-pass objective).

Writes one JSON line per run to OUT/runs.jsonl as it goes and skips runs already
there, so a killed job resumes. Centroid snapshots go to OUT/cent/.

This script imports the e-means source from a faiss checkout (--emeans-src, the
directory that contains e_means/), which is third-party code: run it in the jail.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--data", required=True, help="docs.npy, (n, d) float32")
p.add_argument("--out", required=True)
p.add_argument("--emeans-src", required=True)
p.add_argument("--ntrain", type=int, default=90000)
p.add_argument("--ks", default="16,64,256,1024,4096")
p.add_argument("--seeds", default="0,1,2")
p.add_argument("--lloyd-iters", type=int, default=25)
p.add_argument("--emeans-T", default="1,2,3,5,10")
p.add_argument("--save-lloyd", default="2,5,25")
p.add_argument("--save-emeans", default="2,5")
p.add_argument("--threads", type=int, default=4)
p.add_argument("--smoke", action="store_true")
args = p.parse_args()

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = str(args.threads)

import numpy as np  # noqa: E402
import faiss  # noqa: E402

sys.path.insert(0, args.emeans_src)
from e_means import NumpyEmeans  # noqa: E402

faiss.omp_set_num_threads(args.threads)
OUT = Path(args.out)
(OUT / "cent").mkdir(parents=True, exist_ok=True)
LOG = OUT / "runs.jsonl"

X = np.load(args.data, mmap_mode="r")
if args.smoke:
    Xtr = np.ascontiguousarray(X[:9000], dtype=np.float32)
    Xte = np.ascontiguousarray(X[9000:10000], dtype=np.float32)
else:
    Xtr = np.ascontiguousarray(X[: args.ntrain], dtype=np.float32)
    Xte = np.ascontiguousarray(X[args.ntrain:], dtype=np.float32)
n, d = Xtr.shape
ks = [int(k) for k in args.ks.split(",")]
seeds = [int(s) for s in args.seeds.split(",")]
emeans_T = [float(t) for t in args.emeans_T.split(",")]
save_lloyd = {int(t) for t in args.save_lloyd.split(",")}
save_emeans = {float(t) for t in args.save_emeans.split(",")}

done = set()
if LOG.exists():
    for line in LOG.read_text().splitlines():
        r = json.loads(line)
        done.add((r["arm"], r["k"], r["seed"], r["budget"]))


def evaluate(C):
    """Exact-assignment train and test MSE, dead clusters and size imbalance."""
    dtr, ltr = faiss.knn(Xtr, C, 1)
    dte, _ = faiss.knn(Xte, C, 1)
    sizes = np.bincount(ltr.ravel(), minlength=len(C)).astype(np.float64)
    return {
        "train_mse": float(dtr.mean(dtype=np.float64)),
        "test_mse": float(dte.mean(dtype=np.float64)),
        "empty": int((sizes == 0).sum()),
        # the e-means post counts a cluster dead below a thousandth of the mean size;
        # at these n/k that rounds to zero points, so a tenth of the mean is reported too
        "small": int((sizes < 0.1 * n / len(C)).sum()),
        "imbalance": float(len(C) * (sizes ** 2).sum() / n ** 2),
    }


def emit(rec):
    with LOG.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)


def save(arm, k, seed, budget, C):
    np.save(OUT / "cent" / f"{arm}_k{k}_s{seed}_t{budget:g}.npy", C.astype(np.float32))


for k in ks:
    for seed in seeds:
        init = Xtr[np.random.RandomState(1000 + seed).permutation(n)[:k]].copy()
        if ("init", k, seed, 0) not in done:
            save("init", k, seed, 0, init)
            emit({"arm": "init", "k": k, "seed": seed, "budget": 0, "seconds": 0.0,
                  **evaluate(init)})

        # Lloyd, stepped. A resumed job restarts this (k, seed) from the init.
        if ("lloyd", k, seed, args.lloyd_iters) not in done:
            C = init.copy()
            elapsed = 0.0
            for it in range(1, args.lloyd_iters + 1):
                km = faiss.Kmeans(d, k, niter=1, seed=seed + 1,
                                  max_points_per_centroid=10 ** 9,
                                  min_points_per_centroid=1)
                t0 = time.perf_counter()
                km.train(Xtr, init_centroids=C)
                elapsed += time.perf_counter() - t0
                C = km.centroids.copy()
                if ("lloyd", k, seed, it) in done:
                    continue
                if it in save_lloyd:
                    save("lloyd", k, seed, it, C)
                emit({"arm": "lloyd", "k": k, "seed": seed, "budget": it,
                      "seconds": elapsed, **evaluate(C)})

        # Lloyd in one uninterrupted call, for a wall-clock figure without the
        # per-iteration setup of the stepped loop.
        if seed == seeds[0] and ("lloyd_direct", k, seed, args.lloyd_iters) not in done:
            km = faiss.Kmeans(d, k, niter=args.lloyd_iters, seed=seed + 1,
                              max_points_per_centroid=10 ** 9,
                              min_points_per_centroid=1)
            t0 = time.perf_counter()
            km.train(Xtr, init_centroids=init.copy())
            sec = time.perf_counter() - t0
            emit({"arm": "lloyd_direct", "k": k, "seed": seed,
                  "budget": args.lloyd_iters, "seconds": sec,
                  **evaluate(km.centroids.copy())})

        for T in emeans_T:
            if ("emeans", k, seed, T) in done:
                continue
            em = NumpyEmeans(d, k, nepochs=T, seed=seed)
            t0 = time.perf_counter()
            C, _ = em.train(Xtr, init_centroids=init.copy())
            sec = time.perf_counter() - t0
            if T in save_emeans:
                save("emeans", k, seed, T, C)
            emit({"arm": "emeans", "k": k, "seed": seed, "budget": T, "seconds": sec,
                  "batch_size": int(em.recipe["batch_size"]),
                  "alpha0": float(em.recipe["alpha0"]),
                  "updates": int(em.recipe["total_steps"]),
                  "splits": int(em.n_splits), **evaluate(C)})
print("DONE", flush=True)
