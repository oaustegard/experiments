#!/usr/bin/env python3
"""Where does e-means pay? Time-to-error against Lloyd over a grid of (n, k, d).

Model being tested (see PLAN.md, 'grid'):
    speedup ~= (I_L / P_e) / (1 + rho),   rho = t_o / (B * k * d * t_f)
I_L Lloyd iterations replaced, P_e e-means passes, t_o fixed cost per mini-batch, B batch size,
t_f time per multiply-add of the assignment.

Data per d: a synthetic Gaussian mixture (1,000 components, Dirichlet(1) weights, per-component
scale log-uniform over 4x, centres N(0, I)) so cluster structure exists at every k, plus two real
anchors: Starry Night pixels (d=3) and Ward-pooled LateOn SciFact tokens pf4 (d=128) if present.

Per cell (n, k, d), shared random init (k rows), 4 threads:
  lloyd   faiss.Kmeans stepped one iteration at a time to 25; wall time and train MSE per step
  np-e    faiss.contrib.e_means.NumpyEmeans, nepochs 1, 2, 3, 5 (separate runs)
  torch-e faiss.contrib.e_means.Emeans on CPU tensors, same budgets (the compiled-loop stand-in)
Cells with n*k*d > 2.5e11 (one Lloyd step over ~5 s) or k > n/8 are skipped and listed.
MSE is exact nearest-centroid error over all n rows for every method.

One JSON line per cell in results_grid.jsonl; resumable. Prints 'GRID DONE'.
"""
import json
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np  # noqa: E402
import faiss  # noqa: E402
import torch  # noqa: E402
from faiss.contrib.e_means import NumpyEmeans, Emeans  # noqa: E402

faiss.omp_set_num_threads(4)
torch.set_num_threads(4)
HERE = Path(__file__).resolve().parent
RES = HERE / "results_grid.jsonl"
LLOYD_ITERS = 25
EPOCHS = (1, 2, 3, 5)
MAX_WORK = 2.5e11

done = {json.loads(l)["cell"] for l in open(RES)} if RES.exists() else set()


def gmm(n, d, seed=0, comps=1000):
    rng = np.random.default_rng(seed)
    mu = rng.standard_normal((comps, d)).astype(np.float32)
    w = rng.dirichlet(np.ones(comps))
    sc = np.exp(rng.uniform(np.log(0.15), np.log(0.6), comps)).astype(np.float32)
    z = rng.choice(comps, n, p=w)
    return (mu[z] + rng.standard_normal((n, d)).astype(np.float32) * sc[z, None]).astype(np.float32)


def starry():
    from PIL import Image
    im = Image.open(HERE.parent / "svg-in-browser" / "web" / "starry.jpg").convert("RGB")
    im = im.resize((1000, round(im.height * 1000 / im.width)), Image.LANCZOS)
    return np.asarray(im, dtype=np.float32).reshape(-1, 3) / 255.0


def tokens():
    f = HERE / ".work" / "scifact" / "pf4.npz"
    return np.load(f)["X"] if f.exists() else None


def mse(X, C):
    D, _ = faiss.knn(X, np.ascontiguousarray(C, dtype=np.float32), 1)
    return float(D.mean())


def run_cell(tag, X, k, seed=0):
    n, d = X.shape
    cell = f"{tag}|n{n}|k{k}|d{d}"
    if cell in done:
        return
    if k > n // 8 or n * k * d > MAX_WORK:
        rec = {"cell": cell, "skipped": True, "n": n, "k": k, "d": d}
        open(RES, "a").write(json.dumps(rec) + "\n")
        return
    rng = np.random.default_rng(seed)
    init = X[np.sort(rng.choice(n, k, replace=False))].copy()
    rec = {"cell": cell, "data": tag, "n": n, "k": k, "d": d, "init_mse": mse(X, init)}
    C, t, curve = init.copy(), 0.0, []
    for _ in range(LLOYD_ITERS):
        t0 = time.perf_counter()
        km = faiss.Kmeans(d, k, niter=1, seed=seed + 1, max_points_per_centroid=10 ** 9, verbose=False)
        km.train(X, init_centroids=C)
        C = km.centroids.copy()
        t += time.perf_counter() - t0
        curve.append([round(t, 4), mse(X, C)])
    rec["lloyd"] = curve
    t0 = time.perf_counter()
    faiss.knn(X, init, 1)
    rec["one_assign_pass_s"] = round(time.perf_counter() - t0, 4)
    Xt = torch.from_numpy(X)
    for name in ("np", "torch"):
        pts = []
        for ep in EPOCHS:
            t0 = time.perf_counter()
            if name == "np":
                e = NumpyEmeans(d, k, nepochs=ep, seed=seed)
                Ce, _ = e.train(X, init_centroids=init.copy())
            else:
                e = Emeans(d, k, nepochs=ep, seed=seed)
                Ce, _ = e.train(Xt, init_centroids=torch.from_numpy(init.copy()))
                Ce = Ce.numpy() if hasattr(Ce, "numpy") else np.asarray(Ce)
            el = time.perf_counter() - t0
            pts.append([ep, round(el, 4), mse(X, Ce)])
        rec[f"emeans_{name}"] = pts
        rec[f"batch_{name}"] = int(e.recipe["batch_size"])
    l25 = curve[-1][1]
    for name in ("np", "torch"):
        hit = [p for p in rec[f"emeans_{name}"] if p[2] <= l25]
        rec[f"{name}_time_to_lloyd25"] = hit[0][1] if hit else None
        rec[f"{name}_best_mse_over_lloyd25"] = min(p[2] for p in rec[f"emeans_{name}"]) / l25
    rec["lloyd25_s"] = curve[-1][0]
    with open(RES, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    done.add(cell)
    print(json.dumps({k_: v for k_, v in rec.items() if k_ not in ("lloyd", "emeans_np", "emeans_torch")}),
          flush=True)


KS = (16, 64, 256, 1024, 4096, 16384)
for d in (3, 32, 128, 768):
    for n in (10_000, 100_000, 1_000_000):
        if n * d * 4 > 1.6e9:
            continue
        X = gmm(n, d)
        for k in KS:
            run_cell("gmm", X, k)
        del X
for tag, loader in (("starry", starry), ("lateon-pf4", tokens)):
    X = loader()
    if X is None:
        print(f"{tag}: no data, skipped", flush=True)
        continue
    for k in KS:
        run_cell(tag, np.ascontiguousarray(X, dtype=np.float32), k)
print("GRID DONE", flush=True)
