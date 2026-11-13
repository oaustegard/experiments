#!/usr/bin/env python3
"""Side check: does e-means reach Lloyd's colour-quantization error in fewer passes over the pixels?

The browser tracer (svg-in-browser/web/trace.js) runs up to 20 full Lloyd passes over every pixel
at K=56 after k-means++ seeding on a 20k-pixel sample; k-means is 2.8 of its 3.3 s in JS. Its cost
is one nearest-of-K search per pixel per pass, so the currency here is passes over the data, not
Python wall-clock (numpy batch overheads would not carry over to a JS loop).

Same k-means++ init for both. Lloyd: exact assign + mean update, MSE after each pass. e-means:
faiss NumpyEmeans at nepochs 1, 2, 3, 5 from that init (d=3, so the automatic recipe picks the
'quality' batch law and keeps revival on). MSE is always the exact nearest-centroid error over all
pixels, in RGB units squared per pixel. Images: svg-in-browser/web/starry.jpg and dog.jpg at 1000 px
wide, after the tracer's default blur is skipped (raw pixels; noted in RESULTS).

Writes results_colour.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import faiss
from PIL import Image
from faiss.contrib.e_means import NumpyEmeans

HERE = Path(__file__).resolve().parent
IMG = HERE.parent / "svg-in-browser" / "web"
faiss.omp_set_num_threads(1)


def pixels(name, width=1000):
    im = Image.open(IMG / name).convert("RGB")
    h = round(im.height * width / im.width)
    return np.asarray(im.resize((width, h), Image.LANCZOS), dtype=np.float32).reshape(-1, 3)


def kpp(X, K, seed, m=20000):
    rng = np.random.default_rng(seed)
    S = X[rng.integers(0, len(X), m)]
    C = [S[rng.integers(m)]]
    d = ((S - C[0]) ** 2).sum(1)
    for _ in range(1, K):
        C.append(S[np.searchsorted(np.cumsum(d), rng.random() * d.sum())])
        d = np.minimum(d, ((S - C[-1]) ** 2).sum(1))
    return np.array(C, dtype=np.float32)


def mse(X, C):
    D, _ = faiss.knn(X, C, 1)
    return float(D.mean())


out = {}
for name in sys.argv[1:] or ["starry.jpg", "dog.jpg"]:
    X = pixels(name)
    for K in (16, 56):
        for seed in (1, 2, 3):
            init = kpp(X, K, seed)
            row = {"n": len(X), "K": K, "seed": seed, "init_mse": mse(X, init)}
            C, lloyd = init.copy(), []
            for it in range(20):
                _, I = faiss.knn(X, C, 1)
                I = I[:, 0]
                cnt = np.bincount(I, minlength=K)
                sums = np.zeros((K, 3), np.float64)
                np.add.at(sums, I, X)
                keep = cnt > 0
                C[keep] = (sums[keep] / cnt[keep, None]).astype(np.float32)
                lloyd.append(mse(X, C))
            row["lloyd_mse_by_pass"] = lloyd
            em = {}
            for ep in (1, 2, 3, 5):
                Ce, _ = NumpyEmeans(3, K, nepochs=ep, seed=seed).train(X, init_centroids=init.copy())
                em[ep] = mse(X, np.ascontiguousarray(Ce, dtype=np.float32))
            row["emeans_mse_by_epochs"] = em
            # Lloyd passes needed to match each e-means budget (21 = not matched in 20)
            row["lloyd_passes_to_match"] = {ep: next((i + 1 for i, v in enumerate(lloyd) if v <= m), 21)
                                            for ep, m in em.items()}
            out[f"{name}|K{K}|s{seed}"] = row
            print(name, K, seed, "lloyd 2/5/10/20:", [round(lloyd[i], 2) for i in (1, 4, 9, 19)],
                  "emeans 1/2/3/5:", {k: round(v, 2) for k, v in em.items()},
                  "match:", row["lloyd_passes_to_match"], flush=True)
json.dump(out, open(HERE / "results_colour.json", "w"), indent=1)
