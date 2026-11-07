"""Stage 1: fit a K-splat list (splats.py) to two 128x128 crops of every DTD image, on a GPU.

    python fit_dataset.py --out /content/sd/run --K 512

Each image is resized so its short side is 192 or 288 and one random 128x128 crop is taken at each
scale. Lighting is removed first: the resized image minus a Gaussian blur of it (sigma 24 px), plus its
mean, so a tile carries texture and no tile-wide light or shadow. Writes sets.npz (mu, A, col, o, s as
float16, y, psnr) and resumes from sets_partial.npz.
"""
from __future__ import annotations

import argparse
import io
import json
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
from splats import fit

URL = "https://huggingface.co/datasets/tanganke/dtd/resolve/main/data/{}-00000-of-00001.parquet"
SCALES = (192, 288)
CROP = 128


def highpass(im, sigma=24):
    """Remove low-frequency lighting: image - blur + mean, per channel."""
    from PIL import Image, ImageFilter
    a = np.asarray(im).astype(np.float32)
    b = np.asarray(im.filter(ImageFilter.GaussianBlur(sigma))).astype(np.float32)
    return Image.fromarray(np.clip(a - b + a.mean((0, 1)), 0, 255).astype(np.uint8))


def crops(seed=0, limit=None):
    import pyarrow.parquet as pq
    from PIL import Image
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for split in ("train", "test"):
        tab = pq.read_table(io.BytesIO(urllib.request.urlopen(URL.format(split), timeout=300).read())).to_pydict()
        for img, lab in zip(tab["image"], tab["label"]):
            im = Image.open(io.BytesIO(img["bytes"])).convert("RGB")
            for s in SCALES:
                k = s / min(im.size)
                r = im.resize((max(s, round(im.width * k)), max(s, round(im.height * k))), Image.LANCZOS)
                r = highpass(r)
                l = int(rng.integers(0, r.width - CROP + 1))
                t = int(rng.integers(0, r.height - CROP + 1))
                xs.append(np.asarray(r.crop((l, t, l + CROP, t + CROP))))
                ys.append(lab)
            if limit and len(ys) >= limit:
                return np.stack(xs), np.array(ys)
    return np.stack(xs), np.array(ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--K", type=int, default=512)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--lr", type=float, default=0.03)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if (out / "sets.npz").exists():
        print("sets.npz exists", flush=True)
        return
    t0 = time.time()
    cache = out / "crops.npz"
    if cache.exists():
        z = np.load(cache); X, Y = z["x"], z["y"]
    else:
        X, Y = crops(limit=a.limit)
        np.savez(cache, x=X, y=Y)
    N = len(Y)
    print(f"{N} crops in {time.time()-t0:.0f}s", flush=True)
    part = out / "sets_partial.npz"
    keys = ("mu", "A", "col", "o", "s", "psnr")
    res = {k: [] for k in keys}
    done = 0
    if part.exists():
        z = np.load(part)
        res = {k: [z[k]] for k in res}
        done = len(z["psnr"])
        print(f"resumed at {done}", flush=True)
    log = open(out / "fit_log.jsonl", "a")  # noqa: SIM115 -- held open for the whole run
    while done < N:
        tb = time.time()
        x = torch.from_numpy(X[done:done + a.bs]).to(dev).permute(0, 3, 1, 2).float() / 127.5 - 1
        mu, A, col, o, s_, _z, psnr = fit(x, a.K, steps=a.steps, lr=a.lr, seed=done)
        for k, v in zip(keys, (mu, A, col, o, s_, psnr)):
            res[k].append(v.cpu().numpy().astype(np.float16 if k != "psnr" else np.float32))
        done += x.shape[0]
        rec = {"done": done, "psnr_median": float(psnr.median()), "s_batch": time.time() - tb, "t": time.time() - t0}
        log.write(json.dumps(rec) + "\n"); log.flush()
        print(rec, flush=True)
        if (done // a.bs) % 10 == 0 or done >= N:
            np.savez(part, **{k: np.concatenate(v) for k, v in res.items()})
    np.savez(out / "sets.npz", y=Y, **{k: np.concatenate(v) for k, v in res.items()})
    print("FIT DONE", flush=True)


if __name__ == "__main__":
    main()
