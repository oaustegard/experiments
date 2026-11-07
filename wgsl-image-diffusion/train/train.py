"""Train the image U-Net with rectified flow at 64x64 on AFHQv2 (cat, dog, wild; 15,803 images) or
CelebA (162,770 aligned faces; 4 classes from the Male and Smiling attributes).

    python train.py --out /content/wid/run --steps 60000                     # on a Colab GPU
    python train.py --out /content/wif/run --steps 60000 --dataset celeba
    python train.py --out /tmp/smoke --steps 4 --smoke           # CPU smoke test (40 images)

Writes log.jsonl, ema.pt (EMA weights, refreshed every --ckpt steps), samples_*.png, state.pt and
a DONE file. Resumes from <out>/state.pt if present.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import sys
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import DATASETS, UNet, sample

URL = "https://huggingface.co/datasets/huggan/AFHQv2/resolve/main/data/train-{:05d}-of-00013.parquet"
SIZES = (64, 128)
CELEBA = "https://huggingface.co/datasets/flwrlabs/celeba/resolve/main/img_align%2Bidentity%2Battr/train-{:05d}-of-00019.parquet"


def _shrink(b):
    from PIL import Image
    im = Image.open(io.BytesIO(b)).convert("RGB")
    return [np.asarray(im.resize((s, s), Image.LANCZOS)) for s in SIZES]


def _shrink_celeba(b):
    """178x218 aligned CelebA: centre 140x140 crop around the face (the usual 64x64 CelebA crop), then 64x64."""
    from PIL import Image
    im = Image.open(io.BytesIO(b)).convert("RGB").crop((19, 40, 159, 180))
    return np.asarray(im.resize((64, 64), Image.LANCZOS))


def load_celeba(cache: Path, limit=None):
    """uint8 N x 64 x 64 x 3 and labels 2 * Male + Smiling (0 woman, 1 woman smiling, 2 man, 3 man smiling)."""
    if cache.exists():
        z = np.load(cache)
        return {64: z["s64"]}, z["y"]
    import pyarrow.parquet as pq
    xs, ys = [], []
    with ProcessPoolExecutor(os.cpu_count()) as ex:
        for k in range(19):
            raw = urllib.request.urlopen(CELEBA.format(k), timeout=900).read()
            tab = pq.read_table(io.BytesIO(raw), columns=["image", "Male", "Smiling"]).to_pydict()
            imgs = [d["bytes"] for d in tab["image"]]
            lab = [2 * int(m) + int(sm) for m, sm in zip(tab["Male"], tab["Smiling"])]
            if limit:
                imgs, lab = imgs[:limit], lab[:limit]
            xs += list(ex.map(_shrink_celeba, imgs, chunksize=64))
            ys += lab
            print(f"shard {k}: {len(ys)} images", flush=True)
            if limit:
                break
    x = np.stack(xs)
    y = np.array(ys, np.int64)
    np.savez(cache, y=y, s64=x)
    return {64: x}, y


def load_afhq(cache: Path, limit=None):
    """uint8 arrays N x S x S x 3 for each size in SIZES, and labels. Images are 512x512 already."""
    if cache.exists():
        z = np.load(cache)
        return {s: z[f"s{s}"] for s in SIZES}, z["y"]
    import pyarrow.parquet as pq
    xs = {s: [] for s in SIZES}
    ys = []
    with ProcessPoolExecutor(os.cpu_count()) as ex:
        for k in range(13):
            raw = urllib.request.urlopen(URL.format(k), timeout=600).read()
            tab = pq.read_table(io.BytesIO(raw)).to_pydict()
            imgs, labs = [d["bytes"] for d in tab["image"]], tab["label"]
            if limit:
                imgs, labs = imgs[:limit], labs[:limit]
            for out in ex.map(_shrink, imgs, chunksize=32):
                for s, a in zip(SIZES, out):
                    xs[s].append(a)
            ys += labs
            print(f"shard {k}: {len(ys)} images", flush=True)
            if limit:
                break
    arrs = {s: np.stack(xs[s]) for s in SIZES}
    y = np.array(ys, np.int64)
    np.savez(cache, y=y, **{f"s{s}": a for s, a in arrs.items()})
    return arrs, y


def batch(X, y, B, dev, gen):
    """Random images with random horizontal flips; X is N x S x S x 3 uint8 on dev."""
    idx = torch.randint(X.shape[0], (B,), generator=gen).to(dev)
    x = X[idx].permute(0, 3, 1, 2).float() / 127.5 - 1
    f = (torch.rand(B, generator=gen) < 0.5).to(dev)
    x = torch.where(f[:, None, None, None], x.flip(3), x)
    return x.contiguous(memory_format=torch.channels_last), y[idx]


def grid_png(x, path, nrow):
    from PIL import Image
    x = ((x.clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()
    n, h, w, _ = x.shape
    rows = math.ceil(n / nrow)
    g = np.zeros((rows * (h + 2), nrow * (w + 2), 3), np.uint8)
    for i in range(n):
        r, c = divmod(i, nrow)
        g[r * (h + 2):r * (h + 2) + h, c * (w + 2):c * (w + 2) + w] = x[i]
    Image.fromarray(g).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--ema", type=float, default=0.9998)
    ap.add_argument("--ckpt", type=int, default=5000)
    ap.add_argument("--data", default=None, help="npz cache path (default <out>/afhq.npz)")
    ap.add_argument("--dataset", default="afhq", choices=sorted(DATASETS))
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    t0 = time.time()
    cache = Path(a.data) if a.data else out / f"{a.dataset}{'_smoke' if a.smoke else ''}.npz"
    load = load_celeba if a.dataset == "celeba" else load_afhq
    arrs, y = load(cache, limit=40 if a.smoke else None)
    nclass = len(DATASETS[a.dataset])
    (out / "names.json").write_text(json.dumps(DATASETS[a.dataset]))
    X = torch.from_numpy(arrs[64]).to(dev)
    y = torch.from_numpy(y).to(dev)
    print(f"data {len(y)} images {tuple(X.shape)}, classes {torch.bincount(y).tolist()}, {time.time()-t0:.0f}s",
          flush=True)

    torch.manual_seed(0)
    model = UNet(nclass).to(dev).to(memory_format=torch.channels_last)
    ema = UNet(nclass).to(dev)
    ema.load_state_dict(model.state_dict())
    ema.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0, betas=(0.9, 0.99))
    step = 0
    st = out / "state.pt"
    if st.exists():
        s = torch.load(st, map_location=dev)
        model.load_state_dict(s["model"]); ema.load_state_dict(s["ema"]); opt.load_state_dict(s["opt"])
        step = s["step"]
        print(f"resumed at {step}", flush=True)
    gen = torch.Generator().manual_seed(1234 + step)
    warm = 1000 if not a.smoke else 1

    def lr_at(i):
        if i < warm:
            return a.lr * (i + 1) / warm
        p = (i - warm) / max(1, a.steps - warm)
        return a.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))

    log = open(out / "log.jsonl", "a")  # noqa: SIM115 -- held open for the whole run
    use_amp = dev == "cuda"
    run_loss, run_n, tl = 0.0, 0, time.time()
    every = 200 if not a.smoke else 1
    while step < a.steps:
        x0, yc = batch(X, y, a.bs if not a.smoke else 4, dev, gen)
        B = x0.shape[0]
        drop = torch.rand(B, device=dev) < 0.1
        yc = torch.where(drop, torch.full_like(yc, nclass), yc)
        t = torch.sigmoid(torch.randn(B, device=dev))          # logit-normal(0, 1)
        eps = torch.randn_like(x0)
        xt = (1 - t)[:, None, None, None] * x0 + t[:, None, None, None] * eps
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            v = model(xt, t, yc)
        loss = F.mse_loss(v.float(), eps - x0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        d = min(a.ema, (1 + step) / (10 + step))
        with torch.no_grad():
            torch._foreach_lerp_(list(ema.parameters()), list(model.parameters()), 1 - d)
        step += 1
        if step % 50 == 0 or a.smoke:
            run_loss += loss.item(); run_n += 1
        if step % every == 0:
            rec = {"step": step, "loss": run_loss / max(1, run_n), "gnorm": float(gn), "lr": lr_at(step),
                   "s_per_100": (time.time() - tl) / every * 100, "t": time.time() - t0}
            log.write(json.dumps(rec) + "\n"); log.flush()
            print(rec, flush=True)
            run_loss, run_n, tl = 0.0, 0, time.time()
        if step % a.ckpt == 0 or step == a.steps:
            torch.save(ema.state_dict(), out / "ema.pt.tmp"); os.replace(out / "ema.pt.tmp", out / "ema.pt")
            torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                        "step": step}, out / "state.pt.tmp")
            os.replace(out / "state.pt.tmp", out / "state.pt")
            ema.eval()
            ycls = torch.arange(nclass).repeat_interleave(8)
            xs = sample(ema, ycls, steps=20 if not a.smoke else 2, cfg=2.0, seed=0, device=dev)
            grid_png(xs, out / f"samples_{step:06d}.png", nrow=8)
            print(f"ckpt {step}", flush=True)
            tl = time.time()
    (out / "DONE").write_text(str(step))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
