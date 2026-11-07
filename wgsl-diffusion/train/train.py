"""Train the texture U-Net with rectified flow on DTD (all 47 classes, 5,640 images).

    python train.py --out /content/run --steps 60000          # on a Colab GPU
    python train.py --out /tmp/smoke --steps 20 --smoke        # CPU smoke test

Writes log.jsonl, ema.pt (EMA weights, refreshed every --ckpt steps), samples_*.png
and a DONE file. Resumes from <out>/state.pt if present.
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
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import NCLASS, UNet, sample

URL = "https://huggingface.co/datasets/tanganke/dtd/resolve/main/data/{}-00000-of-00001.parquet"
SCALES = (96, 128, 176)     # each image resized (short side) and centre-cropped to S x S
CROP = 64


def load_dtd(cache: Path, limit=None):
    if cache.exists():
        z = np.load(cache)
        return [z[f"s{s}"] for s in SCALES], z["y"]
    import pyarrow.parquet as pq
    from PIL import Image
    xs = {s: [] for s in SCALES}
    ys = []
    for split in ("train", "test"):
        raw = urllib.request.urlopen(URL.format(split), timeout=300).read()
        tab = pq.read_table(io.BytesIO(raw)).to_pydict()
        for img, lab in zip(tab["image"], tab["label"]):
            im = Image.open(io.BytesIO(img["bytes"])).convert("RGB")
            w, h = im.size
            for s in SCALES:
                k = s / min(w, h)
                r = im.resize((max(s, round(w * k)), max(s, round(h * k))), Image.LANCZOS)
                l, t = (r.width - s) // 2, (r.height - s) // 2
                xs[s].append(np.asarray(r.crop((l, t, l + s, t + s))))
            ys.append(lab)
            if limit and len(ys) >= limit:
                break
        if limit and len(ys) >= limit:
            break
    arrs = [np.stack(xs[s]) for s in SCALES]
    y = np.array(ys, np.int64)
    np.savez(cache, y=y, **{f"s{s}": a for s, a in zip(SCALES, arrs)})
    return arrs, y


def batch(arrs, y, B, dev, gen):
    """Random 64x64 crops from one random scale, with flips and transposes."""
    si = int(torch.randint(len(arrs), (1,), generator=gen))
    X = arrs[si]
    N, S = X.shape[0], X.shape[1]
    idx = torch.randint(N, (B,), generator=gen).to(dev)
    oy = torch.randint(S - CROP + 1, (B,), generator=gen).to(dev)
    ox = torch.randint(S - CROP + 1, (B,), generator=gen).to(dev)
    ar = torch.arange(CROP, device=dev)
    rows = (oy[:, None] + ar[None])[:, :, None]
    cols = (ox[:, None] + ar[None])[:, None, :]
    x = X[idx[:, None, None], rows, cols]                     # B, 64, 64, 3 uint8
    x = x.permute(0, 3, 1, 2).float() / 127.5 - 1
    f = torch.rand(B, 3, generator=gen).to(dev) < 0.5
    x = torch.where(f[:, 0, None, None, None], x.flip(3), x)
    x = torch.where(f[:, 1, None, None, None], x.flip(2), x)
    x = torch.where(f[:, 2, None, None, None], x.transpose(2, 3), x)
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
    ap.add_argument("--ema", type=float, default=0.9995)
    ap.add_argument("--ckpt", type=int, default=5000)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    t0 = time.time()
    arrs, y = load_dtd(out / "dtd.npz" if not a.smoke else out / "dtd_smoke.npz", limit=40 if a.smoke else None)
    arrs = [torch.from_numpy(x).to(dev) for x in arrs]
    y = torch.from_numpy(y).to(dev)
    print(f"data {len(y)} images, scales {[tuple(x.shape) for x in arrs]}, {time.time()-t0:.0f}s", flush=True)

    torch.manual_seed(0)
    model = UNet().to(dev).to(memory_format=torch.channels_last)
    ema = UNet().to(dev)
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
    warm = 1000

    def lr_at(i):
        if i < warm:
            return a.lr * (i + 1) / warm
        p = (i - warm) / max(1, a.steps - warm)
        return a.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))

    log = open(out / "log.jsonl", "a")  # noqa: SIM115 -- held open for the whole run
    use_amp = dev == "cuda"
    run_loss, run_n, tl = 0.0, 0, time.time()
    while step < a.steps:
        x0, yc = batch(arrs, y, a.bs, dev, gen)
        B = x0.shape[0]
        drop = torch.rand(B, device=dev) < 0.1
        yc = torch.where(drop, torch.full_like(yc, NCLASS), yc)
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
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm, 1 - d)
        step += 1
        run_loss += loss.item() if step % 50 == 0 or a.smoke else 0.0
        run_n += 1 if step % 50 == 0 or a.smoke else 0
        if step % 200 == 0 or a.smoke:
            rec = {"step": step, "loss": run_loss / max(1, run_n), "gnorm": float(gn), "lr": lr_at(step),
                   "s_per_100": (time.time() - tl) / (200 if not a.smoke else 1) * 100, "t": time.time() - t0}
            log.write(json.dumps(rec) + "\n"); log.flush()
            print(rec, flush=True)
            run_loss, run_n, tl = 0.0, 0, time.time()
        if step % a.ckpt == 0 or step == a.steps:
            torch.save(ema.state_dict(), out / "ema.pt.tmp"); os.replace(out / "ema.pt.tmp", out / "ema.pt")
            torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                        "step": step}, out / "state.pt.tmp")
            os.replace(out / "state.pt.tmp", out / "state.pt")
            ema.eval()
            ycls = torch.arange(NCLASS if not a.smoke else 4)
            xs = sample(ema, ycls, 64, 64, steps=24 if not a.smoke else 2, cfg=3.0, seed=0, device=dev)
            grid_png(xs, out / f"samples_{step:06d}.png", nrow=12)
            xs = sample(ema, ycls[:12], 128, 128, steps=24 if not a.smoke else 2, cfg=3.0, seed=1,
                        pad="circular", device=dev)
            grid_png(xs, out / f"samples128_{step:06d}.png", nrow=6)
            print(f"ckpt {step}", flush=True)
    (out / "DONE").write_text(str(step))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
