"""Stage 2: train the set denoiser on the fitted splat lists (rectified flow, CFG dropout, EMA).

    python train.py --out /content/sd/run --steps 40000

Reads <out>/sets.npz. Augments every example with a random torus translation and random x/y flips,
which the representation allows exactly. Writes stats.json (feature mean/std), ema.pt, log.jsonl,
samples_*.png and DONE; resumes from state.pt.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from model import NCLASS, SplatDiT, sample
from splats import from_features, render, sort_by_depth, to_features


def augment(mu, A, col, gen):
    B = mu.shape[0]
    dev = mu.device
    shift = torch.rand(B, 1, 2, generator=gen).to(dev)
    mu = (mu + shift) % 1.0
    fx = (torch.rand(B, 1, generator=gen) < 0.5).to(dev)
    fy = (torch.rand(B, 1, generator=gen) < 0.5).to(dev)
    mu = torch.stack([torch.where(fx, 1 - mu[..., 0], mu[..., 0]), torch.where(fy, 1 - mu[..., 1], mu[..., 1])], -1) % 1.0
    flip_b = fx ^ fy                       # one flip negates the off-diagonal term, two cancel
    A = torch.stack([A[..., 0], torch.where(flip_b, -A[..., 1], A[..., 1]), A[..., 2]], -1)
    return mu, A, col


def grid_png(feats, stats, path, size=128, tile=2):
    from PIL import Image
    mean, std = stats
    mu, A, col, o, s, _z = sort_by_depth(*from_features(feats * std + mean))
    img = render(mu.float(), A.float(), col.float(), o.float(), s.float(), size, size).clamp(-1, 1)
    img = ((img + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()
    img = np.tile(img, (1, tile, tile, 1))
    n, h, w, _ = img.shape
    cols = 6
    rows = math.ceil(n / cols)
    g = np.zeros((rows * (h + 2), cols * (w + 2), 3), np.uint8)
    for i in range(n):
        r, c = divmod(i, cols)
        g[r * (h + 2):r * (h + 2) + h, c * (w + 2):c * (w + 2) + w] = img[i]
    Image.fromarray(g).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--ema", type=float, default=0.9995)
    ap.add_argument("--ckpt", type=int, default=5000)
    a = ap.parse_args()
    out = Path(a.out)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
    z = np.load(out / "sets.npz")
    mu_all = torch.from_numpy(z["mu"].astype(np.float32)).to(dev)
    A_all = torch.from_numpy(z["A"].astype(np.float32)).to(dev)
    col_all = torch.from_numpy(z["col"].astype(np.float32)).to(dev)
    o_all = torch.from_numpy(z["o"].astype(np.float32)).to(dev)
    s_all = torch.from_numpy(z["s"].astype(np.float32)).to(dev)
    y_all = torch.from_numpy(z["y"].astype(np.int64)).to(dev)
    N, K = mu_all.shape[:2]
    z_all = ((torch.arange(K, device=dev, dtype=torch.float32) + 0.5) / K)[None].expand(N, K)   # back-to-front depth
    f = to_features(mu_all, A_all, col_all, o_all, s_all, z_all)
    mean = f.reshape(-1, f.shape[-1]).mean(0)
    std = f.reshape(-1, f.shape[-1]).std(0)
    mean[:4] = 0.0                          # keep the cos/sin pairs centred and isotropic,
    std[:4] = std[:4].mean()                # so a torus translation stays a rotation
    (out / "stats.json").write_text(json.dumps({"mean": mean.tolist(), "std": std.tolist(), "K": K}))
    print(f"{N} sets of {K} splats; fit psnr median {np.median(z['psnr']):.2f}", flush=True)

    torch.manual_seed(0)
    model = SplatDiT().to(dev)
    ema = SplatDiT().to(dev)
    ema.load_state_dict(model.state_dict())
    ema.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0, betas=(0.9, 0.99))
    step = 0
    st = out / "state.pt"
    if st.exists():
        s = torch.load(st, map_location=dev)
        model.load_state_dict(s["model"]); ema.load_state_dict(s["ema"]); opt.load_state_dict(s["opt"]); step = s["step"]
    gen = torch.Generator().manual_seed(99 + step)
    warm = 1000

    def lr_at(i):
        if i < warm:
            return a.lr * (i + 1) / warm
        return a.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (i - warm) / max(1, a.steps - warm))))

    log = open(out / "log.jsonl", "a")  # noqa: SIM115 -- held open for the whole run
    t0, tl, run_loss, run_n = time.time(), time.time(), 0.0, 0
    while step < a.steps:
        idx = torch.randint(N, (a.bs,), generator=gen).to(dev)
        mu, A, col = augment(mu_all[idx], A_all[idx], col_all[idx], gen)
        x0 = (to_features(mu, A, col, o_all[idx], s_all[idx], z_all[idx]) - mean) / std
        yc = y_all[idx]
        yc = torch.where(torch.rand(a.bs, device=dev) < 0.1, torch.full_like(yc, NCLASS), yc)
        t = torch.sigmoid(torch.randn(a.bs, device=dev))
        eps = torch.randn_like(x0)
        xt = (1 - t)[:, None, None] * x0 + t[:, None, None] * eps
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
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
        if step % 50 == 0:
            run_loss += loss.item(); run_n += 1
        if step % 200 == 0:
            rec = {"step": step, "loss": run_loss / max(1, run_n), "gnorm": float(gn), "lr": lr_at(step),
                   "s_per_100": (time.time() - tl) / 2, "t": time.time() - t0}
            log.write(json.dumps(rec) + "\n"); log.flush()
            print(rec, flush=True)
            run_loss, run_n, tl = 0.0, 0, time.time()
        if step % a.ckpt == 0 or step == a.steps:
            torch.save(ema.state_dict(), out / "ema.pt.tmp"); os.replace(out / "ema.pt.tmp", out / "ema.pt")
            torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(), "step": step},
                       out / "state.pt.tmp")
            os.replace(out / "state.pt.tmp", out / "state.pt")
            ema.eval()
            ys = torch.tensor([23, 42, 9, 41, 26, 18, 33, 7, 21, 3, 44, 6])
            xs = sample(ema, ys, K, steps=24, cfg=3.0, seed=0, device=dev)
            grid_png(xs, (mean, std), out / f"samples_{step:06d}.png")
            print(f"ckpt {step}", flush=True)
    (out / "DONE").write_text(str(step))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
