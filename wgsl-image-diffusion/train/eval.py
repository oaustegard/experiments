"""FID of the trained model against its training set (AFHQv2 or CelebA) at 64x64, over guidance strengths and step counts.

    python eval.py --ema /content/wid/run/ema.pt --data /content/wid/afhq.npz --out /content/wid/eval

Samples are drawn class-balanced with the same Euler sampler the page uses. FID is clean-fid
("clean" mode: PIL bicubic resize to 299 before Inception), against every training image.
Writes eval.json (one record per config, appended as each finishes) and a sample grid per config.
Prints 'eval done' at the end.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import UNet, sample
from train import grid_png


def write_pngs(arr, d: Path):
    from PIL import Image
    d.mkdir(parents=True, exist_ok=True)
    for i, a in enumerate(arr):
        Image.fromarray(a).save(d / f"{i:05d}.png")


def stats(f):
    f = f.astype(np.float64)
    return f.mean(0), np.cov(f, rowvar=False)


def frechet(mu1, s1, mu2, s2):
    from scipy import linalg
    covmean = linalg.sqrtm(s1 @ s2)
    if not np.isfinite(covmean).all():
        off = np.eye(s1.shape[0]) * 1e-6
        covmean = linalg.sqrtm((s1 + off) @ (s2 + off))
    covmean = covmean.real
    d = mu1 - mu2
    return float(d @ d + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ema", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--bs", type=int, default=500)
    ap.add_argument("--configs", default="1:24,1.5:24,2:24,3:24,2:12,2:48", help="cfg:steps list")
    a = ap.parse_args()
    from cleanfid import fid
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sd = torch.load(a.ema, map_location=dev)
    nclass = sd["cls.weight"].shape[0] - 1
    m = UNet(nclass).to(dev).eval()
    m.load_state_dict(sd)
    real = out / "real"
    if not real.exists():
        write_pngs(np.load(a.data)["s64"], real)
    # clean-fid's Inception features; the Frechet distance is computed here because clean-fid 0.1.35
    # passes sqrtm(disp=False), which newer scipy rejects
    from cleanfid.features import build_feature_extractor
    fmodel = build_feature_extractor("clean", torch.device(dev))
    feats = lambda d: fid.get_folder_features(str(d), fmodel, num_workers=8 if dev == "cuda" else 0,  # noqa: E731
                                              batch_size=256, device=torch.device(dev), mode="clean", verbose=False)
    rs, rf = out / "real_stats.npz", out / "real_feats.npy"
    if not rs.exists() or not rf.exists():
        f_real = feats(real)
        np.save(rf, f_real.astype(np.float16))
        np.savez(rs, *stats(f_real))
    z = np.load(rs)
    real_stats = (z["arr_0"], z["arr_1"])
    floor = out / "floor.json"      # FID of n real training images against all of them: the best a model can score
    if not floor.exists():
        n_real = len(list(real.glob("*.png")))
        pick = np.random.default_rng(0).choice(n_real, min(a.n, n_real), replace=False)
        sub = out / "real_sub"
        sub.mkdir(exist_ok=True)
        for i in pick:
            (sub / f"{i:05d}.png").write_bytes((real / f"{i:05d}.png").read_bytes())
        floor.write_text(json.dumps({"n": len(pick), "fid": frechet(*stats(feats(sub)), *real_stats)}))
    print("floor", floor.read_text(), flush=True)
    done = set()
    res_path = out / "eval.json"
    if res_path.exists():
        done = {(r["cfg"], r["steps"]) for r in map(json.loads, res_path.read_text().splitlines())}
    configs = [(float(c), int(k)) for c, k in (x.split(":") for x in a.configs.split(","))]
    for cfg, steps in configs:
        if (cfg, steps) in done:
            continue
        t0 = time.time()
        ys = torch.arange(a.n) % nclass
        imgs = []
        for i in range(0, a.n, a.bs):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
                x = sample(m, ys[i:i + a.bs], 64, 64, steps=steps, cfg=cfg, seed=1000 + i, device=dev)
            imgs.append(((x.float() + 1) * 127.5).round().clamp(0, 255).byte().permute(0, 2, 3, 1).cpu().numpy())
        imgs = np.concatenate(imgs)
        gen = out / f"gen_cfg{cfg}_s{steps}"
        write_pngs(imgs, gen)
        t1 = time.time()
        score = frechet(*stats(feats(gen)), *real_stats)
        rec = {"cfg": cfg, "steps": steps, "n": a.n, "fid": float(score), "sample_s": t1 - t0, "fid_s": time.time() - t1}
        with open(res_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(rec, flush=True)
        sel = np.concatenate([imgs[k::nclass][:8] for k in range(nclass)])
        grid_png(torch.from_numpy(sel).permute(0, 3, 1, 2).float() / 127.5 - 1, out / f"grid_cfg{cfg}_s{steps}.png", nrow=8)
    copy_check(out, feats, rf, a.data, dev)
    print("eval done", flush=True)


def copy_check(out, feats, rf, data, dev, n=1000):
    n = min(n, len(np.load(rf, mmap_mode="r")) - 1)
    """Does the model copy training images? For n samples (guidance 2, 24 steps), the distance in Inception
    feature space to the nearest training image, against the same for n held-back training images to their
    nearest other training image. Copies would sit far below the real-to-real distances. Also saves a grid:
    24 samples, each above its nearest training image."""
    if (out / "copy_check.json").exists():
        return
    from PIL import Image
    R = torch.from_numpy(np.load(rf).astype(np.float32)).to(dev)
    gen_dir = out / "gen_cfg2.0_s24"
    G = torch.from_numpy(feats(gen_dir)[:n]).float().to(dev)
    n = min(n, len(G))

    def nn_dist(Q, skip_self=None):
        d = torch.cdist(Q, R)
        if skip_self is not None:
            d[torch.arange(len(Q)), skip_self] = float("inf")
        v, i = d.min(1)
        return v.cpu().numpy(), i.cpu().numpy()

    dg, ig = nn_dist(G)
    pick = torch.from_numpy(np.random.default_rng(1).choice(len(R), n, replace=False)).to(dev)
    dr, _ = nn_dist(R[pick], skip_self=pick)
    res = {"n": n, "gen_nn_median": float(np.median(dg)), "real_nn_median": float(np.median(dr)),
           "gen_nn_p01": float(np.quantile(dg, 0.01)), "real_nn_p01": float(np.quantile(dr, 0.01)),
           "gen_below_real_p01": float((dg < np.quantile(dr, 0.01)).mean())}
    (out / "copy_check.json").write_text(json.dumps(res, indent=1))
    print("copy check", res, flush=True)
    X = np.load(data)["s64"]
    order = np.argsort(dg)[:24]           # the 24 samples closest to a training image: the likeliest copies
    top = [np.asarray(Image.open(gen_dir / f"{i:05d}.png")) for i in order]
    bot = [X[ig[i]] for i in order]
    pairs = np.stack(top[:12] + bot[:12] + top[12:] + bot[12:])
    grid_png(torch.from_numpy(pairs).permute(0, 3, 1, 2).float() / 127.5 - 1, out / "copy_check.png", nrow=12)


if __name__ == "__main__":
    main()
