"""Render emitted shaders in headless Chromium (SwiftShader WebGPU) and compare with the torch decode.

    python3 verify_gpu.py run.json image.png [--n K ...] [--encoding packed|columnar]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "wgsl-forge"))
import codec  # noqa: E402
from forge import gpu  # noqa: E402


def gpu_render(wgsl_list, W, H, time=0.0):
    jobs = [{"id": str(i), "kind": "render", "wgsl": w, "width": W, "height": H, "time": time}
            for i, w in enumerate(wgsl_list)]
    res = gpu.run_jobs(jobs)
    out = []
    for i in range(len(wgsl_list)):
        r = res[str(i)]
        if not r.get("ok"):
            out.append((None, r.get("error"), r))
        else:
            out.append((gpu.image(r, W, H)[..., :3].copy(), None, r))
    return out


def check(run, residual=None, ns=(None,), encodings=("packed",)):
    W, H = run["W"], run["H"]
    target8 = np.rint(run["target"].numpy() * 255).astype(np.uint8)
    srcs, meta = [], []
    for enc in encodings:
        for n in ns:
            s = codec.emit_wgsl(run, n=n, encoding=enc, residual=residual)
            r = codec.rate(s)
            srcs.append(r["wgsl"])
            meta.append({"encoding": enc, "n": n or int(run["P"].shape[0]), "gz": r["gz"], "min": r["min"]})
    imgs = gpu_render(srcs, W, H)
    rows = []
    for m, (img, err, raw) in zip(meta, imgs):
        ref = codec.render_image(run["P"], run["bg"], H, W, run["A"], run["layout"], n=m["n"], residual=residual).numpy()
        ref8 = np.rint(ref * 255).astype(np.uint8)
        if img is None:
            m.update(ok=False, error=err)
        else:
            diff = np.abs(img.astype(int) - ref8.astype(int))
            m.update(ok=True, psnr_gpu_vs_torch=codec.psnr8(img, ref8), max_abs=int(diff.max()),
                     frac_differ=float((diff > 0).mean()), psnr_gpu=codec.psnr8(img, target8),
                     psnr_torch=codec.psnr8(ref8, target8), draw_ms=raw.get("draw_ms"))
        rows.append(m)
    return rows, imgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("image")
    ap.add_argument("--n", type=int, nargs="*", default=[])
    ap.add_argument("--encoding", nargs="*", default=["packed"])
    ap.add_argument("--save", default=None)
    a = ap.parse_args()
    run, res = codec.load_run(a.run, a.image)
    rows, imgs = check(run, res, ns=a.n or [None], encodings=a.encoding)
    for r in rows:
        print(json.dumps(r))
    if a.save and imgs[-1][0] is not None:
        gpu.save_png(a.save, imgs[-1][0])


if __name__ == "__main__":
    main()
