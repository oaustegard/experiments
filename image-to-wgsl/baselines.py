"""Rate-distortion sweeps for JPEG, WebP and AVIF (Pillow) on the same images.

None of them reach the shader's lowest byte counts at full resolution, so each also gets a
`_down` arm: encode a 2x/4x/8x-downscaled image, decode by bicubic upscaling.
    python3 baselines.py  -> results/baselines.json
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from codec import psnr8  # noqa: E402
from run_sweep import IMAGES  # noqa: E402


def enc(img, fmt, **kw):
    b = io.BytesIO()
    img.save(b, fmt, **kw)
    data = b.getvalue()
    return len(data), Image.open(io.BytesIO(data)).convert("RGB")


def sweep(img):
    ref = np.asarray(img)
    out = {"jpeg": [], "jpeg_down": [], "webp": [], "webp_down": [], "avif": [], "avif_down": []}
    for q in [1, 2, 3, 5, 8, 10, 15, 20, 30, 40, 50, 60, 75, 90]:
        n, d = enc(img, "JPEG", quality=q, optimize=True)
        out["jpeg"].append({"q": q, "bytes": n, "psnr": psnr8(np.asarray(d), ref)})
    for s in [2, 4, 8]:
        small = img.resize((img.width // s, img.height // s), Image.LANCZOS)
        for arm, fmt, kw, qs in [("jpeg_down", "JPEG", {"optimize": True}, [5, 10, 20, 30, 50, 75, 90]),
                                 ("webp_down", "WEBP", {"method": 6}, [0, 5, 10, 20, 30, 50, 75, 90]),
                                 ("avif_down", "AVIF", {"speed": 4}, [0, 5, 10, 20, 30, 50, 75])]:
            for q in qs:
                n, d = enc(small, fmt, quality=q, **kw)
                up = d.resize(img.size, Image.BICUBIC)
                out[arm].append({"q": q, "scale": s, "bytes": n, "psnr": psnr8(np.asarray(up), ref)})
    for q in [0, 1, 2, 5, 10, 15, 20, 30, 40, 50, 60, 75, 90]:
        n, d = enc(img, "WEBP", quality=q, method=6)
        out["webp"].append({"q": q, "bytes": n, "psnr": psnr8(np.asarray(d), ref)})
    for q in [0, 1, 2, 3, 5, 8, 10, 15, 20, 25, 30, 40, 50, 60, 75]:
        n, d = enc(img, "AVIF", quality=q, speed=4)
        out["avif"].append({"q": q, "bytes": n, "psnr": psnr8(np.asarray(d), ref)})
    return out


def main():
    res = {}
    for im in IMAGES:
        img = Image.open(HERE / "data" / "kodak" / f"{im}.png").convert("RGB")
        res[im] = sweep(img)
        a = res[im]
        print(im, "min bytes:", {k: min(r["bytes"] for r in v) for k, v in a.items()}, flush=True)
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "baselines.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
