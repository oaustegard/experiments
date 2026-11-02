"""Build web/index.html: each Kodak image as a live, self-painting WGSL shader.

    python3 web/build.py
Writes web/index.html, web/shaders.js and web/img/<im>_{orig.jpg,avif.avif,webp.webp}.
The AVIF/WebP files are the downscaled-arm encodes closest to (not above) the full shader's bytes.
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import torch
from PIL import Image

WEB = Path(__file__).resolve().parent
HERE = WEB.parent
sys.path.insert(0, str(HERE))
import codec  # noqa: E402
from run_sweep import IMAGES, RUNS  # noqa: E402


def encode_under(img, fmt, budget, kw):
    """Best (largest) encode at or under budget bytes, across scales 1/2/4/8 and qualities."""
    best = None
    for s in [1, 2, 4, 8]:
        small = img if s == 1 else img.resize((img.width // s, img.height // s), Image.LANCZOS)
        for q in range(0, 101, 2):
            b = io.BytesIO()
            small.save(b, fmt, quality=q, **kw)
            n = b.tell()
            if n <= budget:
                dec = Image.open(io.BytesIO(b.getvalue())).convert("RGB").resize(img.size, Image.BICUBIC)
                p = codec.psnr8(__import__("numpy").asarray(dec), __import__("numpy").asarray(img))
                if best is None or p > best[0]:
                    best = (p, b.getvalue(), s, q, n)
    return best


def main():
    summary = json.loads((HERE / "results" / "summary.json").read_text())
    (WEB / "img").mkdir(exist_ok=True)
    entries = []
    for im in IMAGES:
        S = summary["images"].get(im)
        if not S:
            continue
        run, _ = codec.load_run(RUNS / f"{im}_prefix1024.json")
        wgsl = codec.rate(codec.emit_wgsl(run))["wgsl"]
        pts = S["prefix1024"]["points"]
        full = pts[-1]
        img = Image.open(HERE / "data" / "kodak" / f"{im}.png").convert("RGB")
        img.save(WEB / "img" / f"{im}_orig.jpg", quality=88)
        cmp = {}
        for fmt, ext, kw in [("AVIF", "avif", {"speed": 4}), ("WEBP", "webp", {"method": 6})]:
            b = encode_under(img, fmt, full["gz_packed"], kw)
            if b:
                (WEB / "img" / f"{im}_{ext}.{ext}").write_bytes(b[1])
                cmp[ext] = {"psnr": round(b[0], 2), "scale": b[2], "q": b[3], "bytes": b[4]}
        entries.append({"id": im, "wgsl": wgsl, "n": int(run["P"].shape[0]), "W": run["W"], "H": run["H"],
                        # the splat data itself, for the instanced renderer (instanced.js)
                        "P": run["P"].to(torch.int64).flatten().tolist(), "bg": run["bg"].to(torch.int64).tolist(), "A": run["A"],
                        "points": [{"n": r["n"], "bytes": r["gz_packed"], "psnr": round(r["psnr"], 2)} for r in pts],
                        "cmp": cmp})
    (WEB / "shaders.js").write_text("export const IMAGES = " + json.dumps(entries) + ";\n")
    print("wrote", len(entries), "images;", (WEB / "shaders.js").stat().st_size, "bytes of shaders.js")


if __name__ == "__main__":
    main()
