"""The image-to-svg route: paint regions traced by the image-to-svg skill, compiled by wgsl-forge.

    python3 svg_arm.py   -> results/svg_arm.json
For each image and colour count K: SVG gzip bytes, forge shader gzip bytes, and PSNR of the
shader rendered in headless Chromium (premultiplied, composited over white like forge's tests).
"""
from __future__ import annotations

import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "wgsl-forge"))
sys.path.insert(0, "/mnt/skills/user/image-to-svg/scripts")
sys.path.insert(0, str(HERE))
from codec import minify, psnr8  # noqa: E402
from forge.gpu import image, over_white, run_jobs  # noqa: E402
from forge.svg import read_svg  # noqa: E402
from pipeline import image_to_svg  # noqa: E402



def inline_classes(svg: str) -> str:
    """image-to-svg styles paths with id-scoped CSS classes (`#id .c0{fill:..;stroke:..}`), which
    forge's SVG reader does not resolve (every path renders black). Inline them as attributes."""
    import re
    rules = dict(re.findall(r"\.(c\d+)\{([^}]*)\}", svg))
    def attrs(decl):
        return " ".join(f'{k.strip()}="{v.strip()}"' for k, v in (d.split(":") for d in decl.split(";") if d))
    svg = re.sub(r'class="(c\d+)"', lambda m: attrs(rules[m.group(1)]), svg)
    return re.sub(r"<style>.*?</style>", "", svg, flags=re.S)


IMAGES = sys.argv[1:] or ["kodim23", "kodim05"]
KS = [4, 8, 16]


def main():
    out_path = HERE / "results" / "svg_arm.json"
    res = json.loads(out_path.read_text()) if out_path.exists() else {}
    for im in IMAGES:
        src = HERE / "data" / "kodak" / f"{im}.png"
        ref = np.asarray(Image.open(src).convert("RGB"))
        H, W = ref.shape[:2]
        for K in KS:
            key = f"{im}_K{K}"
            if key in res:
                continue
            t0 = time.time()
            svg, _ = image_to_svg(str(src), mode="painting", svg_width=W, K=K)
            svg = inline_classes(svg)
            scene = read_svg(svg)
            wgsl = minify(scene.compile())
            r = run_jobs([{"id": "s", "kind": "render", "wgsl": wgsl, "width": W, "height": H},
                          {"id": "c", "kind": "svg", "svg": svg, "width": W, "height": H}])
            row = {"K": K, "svg_gz": len(gzip.compress(svg.encode(), 9, mtime=0)),
                   "wgsl_gz": len(gzip.compress(wgsl.encode(), 9, mtime=0)), "wgsl_min": len(wgsl),
                   "seconds": time.time() - t0}
            if r["s"].get("ok"):
                got = over_white(image(r["s"], W, H), premultiplied=True)
                row["psnr_shader"] = psnr8(np.rint(got).astype(np.uint8), ref)
                row["draw_ms"] = r["s"].get("draw_ms")
            else:
                row["error"] = r["s"].get("error")
            if r["c"].get("ok"):
                chrome = over_white(image(r["c"], W, H), premultiplied=False)
                row["psnr_svg_chromium"] = psnr8(np.rint(chrome).astype(np.uint8), ref)
            res[key] = row
            print(key, row, flush=True)
            out_path.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
