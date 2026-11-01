"""Resolution independence: the same compiled shader against Chromium at several sizes.
Reports mean absolute error and the ink ratio (shader ink / reference ink) per size;
a shape-geometry error shows up as an ink ratio that stays off 1 as the size grows.

    python3 tests/scale_check.py [--json results/scale_check.json]
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from forge.gpu import image, over_white, run_jobs   # noqa: E402
from forge.svg import read_svg                       # noqa: E402

FILES = ["tests/svg/real/fe_settings.svg", "tests/svg/real/si_github.svg",
         "tests/svg/real/tw_1f981.svg", "tests/svg/paths.svg"]
SIZES = [48, 96, 256, 512, 768]


def main():
    jobs, meta = [], []
    for f in FILES:
        text = (HERE / f).read_text()
        sc = read_svg(text)
        src = sc.compile()
        for size in SIZES:
            w, h = size, round(size * sc.viewbox[3] / sc.viewbox[2])
            jobs.append({"id": f"{f}|{size}|r", "kind": "svg", "svg": text, "width": w, "height": h})
            jobs.append({"id": f"{f}|{size}|w", "kind": "render", "wgsl": src, "width": w, "height": h})
            meta.append((f, size, w, h))
    res = run_jobs(jobs)
    out = {}
    for f, size, w, h in meta:
        r = over_white(image(res[f"{f}|{size}|r"], w, h), False)
        g = over_white(image(res[f"{f}|{size}|w"], w, h), True)
        ink = float((255 - g).sum() / max((255 - r).sum(), 1e-9))
        mae = float(np.abs(r - g).mean())
        out.setdefault(Path(f).stem, {})[size] = {"mae": round(mae, 3), "ink_ratio": round(ink, 4)}
        print(f"{Path(f).stem:<14} {size:4d}px  mae {mae:5.2f}  ink ratio {ink:.4f}")
    if "--json" in sys.argv:
        p = Path(sys.argv[sys.argv.index("--json") + 1])
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
