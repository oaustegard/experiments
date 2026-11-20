"""Fixtures for web/test/check_lineprep.mjs: 48 lines (eng clean/shade/invert/blur synthetic, 16 real scans) as
raw u8 plus the Python tight crop + flat-field (crop_check.py: flatfield(tight(g))) the page's lineprep.js ports.

    python3 run/check_lineprep.py && node web/test/check_lineprep.mjs
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "run"))
sys.path.insert(0, str(ROOT))
from flatfield_check import flatfield  # noqa: E402

OUT = ROOT / "data/fixtures/lineprep"


def tight(g, pad=4):
    ink = flatfield(g) < 128
    ys, xs = np.where(ink)
    if len(ys) == 0:
        return g
    return g[max(0, ys.min() - pad):min(g.shape[0], ys.max() + pad + 1),
             max(0, xs.min() - pad):min(g.shape[1], xs.max() + pad + 1)]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    picks = []
    for base, conds, n in ((ROOT / "data/lines", ("clean", "shade", "invert", "blur"), 8), (ROOT / "data/real", ("real",), 16)):
        seen = {}
        for l in open(base / "labels.jsonl", encoding="utf-8"):
            r = json.loads(l)
            if r["lang"] == "eng" and r["cond"] in conds and seen.get(r["cond"], 0) < n:
                seen[r["cond"]] = seen.get(r["cond"], 0) + 1
                p = base / r["file"] if (base / r["file"]).exists() else ROOT / r["file"]
                picks.append((r["id"] + "_" + r["cond"], p))
    meta = []
    for name, p in picks:
        g = np.asarray(Image.open(p).convert("L"))
        t = tight(g)
        out = flatfield(t)
        (OUT / f"{name}.in.u8").write_bytes(g.tobytes())
        (OUT / f"{name}.out.u8").write_bytes(out.tobytes())
        meta.append(dict(name=name, w=g.shape[1], h=g.shape[0], ow=out.shape[1], oh=out.shape[0]))
    (OUT / "index.json").write_text(json.dumps(meta, indent=0))
    print(len(meta), "fixtures ->", OUT)


if __name__ == "__main__":
    main()
