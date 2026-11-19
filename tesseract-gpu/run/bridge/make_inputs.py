"""Inputs for the GPU Bridge timing job: the English benchmark sample at height 36, the web
packs of eng best and eng fast, and the reference's greedy text for every line.

    python3 run/bridge/make_inputs.py SAMPLE_DIR OUT_DIR
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tessgpu import decode, ref  # noqa: E402
from tessgpu.traineddata import load  # noqa: E402

_M = {}


def text(args):
    variant, png = args
    if variant not in _M:
        m = load(ROOT / f"models/{variant}/eng.traineddata")
        _M[variant] = (m, decode.Codec(m))
    m, c = _M[variant]
    return decode.greedy(ref.forward(m, np.asarray(Image.open(png))), c)


def main():
    sample, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in open(sample / "sample.jsonl") if json.loads(l)["lang"] == "eng"]
    imgs = [np.asarray(Image.open(sample / r["png"])) for r in rows]
    assert all(g.shape[0] == 36 for g in imgs)
    np.concatenate([g.ravel() for g in imgs]).astype(np.uint8).tofile(out / "lines.bin")
    np.array([g.shape[1] for g in imgs], np.uint32).tofile(out / "widths.bin")
    with Pool(3) as p:
        exp = {v: p.map(text, [(v, str(sample / r["png"])) for r in rows]) for v in ("best", "fast")}
    (out / "expect.json").write_text(json.dumps({"ids": [r["id"] + "_" + r["cond"] for r in rows], **exp},
                                                ensure_ascii=False))
    print(len(rows), "lines,", sum(g.shape[1] for g in imgs), "px wide in total")


if __name__ == "__main__":
    main()
