"""Resize a benchmark sample to each model's input height for the jailed parity harness.

    python3 run/parity_prep.py OUT_DIR [step]
Writes OUT_DIR/<lang>/<id>_<cond>.png and OUT_DIR/sample.jsonl (every step-th synthetic row).
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tessgpu import bench, ref  # noqa: E402

HEIGHT = {"eng": 36, "fra": 36, "grc": 48}


def main():
    out = Path(sys.argv[1])
    step = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    rows = bench.rows("synth")[::step]
    with open(out / "sample.jsonl", "w") if out.mkdir(parents=True, exist_ok=True) is None else None as f:
        for r in rows:
            g = np.asarray(Image.open(r["path"]).convert("L"))
            g = ref.scale_to_height(g, HEIGHT[r["lang"]])
            d = out / r["lang"]
            d.mkdir(exist_ok=True)
            name = f"{r['id']}_{r['cond']}.png"
            Image.fromarray(g).save(d / name)
            f.write(json.dumps({**r, "png": f"{r['lang']}/{name}"}) + "\n")
    print(len(rows), "lines")


if __name__ == "__main__":
    main()
