"""Parity of the engines on a community fine-tune (tessdata_contrib grc_hist) against Tesseract 5.3.4's own output.

    python3 run/check_contrib.py MODEL BENCH_DIR DUMP_DIR     # harness/dump_contrib.sh makes DUMP_DIR
Max |softmax difference| of the exact engine against Tesseract, argmax agreement, and whether the beam-search port
on the engine's softmax returns Tesseract's ids.
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tessgpu import beam, ref, tessdump  # noqa: E402
from tessgpu.traineddata import load  # noqa: E402


def main(model, bench, dumps):
    """The exact engine (checked against tessgpu.ref on the stock models) straight against Tesseract's dumps."""
    import torch
    from server.torch_engine import TorchTess
    torch.set_num_threads(2)
    m = load(model)
    eng = TorchTess(model, mode="exact")
    pairs = [(png, Path(dumps) / (png.name + ".bin")) for png in sorted(Path(bench).glob("*.png"))]
    pairs = [(p, d) for p, d in pairs if d.exists()]
    imgs = [np.asarray(Image.open(p)) for p, _ in pairs]
    probs = eng.recognize(imgs)
    worst, all_arg, same_ids = 0.0, 0, 0
    for (png, dump), p in zip(pairs, probs):
        d = tessdump.read(str(dump))
        tp = d["probs"]
        worst = max(worst, float(np.abs(p - tp).max()))
        all_arg += bool((p.argmax(1) == tp.argmax(1)).all())
        same_ids += list(beam.decode(p.astype(np.float32), m)) == list(d["ids"])
    n = len(pairs)
    print(f"{Path(model).name}: {n} lines; exact engine vs Tesseract max |dp| {worst:.2e}; argmax identical at every "
          f"step on {all_arg}/{n} lines; beam port on the engine's softmax = Tesseract's ids on {same_ids}/{n}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
