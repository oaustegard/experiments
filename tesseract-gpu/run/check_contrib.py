"""Parity of the engines on a community fine-tune (tessdata_contrib grc_hist) against Tesseract 5.3.4's own output.

    python3 run/check_contrib.py MODEL BENCH_DIR DUMP_DIR     # harness/dump_contrib.sh makes DUMP_DIR
Per line: max |softmax difference| reference vs Tesseract, argmax agreement, and whether the beam-search port
returns Tesseract's ids. Also the exact PyTorch engine against the reference.
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
    from server.torch_engine import TorchTess
    m = load(model)
    eng = TorchTess(model, mode="exact")
    pngs = sorted(Path(bench).glob("*.png"))
    worst, arg, same_ids, eng_worst, n = 0.0, [], 0, 0.0, 0
    for png in pngs:
        dump = Path(dumps) / (png.name + ".bin")
        if not dump.exists():
            continue
        g = np.asarray(Image.open(png))
        p = ref.forward(m, g)
        d = tessdump.read(str(dump))
        tp = d["probs"]
        worst = max(worst, float(np.abs(p - tp).max()))
        arg.append(float((p.argmax(1) == tp.argmax(1)).mean()))
        same_ids += list(beam.decode(tp.astype(np.float32), m)) == list(d["ids"])
        pe = eng.recognize([g])[0]
        eng_worst = max(eng_worst, float(np.abs(pe - p).max()))
        n += 1
    print(f"{Path(model).name}: {n} lines; reference vs Tesseract max |dp| {worst:.2e}, argmax identical at every step on "
          f"{sum(a == 1.0 for a in arg)}/{n} lines; beam port = Tesseract ids on {same_ids}/{n}; exact engine vs reference max |dp| {eng_worst:.2e}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
