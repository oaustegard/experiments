"""Free handwriting baselines on the IAM test lines the EG2 handwriting probe is scored on.

  tesseract     Tesseract 5 LSTM, English, --psm 7, on the original 128 px line images
  trocr_small   microsoft/trocr-small-handwritten (62M, fine-tuned on IAM), greedy decoding, CPU

Scored like iam_probe.py: CER on raw IAM text and on text with its spaced punctuation closed up.

    PROBE_CACHE=<dir with iam_test.parquet> python3 iam_baselines.py [--limit N]
"""
import argparse
import io
import json
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from iam_probe import CACHE, cer, norm


def tess_one(png_bytes):
    with tempfile.NamedTemporaryFile(suffix=".png") as f:
        f.write(png_bytes)
        f.flush()
        out = subprocess.run(["tesseract", f.name, "-", "--psm", "7", "-l", "eng"], capture_output=True,
                             text=True, check=False, env={**os.environ, "OMP_THREAD_LIMIT": "1"})
    return " ".join(out.stdout.split())


def trocr(images, bs=16):
    import torch
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    torch.set_num_threads(4)
    proc = TrOCRProcessor.from_pretrained("microsoft/trocr-small-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-small-handwritten").eval()
    out = []
    for b0 in range(0, len(images), bs):
        pv = proc(images=images[b0:b0 + bs], return_tensors="pt").pixel_values
        with torch.no_grad():
            ids = model.generate(pv, max_new_tokens=128, num_beams=1)
        out += [" ".join(t.split()) for t in proc.batch_decode(ids, skip_special_tokens=True)]
        if b0 % (bs * 20) == 0:
            print(f"trocr {b0 + len(ids)}/{len(images)}", flush=True)
    return out


def scores(preds, refs, seconds):
    return {"cer_raw": cer(preds, refs), "cer_norm": cer([norm(p) for p in preds], [norm(r) for r in refs]),
            "exact_norm": sum(norm(p) == norm(r) for p, r in zip(preds, refs)) / len(refs),
            "seconds": round(seconds), "samples": [[r, p] for r, p in zip(refs[:15], preds[:15])]}


def main():
    import pandas as pd
    from PIL import Image
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    df = pd.read_parquet(CACHE / "iam_test.parquet")
    if a.limit:
        df = df.iloc[:a.limit]
    refs = list(df["text"])
    raw = [r["bytes"] for r in df["image"]]
    out = {"lines": len(refs)}
    t0 = time.time()
    with ProcessPoolExecutor(4) as ex:
        p = list(ex.map(tess_one, raw, chunksize=16))
    out["tesseract"] = scores(p, refs, time.time() - t0)
    print("tesseract", {k: v for k, v in out["tesseract"].items() if k != "samples"}, flush=True)
    t0 = time.time()
    p = trocr([Image.open(io.BytesIO(b)).convert("RGB") for b in raw])
    out["trocr_small"] = scores(p, refs, time.time() - t0)
    print("trocr_small", {k: v for k, v in out["trocr_small"].items() if k != "samples"}, flush=True)
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results/iam_baselines.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
