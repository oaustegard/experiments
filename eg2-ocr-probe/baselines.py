"""Free OCR engines on the exact evaluation lines the EG2 probe was scored on.

Regenerates probe.py's two evaluation sets (2,000 lines in the training fonts, 1,000 in the
held-out font; same seeds), checks the first lines match the texts stored in
results/result_pretrained.json, and scores:

  tesseract   Tesseract 5 LSTM, English, --psm 7 (single line)
  ppocr_v3    PaddleOCR PP-OCRv3 recognizer via rapidocr_onnxruntime (recognition only)

    PROBE_CACHE=<dir with fonts/ and test.parquet> python3 baselines.py
"""
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
import probe


def tess_one(a):
    from PIL import Image
    with tempfile.NamedTemporaryFile(suffix=".png") as f:
        Image.fromarray(a).save(f.name)
        out = subprocess.run(["tesseract", f.name, "-", "--psm", "7", "-l", "eng"],
                             capture_output=True, text=True, check=False, env={**os.environ, "OMP_THREAD_LIMIT": "1"})
    return " ".join(out.stdout.split())


_ocr = None


def ppocr_one(a):
    global _ocr
    import numpy as np
    if _ocr is None:
        from rapidocr_onnxruntime import RapidOCR
        _ocr = RapidOCR()
    res, _ = _ocr(np.stack([a] * 3, -1), use_det=False, use_cls=False, use_rec=True)
    return " ".join((res[0][1] if res else "").split())


def score(name, fn, imgs, texts):
    t0 = time.time()
    with ProcessPoolExecutor(4) as ex:
        preds = list(ex.map(fn, list(imgs), chunksize=16))
    refs = [" ".join(t.split()) for t in texts]
    return {"cer": probe.cer(preds, refs), "exact": sum(p == r for p, r in zip(preds, refs)) / len(refs),
            "seconds": round(time.time() - t0, 1), "samples": [[r, p] for r, p in zip(refs[:12], preds[:12])]}


def main():
    seen = probe.make_eval_set("test", False, 2000, seed=10)
    held = probe.make_eval_set("test", True, 1000, seed=11)
    stored = json.loads((HERE / "results/result_pretrained.json").read_text())
    ref12 = [r for r, _ in stored["seen_fonts"]["prepool"]["samples"]]
    same = sum(a == b for a, b in zip(ref12, seen[1][:12]))
    print(f"regenerated eval texts match the Kaggle run's: {same}/12", flush=True)
    out = {"eval_text_match_of_12": same}
    for name, fn in (("tesseract", tess_one), ("ppocr_v3", ppocr_one)):
        out[name] = {"seen_fonts": score(name, fn, *seen), "heldout_font": score(name, fn, *held)}
        print(name, {k: round(v["cer"], 4) for k, v in out[name].items()}, flush=True)
    (HERE / "results/baselines.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
