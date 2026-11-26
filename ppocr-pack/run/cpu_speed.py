#!/usr/bin/env python3
"""How much of OCRia-nano-1's CPU lead over PP-OCRv6 small's recognizer can be had without retraining?

Arms, all onnxruntime CPU on one thread, one line per call, on the 400 FUNSD lines of tesseract-gpu/data/real:
  pp_small        PP-OCRv6 small recognizer as published (CTC head over 18,710 characters)
  pp_small_latin  the same with the head's MatMul/Add cut to blank + the 791 characters of OCRia's Latin
                  recognizer (the softmax then runs over those only)
  pp_small_latin_i8   pp_small_latin with onnxruntime dynamic int8 quantization of its MatMuls
  pp_tiny         PP-OCRv6 tiny recognizer as published, for scale
  ocria_latin     OCRia-nano-1's rec_latin (gray, height 32, pixel/127.5-1), for scale
Per arm: total ms over the lines (after 10 warm-up lines), ms per line, mean CER (tess_baseline.cer).

    python3 run/cpu_speed.py      # writes results/cpu_speed.json
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import yaml
from onnx import numpy_helper
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
from _lib.paths import experiment  # noqa: E402

TESS = experiment("tesseract-gpu")
OCRIA = experiment("ocria-pack") / "models"
sys.path.insert(0, str(TESS / "run"))
from tess_baseline import cer  # noqa: E402

WORK = ROOT / "data" / "cpu_speed"


def sess(path):
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def pp_chars(size):
    cfg = yaml.safe_load(open(ROOT / f"models/{size}_rec/inference.yml"))
    return ["<blank>"] + list(cfg["PostProcess"]["character_dict"]) + [" "]


def latin_head(chars):
    """PP small with its CTC head cut to blank + OCRia's Latin characters. Returns (path, kept chars)."""
    keep_set = set(json.load(open(OCRIA / "rec_latin.charset.json")))
    keep = [0] + [i for i, c in enumerate(chars) if i and c in keep_set]
    m = onnx.load(str(ROOT / "models/small_rec/inference.onnx"))
    init = {t.name: t for t in m.graph.initializer}
    for name, axis in (("linear_8.w_0", 1), ("linear_8.b_0", 0)):
        a = numpy_helper.to_array(init[name])
        init[name].CopyFrom(numpy_helper.from_array(np.take(a, keep, axis=axis).astype(np.float32), name))
    out = m.graph.output[0].type.tensor_type.shape.dim[2]
    out.dim_value = len(keep)
    WORK.mkdir(parents=True, exist_ok=True)
    p = WORK / "pp_small_latin.onnx"
    onnx.save(m, str(p))
    return p, [chars[i] for i in keep]


def pp_prep(path):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    rw = max(1, min(3200, math.ceil(48 * w / h)))
    a = np.asarray(im.resize((rw, 48), Image.BILINEAR), dtype=np.float32)[:, :, ::-1]
    x = np.zeros((1, 3, 48, max(320, rw)), np.float32)
    x[0, :, :, :rw] = ((a / 255.0 - 0.5) / 0.5).transpose(2, 0, 1)
    return x


def ocria_prep(path):
    im = Image.open(path).convert("L")
    w, h = im.size
    rw = min(1280, max(8, round(w * 32 / h)))
    x = np.ones((1, 1, 32, max(128, -(-rw // 128) * 128)), np.float32)
    x[0, 0, :, :rw] = np.asarray(im.resize((rw, 32), Image.BILINEAR), dtype=np.float32) / 127.5 - 1.0
    return x, rw


def ctc(ids, chars, offset=0):
    out, prev = [], -1
    for i in ids:
        if i != prev and i != 0:
            j = i - offset
            if 0 <= j < len(chars):
                out.append(chars[j])
        prev = i
    return "".join(out)


def run_arm(name, s, rows, read):
    for r in rows[:10]:
        read(s, r["path"])
    t0 = time.perf_counter()
    preds = [read(s, r["path"]) for r in rows]
    ms = 1000 * (time.perf_counter() - t0)
    c = float(np.mean([cer(r["text"], p) for r, p in zip(rows, preds)]))
    print(f"{name:20s} {ms:8.0f} ms  {ms / len(rows):6.1f} ms/line  CER {100 * c:.2f}%", flush=True)
    return {"ms_total": round(ms), "ms_per_line": round(ms / len(rows), 2), "cer": round(c, 5)}, preds


def main():
    base = TESS / "data/real"
    rows = [json.loads(l) for l in open(base / "labels.jsonl", encoding="utf-8")]
    rows = [r for r in rows if r.get("set") == "funsd"]
    for r in rows:
        p = base / r["file"]
        r["path"] = p if p.exists() else TESS / r["file"]
    small_chars, tiny_chars = pp_chars("small"), pp_chars("tiny")
    latin_path, latin_chars = latin_head(small_chars)
    from onnxruntime.quantization import QuantType, quantize_dynamic

    i8_path = WORK / "pp_small_latin_i8.onnx"
    quantize_dynamic(str(latin_path), str(i8_path), op_types_to_quantize=["MatMul"], weight_type=QuantType.QInt8)

    def pp_read(chars):
        return lambda s, p: ctc(s.run(None, {"x": pp_prep(p)})[0][0].argmax(-1), chars)

    ocria_chars = json.load(open(OCRIA / "rec_latin.charset.json"))

    def ocria_read(s, p):
        x, rw = ocria_prep(p)
        ids = s.run(["ids"], {"x": x})[0][0]
        stride = 4
        return ctc(ids[: rw // stride], ocria_chars, offset=1)

    arms = [("pp_small", ROOT / "models/small_rec/inference.onnx", pp_read(small_chars)),
            ("pp_small_latin", latin_path, pp_read(latin_chars)),
            ("pp_small_latin_i8", i8_path, pp_read(latin_chars)),
            ("pp_tiny", ROOT / "models/tiny_rec/inference.onnx", pp_read(tiny_chars)),
            ("ocria_latin", OCRIA / "rec_latin.onnx", ocria_read)]
    res, preds = {"lines": len(rows), "threads": 1, "onnxruntime": ort.__version__, "arms": {}}, {}
    for name, path, read in arms:
        res["arms"][name], preds[name] = run_arm(name, sess(path), rows, read)
    same = sum(a == b for a, b in zip(preds["pp_small"], preds["pp_small_latin"]))
    res["pp_small_latin_same_text_as_pp_small"] = same
    print(f"pp_small_latin reads the same text as pp_small on {same}/{len(rows)} lines")
    out = ROOT / "results/cpu_speed.json"
    out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
