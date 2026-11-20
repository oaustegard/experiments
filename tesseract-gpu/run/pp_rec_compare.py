"""Read the benchmark line crops with PP-OCRv6's recognizers and compare against Tesseract's readings.

Same crops, same ground truth, same CER (NFC, whitespace collapsed; tess_baseline.norm/cer), so the
only thing that changes is the recognizer.

    python3 run/pp_rec_compare.py run tiny small medium    # per-line outputs to data/<set>/pp_<model>.jsonl
    python3 run/pp_rec_compare.py summary                  # results/pp_rec_compare.json

Models: PaddlePaddle/PP-OCRv6_{tiny,small,medium}_rec_onnx (Apache-2.0) in models/pprec/<model>/.
Preprocessing follows PaddleOCR's RecResizeImg for image_shape [3, 48, 320]: BGR, resize to height 48
keeping the aspect ratio, (x/255 - 0.5)/0.5, pad right with zeros to max(320, resized width). One line
per inference call, so no line is padded to a longer neighbour. Greedy CTC decode with blank 0 and a
space appended to the dictionary, as PaddleOCR's CTCLabelDecode does.
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "run"))
from tess_baseline import cer  # noqa: E402

SETS = {"synth": ROOT / "data/lines", "real": ROOT / "data/real"}


def load_rows(name):
    base = SETS[name]
    rows = [json.loads(l) for l in open(base / "labels.jsonl", encoding="utf-8")]
    for r in rows:
        p = ROOT / r["file"]
        r["path"] = str(p if p.exists() else base / r["file"])
    return rows


class PPRec:
    def __init__(self, model):
        d = ROOT / "models/pprec" / model
        cfg = yaml.safe_load(open(d / "inference.yml"))
        self.chars = ["<blank>"] + list(cfg["PostProcess"]["character_dict"]) + [" "]
        so = ort.SessionOptions()
        so.intra_op_num_threads = 4
        self.sess = ort.InferenceSession(str(d / "inference.onnx"), so, providers=["CPUExecutionProvider"])

    def prep(self, path):
        im = Image.open(path).convert("RGB")
        w, h = im.size
        rw = max(1, min(3200, math.ceil(48 * w / h)))
        im = im.resize((rw, 48), Image.BILINEAR)
        a = np.asarray(im, dtype=np.float32)[:, :, ::-1]  # BGR
        a = (a / 255.0 - 0.5) / 0.5
        x = np.zeros((1, 3, 48, max(320, rw)), np.float32)
        x[0, :, :, :rw] = a.transpose(2, 0, 1)
        return x

    def read(self, path):
        out = self.sess.run(None, {"x": self.prep(path)})[0][0]
        ids = out.argmax(-1)
        text, prev = [], -1
        for i in ids:
            if i != prev and i != 0:
                text.append(self.chars[i])
            prev = i
        return "".join(text)


def run(models):
    for model in models:
        rec = PPRec(model)
        for name, base in SETS.items():
            out = base / f"pp_{model}.jsonl"
            done = {}
            if out.exists():
                for l in open(out, encoding="utf-8"):
                    r = json.loads(l)
                    done[r["file"]] = r
            rows = [r for r in load_rows(name) if r["file"] not in done]
            t0 = time.perf_counter()
            with open(out, "a", encoding="utf-8") as f:
                for i, r in enumerate(rows):
                    t = time.perf_counter()
                    try:
                        pred, err = rec.read(r["path"]), None
                    except Exception as e:  # recorded, never scored as empty
                        pred, err = None, repr(e)[:200]
                    rec_row = {k: r[k] for k in ("id", "file", "lang", "cond", "text")}
                    rec_row.update(model=model, pred=pred, sec=time.perf_counter() - t)
                    if err:
                        rec_row["error"] = err
                    f.write(json.dumps(rec_row, ensure_ascii=False) + "\n")
                    if i % 500 == 0:
                        f.flush()
                        print(f"{model} {name} {i}/{len(rows)} {time.perf_counter() - t0:.0f}s", flush=True)
            print(f"{model} {name} done {len(rows)} lines {time.perf_counter() - t0:.0f}s", flush=True)


def summary():
    readers = {"tess_best": "tess_best.jsonl", "tess_fast": "tess_fast.jsonl"}
    for m in ("tiny", "small", "medium"):
        readers[f"pp_{m}"] = f"pp_{m}.jsonl"
    groups = {}
    for name, base in SETS.items():
        for reader, fn in readers.items():
            p = base / fn
            if not p.exists():
                continue
            for l in open(p, encoding="utf-8"):
                r = json.loads(l)
                if r.get("pred") is None:
                    groups.setdefault((name, reader, r["lang"], "failed"), []).append(None)
                    continue
                c = (cer(r["text"], r["pred"]), cer(r["text"].casefold(), r["pred"].casefold()))
                cond = "real" if name == "real" else r["cond"]
                for key in ((name, reader, r["lang"], "all"), (name, reader, r["lang"], cond)):
                    groups.setdefault(key, []).append((c, float(r.get("sec") or 0)))
    rows = []
    for (name, reader, lang, cond), v in sorted(groups.items()):
        if cond == "failed":
            continue
        c = np.array([x[0][0] for x in v])
        ci = np.array([x[0][1] for x in v])
        rows.append(dict(set=name, reader=reader, lang=lang, cond=cond, n=len(v),
                         n_failed=len(groups.get((name, reader, lang, "failed"), [])),
                         mean_cer=round(float(c.mean()), 5), mean_cer_nocase=round(float(ci.mean()), 5), exact=round(float((c == 0).mean()), 4),
                         ms_per_line=round(1000 * float(np.mean([x[1] for x in v])), 1)))
    res = {"note": "CER = Levenshtein(norm(gt), norm(pred))/len(norm(gt)), norm = NFC + collapsed whitespace; _nocase casefolds both sides (SROIE truth is uppercase). "
                   "Tesseract: CLI 5.3.4 --oem 1 --psm 7 (run/tess_baseline.py), 4 processes. "
                   "PP-OCRv6: onnxruntime CPU, 4 threads, one line per call.",
           "rows": rows}
    out = ROOT / "results/pp_rec_compare.json"
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    for r in rows:
        if r["cond"] in ("all", "real"):
            print(f"{r['set']:5} {r['lang']} {r['cond']:5} {r['reader']:10} n={r['n']:4} cer={100*r['mean_cer']:6.2f}% nocase={100*r['mean_cer_nocase']:6.2f}% exact={100*r['exact']:5.1f}% {r['ms_per_line']:6.1f} ms")


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2:] or ["tiny", "small", "medium"])
    else:
        summary()
