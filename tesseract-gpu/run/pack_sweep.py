"""What does packing tessdata_best's weights cost? CER on the benchmark sample per format.

    python3 run/pack_sweep.py SAMPLE_DIR [--langs eng,fra,grc] [--formats f32,f16,q8,q6,q4]

Each format replaces every weight matrix by what run/pack.py's format decodes to (pack.fake_quant:
f16 rounding, or per-row codes with scale max|row|/qmax), activations stay float. Lines come from
run/parity_prep.py (already at the model height); inverted lines are skipped (that is an image
problem, measured separately). Text is Tesseract's beam search without dictionary (tessgpu.beam).
Also runs tessdata_fast (Tesseract's own int8 model) for comparison.
Writes results/pack_sweep.json.
"""

import argparse
import copy
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image
from rapidfuzz.distance import Levenshtein

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "run"))
from pack import fake_quant  # noqa: E402
from tessgpu import beam, decode, ref  # noqa: E402
from tessgpu.traineddata import LSTMS, load  # noqa: E402

_CACHE = {}


def packed_model(variant, lang, fmt):
    key = (variant, lang, fmt)
    if key in _CACHE:
        return _CACHE[key]
    m = load(ROOT / f"models/{variant}/{lang}.traineddata")
    if variant == "best" and fmt != "f32":
        m = copy.deepcopy(m)
        for _, n in m.network.walk():
            mats = list(n.gates.values()) if n.type in LSTMS else ([n.weights] if n.weights is not None else [])
            for w in mats:
                w.w = fake_quant(w.w, fmt).astype(np.float64)
    _CACHE[key] = m
    return m


def cer(hyp, gt):
    gt = decode.clean(gt)
    return Levenshtein.distance(decode.clean(hyp), gt) / max(1, len(gt))


def one(job):
    variant, lang, fmt, png, gt = job
    m = packed_model(variant, lang, fmt)
    p = ref.forward(m, np.asarray(Image.open(png)))
    text = decode.clean("".join(m.unicharset[i] for i in beam.decode(p, m)))
    return {"variant": variant, "lang": lang, "fmt": fmt, "png": png, "text": text, "cer": cer(text, gt)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sample")
    ap.add_argument("--langs", default="eng,fra,grc")
    ap.add_argument("--formats", default="f32,f16,q8,q6,q4")
    ap.add_argument("--procs", type=int, default=3)
    a = ap.parse_args()
    sample = Path(a.sample)
    rows = [json.loads(l) for l in open(sample / "sample.jsonl")]
    rows = [r for r in rows if r["cond"] != "invert" and r["lang"] in a.langs.split(",")]
    jobs = []
    for r in rows:
        for fmt in a.formats.split(","):
            jobs.append(("best", r["lang"], fmt, str(sample / r["png"]), r["text"]))
        jobs.append(("fast", r["lang"], "int8", str(sample / r["png"]), r["text"]))
    jobs.sort(key=lambda j: (j[0], j[1], j[2]))  # keep one model per worker chunk
    with Pool(a.procs) as pool:
        res = pool.map(one, jobs, chunksize=16)
    out = {"lines": len(rows), "by": {}}
    base = {(r["lang"], r["png"]): r["text"] for r in res if r["variant"] == "best" and r["fmt"] == "f32"}
    for key in sorted({(r["variant"], r["lang"], r["fmt"]) for r in res}):
        rs = [r for r in res if (r["variant"], r["lang"], r["fmt"]) == key]
        out["by"]["/".join(key)] = {
            "lines": len(rs), "cer": float(np.mean([r["cer"] for r in rs])),
            "same_text_as_f32": float(np.mean([r["text"] == base.get((r["lang"], r["png"])) for r in rs])),
        }
    for lang in a.langs.split(","):
        m = load(ROOT / f"models/best/{lang}.traineddata")
        n = sum(int(np.prod(w.shape)) for _, node in m.network.walk()
                for w in (list(node.gates.values()) if node.type in LSTMS else
                          ([node.weights] if node.weights is not None else [])))
        out.setdefault("params", {})[lang] = n
    (ROOT / "results/pack_sweep.json").write_text(json.dumps(out, indent=1))
    for k, v in out["by"].items():
        print(k, {a: round(b, 5) if isinstance(b, float) else b for a, b in v.items()})


if __name__ == "__main__":
    main()
