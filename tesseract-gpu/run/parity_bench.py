"""Reference vs Tesseract 5.3.4's own recognizer on the resized benchmark sample.

    python3 run/parity_bench.py SAMPLE_DIR DUMP_DIR   # from run/parity_prep.py and the jailed harness
Writes results/parity_tess.json: per model, the largest softmax difference, argmax agreement,
and how often greedy decoding of our outputs equals Tesseract's beam-search text (no dictionary),
with CER of both against the ground truth.
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image
from rapidfuzz.distance import Levenshtein

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tessgpu import decode, ref, tessdump  # noqa: E402
from tessgpu.traineddata import load  # noqa: E402

_M = {}


def cer(hyp, gt):
    gt = decode.clean(gt)
    return Levenshtein.distance(decode.clean(hyp), gt) / max(1, len(gt))


def one(args):
    variant, lang, png, dump, gt = args
    key = (variant, lang)
    if key not in _M:
        m = load(ROOT / f"models/{variant}/{lang}.traineddata")
        _M[key] = (m, decode.Codec(m))
    m, codec = _M[key]
    g = np.asarray(Image.open(png))
    p = ref.forward(m, g)
    d = tessdump.read(dump)
    tp = d["probs"]
    beam = decode.clean("".join(m.unicharset[i] for i in d["ids"]))
    ours = decode.greedy(p, codec)
    return {"variant": variant, "lang": lang, "max_dp": float(np.abs(p - tp).max()),
            "argmax": float((p.argmax(1) == tp.argmax(1)).mean()), "same_text": ours == beam,
            "cer_ours": cer(ours, gt), "cer_beam": cer(beam, gt), "ours": ours, "beam": beam}


def main():
    sample, dumps = Path(sys.argv[1]), Path(sys.argv[2])
    rows = [json.loads(l) for l in open(sample / "sample.jsonl")]
    jobs = []
    for variant in ("best", "fast"):
        for r in rows:
            png = sample / r["png"]
            dump = dumps / f"{variant}_{r['lang']}" / (Path(r["png"]).name + ".bin")
            jobs.append((variant, r["lang"], str(png), str(dump), r["text"]))
    with Pool(3) as pool:
        res = pool.map(one, jobs, chunksize=8)
    summary = {}
    for variant in ("best", "fast"):
        for lang in ("eng", "fra", "grc"):
            rs = [r for r in res if r["variant"] == variant and r["lang"] == lang]
            summary[f"{variant}_{lang}"] = {
                "lines": len(rs),
                "max_dp": max(r["max_dp"] for r in rs),
                "median_line_max_dp": float(np.median([r["max_dp"] for r in rs])),
                "argmax_agree_all_steps": sum(r["argmax"] == 1.0 for r in rs) / len(rs),
                "greedy_eq_beam": sum(r["same_text"] for r in rs) / len(rs),
                "cer_greedy": float(np.mean([r["cer_ours"] for r in rs])),
                "cer_tess_beam": float(np.mean([r["cer_beam"] for r in rs])),
            }
    diffs = [{k: r[k] for k in ("variant", "lang", "ours", "beam")} for r in res if not r["same_text"]]
    out = {"summary": summary, "text_differences": diffs[:60]}
    (ROOT / "results/parity_tess.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    for k, v in summary.items():
        print(k, {a: round(b, 6) if isinstance(b, float) else b for a, b in v.items()})


if __name__ == "__main__":
    main()
