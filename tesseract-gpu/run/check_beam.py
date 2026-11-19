"""Check tessgpu.beam against Tesseract 5.3.4's own RecodeBeamSearch output.

For every dump in /tmp/tdump/bench/<variant>_<lang>/*.png.bin, decode Tesseract's own softmax
probs with tessgpu.beam.decode and compare the unichar ids with the ids Tesseract returned.
Also reports plain greedy decoding (decode.best_path + Codec) as a contrast.

Writes:
  results/beam_check.json     per model: lines, identical, share, first 10 mismatches
  data/beam_py_ids.json       Python ids per dump plus the model fields, for run/check_beam.mjs
                              (data/ is gitignored and regenerable)

Exit status is 0 only when every model clears the 99.5% bar.
"""

import glob
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tessgpu import beam, decode, tessdump  # noqa: E402
from tessgpu.traineddata import load  # noqa: E402

BENCH = "/tmp/tdump/bench"
VARIANTS = ["best", "fast"]
LANGS = ["eng", "fra", "grc"]
BAR = 0.995


def text(model, ids):
    return "".join(model.unicharset[i] if 0 <= i < len(model.unicharset) else f"<{i}>"
                   for i in ids)


def main():
    t0 = time.time()
    results = {}
    py_ids = {}
    models_js = {}
    all_ok = True
    for variant in VARIANTS:
        for lang in LANGS:
            key = f"{variant}/{lang}"
            model = load(os.path.join(ROOT, "models", variant, f"{lang}.traineddata"))
            codec = decode.Codec(model)
            paths = sorted(glob.glob(os.path.join(BENCH, f"{variant}_{lang}", "*.png.bin")))
            identical = 0
            greedy_same = 0
            mismatches = []
            ids_by_file = {}
            for p in paths:
                name = os.path.basename(p)[: -len(".png.bin")]
                d = tessdump.read(p)
                ours = beam.decode(d["probs"], model)
                theirs = list(d["ids"])
                ids_by_file[name] = ours
                if ours == theirs:
                    identical += 1
                else:
                    mismatches.append({"file": name, "ours": text(model, ours),
                                       "theirs": text(model, theirs)})
                codes, _ = decode.best_path(d["probs"], model.null_char)
                if codec.codes_to_text(codes) == text(model, theirs):
                    greedy_same += 1
            n = len(paths)
            share = identical / n if n else None
            ok = share is not None and share >= BAR
            all_ok = all_ok and ok
            results[key] = {
                "lines": n,
                "identical": identical,
                "share": share,
                "meets_bar_99_5": ok,
                "greedy_text_identical": greedy_same,
                "greedy_text_share": greedy_same / n if n else None,
                "mismatches_first10": mismatches[:10],
            }
            py_ids[key] = ids_by_file
            models_js[key] = {
                "null_char": model.null_char,
                "codes": [list(c) for c in model.recoder],
                "unichars": list(model.unicharset),
                "simple_text": bool(model.simple_text),
            }
            print(f"{key:10s} lines={n:4d} identical={identical:4d} share={share:.4f} "
                  f"greedy_text_share={greedy_same / n:.4f}", flush=True)

    os.makedirs(os.path.join(ROOT, "data"), exist_ok=True)
    with open(os.path.join(ROOT, "data", "beam_py_ids.json"), "w") as f:
        json.dump({"models": models_js, "ids": py_ids}, f)
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    out = {
        "what": "tessgpu.beam.decode vs Tesseract 5.3.4 RecodeBeamSearch ids (no dictionary)",
        "bar": BAR,
        "wall_seconds": round(time.time() - t0, 1),
        "models": results,
    }
    with open(os.path.join(ROOT, "results", "beam_check.json"), "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(f"all_models_meet_bar={all_ok} wall={time.time() - t0:.1f}s")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
