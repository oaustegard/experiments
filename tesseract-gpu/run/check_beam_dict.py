"""Tesseract's LSTM dictionary in the beam search: parity with Tesseract 5.3.4 and its accuracy.

    python3 run/check_beam_dict.py                  parity + accuracy -> results/beam_dict.json
    python3 run/check_beam_dict.py --parity-only    parity only (keeps any accuracy in the json)
    python3 run/check_beam_dict.py --accuracy-only  accuracy only (keeps any parity in the json)

1. Parity. harness/tess_dump_dict (Tesseract 5.3.4, built and run inside the hub's jail.sh,
   see harness/build_dict.sh, dump_dict_all.sh, dump_dict_list.sh) writes, per line, the
   network softmax, the RecodeBeamSearch ids without a dictionary, and the ids, xcoords and
   words (length, permuter) with the model's LSTM dictionary loaded as LSTMRecognizer::Load
   does. Dumps are read from --dumps (default /tmp/tdump/dict/<set>/*.dict.bin). For every
   line, on Tesseract's own softmax:
     dict ids       tessgpu.beam.decode(p, model, dict=True)  == Tesseract's dict ids
     dict words     beam_dict.decode_full word split + permuters == ExtractBestPathAsWords
     nodict ids     tessgpu.beam.decode(p, model)             == Tesseract's no-dict ids
     nodict (port)  beam_dict.decode(p, model, None)          == Tesseract's no-dict ids
   Sets: the 686 parity lines (343 eng, 172 fra, 171 grc; work/bench/*.list) for best and
   fast, and the 800 real lines (clean_line, scaled to 36 px) for eng best and fast. The
   Python ids and words go to data/beam_dict_py_ids.json for run/check_beam_dict.mjs (the JS
   port in web/beam_dict.js).

2. Accuracy. clean_line + TorchTess(exact) on tessdata_best, then one decode with the
   dictionary off and one with it on over the same softmax. CER is run/tess_baseline.cer (NFC,
   whitespace collapsed); the real set is scored case-insensitively (casefold both sides,
   SROIE ground truth is uppercase). Mean CER per arm, per condition, the paired difference
   with a 95% bootstrap interval over lines (2,000 resamples, seed 0), lines improved/worsened,
   and decode milliseconds per line for each arm. torch threads = 2.

A failure is recorded as null, never as 0.
"""

import argparse
import glob
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tessgpu import beam, beam_dict, tessdump  # noqa: E402
from tessgpu.traineddata import load  # noqa: E402

OUT = os.path.join(ROOT, "results", "beam_dict.json")
PARITY_SETS = [  # (dump set, model variant, lang)
    ("best_eng", "best", "eng"), ("fast_eng", "fast", "eng"),
    ("best_fra", "best", "fra"), ("fast_fra", "fast", "fra"),
    ("best_grc", "best", "grc"), ("fast_grc", "fast", "grc"),
    ("best_eng_real", "best", "eng"), ("fast_eng_real", "fast", "eng"),
]
BAR = 0.99
BATCH = 64
BOOT = 2000


def text_of(model, ids):
    return "".join(model.unicharset[i] if 0 <= i < len(model.unicharset) else f"<{i}>"
                   for i in ids)


# ---------------------------------------------------------------- parity

def parity(dumps, py_ids_path=None):
    out = {}
    models = {}
    py_ids = {}
    for name, variant, lang in PARITY_SETS:
        paths = sorted(glob.glob(os.path.join(dumps, name, "*.dict.bin")))
        if not paths:
            out[name] = {"lines": None, "note": f"no dumps under {dumps}/{name}"}
            print(f"{name:14s} no dumps", flush=True)
            continue
        key = (variant, lang)
        if key not in models:
            m = load(os.path.join(ROOT, "models", variant, f"{lang}.traineddata"))
            models[key] = (m, beam_dict.load_dict(m))
        model, d = models[key]
        n = len(paths)
        same_ids = same_words = same_nodict = same_nodict_port = changed = 0
        mism = []
        ids_by_file = {}
        for p in paths:
            dump = tessdump.read_dict(p)
            probs = dump["probs"]
            full = beam_dict.decode_full(probs, model, d)
            ours = beam.decode(probs, model, dict=d)
            if ours != full["ids"]:
                raise AssertionError(f"{p}: beam.decode(dict=) and decode_full disagree")
            theirs = dump["ids"]
            ids_by_file[os.path.basename(p)[:-len(".dict.bin")]] = {
                "ids": ours, "words": [list(w) for w in full["words"]]}
            words = [(e - s, perm) for s, e, perm in full["words"]]
            twords = [(w["len"], w["perm"]) for w in dump["words"]]
            ok_ids = ours == theirs
            ok_words = words == twords
            same_ids += ok_ids
            same_words += ok_words
            same_nodict += beam.decode(probs, model) == dump["ids_nodict"]
            same_nodict_port += beam_dict.decode(probs, model, None) == dump["ids_nodict"]
            changed += theirs != dump["ids_nodict"]
            if not (ok_ids and ok_words):
                mism.append({"file": os.path.basename(p)[:-len(".png.dict.bin")],
                             "ours": text_of(model, ours), "theirs": text_of(model, theirs),
                             "our_words": words, "their_words": twords})
        py_ids[name] = {"model": f"{variant}/{lang}", "files": ids_by_file}
        out[name] = {
            "model": f"{variant}/{lang}",
            "lines": n,
            "dict_ids_identical": same_ids,
            "dict_ids_share": same_ids / n,
            "dict_words_identical": same_words,
            "nodict_ids_identical": same_nodict,
            "nodict_port_ids_identical": same_nodict_port,
            "lines_where_tesseract_dict_changes_ids": changed,
            "meets_bar_99": same_ids / n >= BAR,
            "mismatches": mism[:20],
        }
        print(f"{name:14s} lines={n:4d} dict_ids={same_ids:4d} dict_words={same_words:4d} "
              f"nodict={same_nodict:4d} nodict_port={same_nodict_port:4d} "
              f"dict_changes_tess={changed:4d}", flush=True)
    if py_ids_path:
        os.makedirs(os.path.dirname(py_ids_path), exist_ok=True)
        with open(py_ids_path, "w") as f:
            json.dump(py_ids, f)
    return out


# ---------------------------------------------------------------- accuracy

def boot_ci(diff, seed=0):
    diff = np.asarray(diff, dtype=np.float64)
    if len(diff) == 0:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(BOOT, len(diff)))
    means = diff[idx].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def summarize(rows):
    off = [r["cer_off"] for r in rows]
    on = [r["cer_on"] for r in rows]
    diff = [b - a for a, b in zip(off, on)]
    n = len(rows)
    return {
        "n": n,
        "cer_off": round(float(np.mean(off)), 5) if n else None,
        "cer_on": round(float(np.mean(on)), 5) if n else None,
        "delta_on_minus_off": round(float(np.mean(diff)), 5) if n else None,
        "delta_ci95": [round(x, 5) for x in boot_ci(diff)] if n else None,
        "exact_off": round(sum(c == 0 for c in off) / n, 4) if n else None,
        "exact_on": round(sum(c == 0 for c in on) / n, 4) if n else None,
        "lines_text_changed": sum(r["changed"] for r in rows),
        "lines_better": sum(d < 0 for d in diff),
        "lines_worse": sum(d > 0 for d in diff),
        "decode_ms_off": round(1000 * float(np.mean([r["t_off"] for r in rows])), 2) if n else None,
        "decode_ms_on": round(1000 * float(np.mean([r["t_on"] for r in rows])), 2) if n else None,
    }


def accuracy(examples_out):
    import torch
    from PIL import Image

    from run.tess_baseline import cer, norm
    from server.torch_engine import TorchTess
    from tessgpu import lineprep

    torch.set_num_threads(2)
    sets = []
    for line in open(os.path.join(ROOT, "data", "lines", "labels.jsonl")):
        r = json.loads(line)
        r["path"] = os.path.join(ROOT, "data", "lines", r["file"])
        r["setname"] = f"synth_{r['lang']}"
        sets.append(r)
    for line in open(os.path.join(ROOT, "data", "real", "labels.jsonl")):
        r = json.loads(line)
        r["path"] = os.path.join(ROOT, r["file"])
        r["setname"] = "real_eng"
        sets.append(r)

    by_lang = defaultdict(list)
    for r in sets:
        by_lang[r["lang"]].append(r)

    rows_by_set = defaultdict(list)
    examples = defaultdict(list)
    for lang in ("eng", "fra", "grc"):
        eng = TorchTess(os.path.join(ROOT, "models", "best", f"{lang}.traineddata"),
                        device="cpu", mode="exact")
        model = eng.model
        d = beam_dict.load_dict(model)
        if d is None:
            print(f"{lang}: model has no LSTM dawgs, skipped", flush=True)
            continue
        todo = by_lang[lang]
        t_rec = 0.0
        for c0 in range(0, len(todo), BATCH):
            chunk = todo[c0:c0 + BATCH]
            imgs = [lineprep.clean_line(np.asarray(Image.open(r["path"]).convert("L")))
                    for r in chunk]
            t0 = time.perf_counter()
            probs = eng.recognize(imgs)
            t_rec += time.perf_counter() - t0
            for r, p in zip(chunk, probs):
                t0 = time.perf_counter()
                ids_off = beam.decode(p, model)
                t1 = time.perf_counter()
                ids_on = beam.decode(p, model, dict=d)
                t2 = time.perf_counter()
                off = norm(text_of(model, ids_off))
                on = norm(text_of(model, ids_on))
                gt = r["text"]
                if r["setname"] == "real_eng":
                    c_off = cer(gt.casefold(), off.casefold())
                    c_on = cer(gt.casefold(), on.casefold())
                else:
                    c_off = cer(gt, off)
                    c_on = cer(gt, on)
                row = {"cond": r.get("set", r["cond"]) if r["setname"] == "real_eng" else r["cond"],
                       "cer_off": c_off, "cer_on": c_on, "changed": off != on,
                       "t_off": t1 - t0, "t_on": t2 - t1}
                rows_by_set[r["setname"]].append(row)
                if off != on:
                    examples[r["setname"]].append({"id": r["id"], "cond": row["cond"], "gt": gt,
                                                   "off": off, "on": on,
                                                   "cer_off": round(c_off, 4),
                                                   "cer_on": round(c_on, 4)})
            print(f"{lang} {min(c0 + BATCH, len(todo))}/{len(todo)} rec {t_rec:.0f}s", flush=True)

    out = {}
    for name, rows in rows_by_set.items():
        per_cond = defaultdict(list)
        for row in rows:
            per_cond[row["cond"]].append(row)
        out[name] = {"all": summarize(rows),
                     "by_cond": {c: summarize(v) for c, v in per_cond.items()}}
        s = out[name]["all"]
        print(f"{name:10s} n={s['n']} cer off {s['cer_off']:.5f} on {s['cer_on']:.5f} "
              f"delta {s['delta_on_minus_off']:+.5f} ci {s['delta_ci95']} "
              f"better {s['lines_better']} worse {s['lines_worse']} "
              f"ms off {s['decode_ms_off']} on {s['decode_ms_on']}", flush=True)
    if examples_out:
        with open(examples_out, "w") as f:
            json.dump(examples, f, indent=1, ensure_ascii=False)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", default="/tmp/tdump/dict")
    ap.add_argument("--parity-only", action="store_true")
    ap.add_argument("--accuracy-only", action="store_true")
    ap.add_argument("--examples", default=os.path.join(ROOT, "data", "beam_dict_examples.json"),
                    help="every line whose text the dictionary changes (data/ is gitignored)")
    ap.add_argument("--py-ids", default=os.path.join(ROOT, "data", "beam_dict_py_ids.json"),
                    help="Python dict ids and words per dump, for run/check_beam_dict.mjs")
    a = ap.parse_args()
    t0 = time.time()
    res = {}
    if os.path.exists(OUT):
        res = json.load(open(OUT))
    res["what"] = ("Tesseract 5.3.4 LSTM beam search with its dictionary (tessgpu/beam_dict.py): "
                   "parity with the library and CER with the dictionary off and on")
    res["constants"] = {"dict_ratio": beam.DICT_RATIO, "cert_offset": beam.CERT_OFFSET,
                        "worst_dict_cert": beam_dict.WORST_DICT_CERT,
                        "dawgs": "lstm-punc-dawg, lstm-word-dawg, lstm-number-dawg"}
    if not a.accuracy_only:
        t1 = time.time()
        res["parity"] = parity(a.dumps, a.py_ids)
        res["parity_wall_seconds"] = round(time.time() - t1, 1)
    if not a.parity_only:
        t1 = time.time()
        res["accuracy"] = accuracy(a.examples)
        res["accuracy_setup"] = ("clean_line + TorchTess exact, tessdata_best per language, "
                                 "dictionary off vs on over the same softmax; real set "
                                 "case-insensitive; torch threads 2")
        res["accuracy_wall_seconds"] = round(time.time() - t1, 1)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    ok = all(v.get("meets_bar_99") for v in res.get("parity", {}).values()
             if v.get("lines"))
    print(f"parity_bar_met={ok} wall={time.time() - t0:.1f}s")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
