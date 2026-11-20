"""Score eng .traineddata files (stock tessdata_best and fine-tuned ones) on the experiment's evaluation sets.

    python3 run/ft/eval_models.py stock=models/best/eng.traineddata A=data/ft/kaggle1/armA/eng.traineddata ...
    -> results/ft_eval.json

Two readers per model, on 800 real lines (FUNSD forms, SROIE receipts; CER after casefolding, since SROIE's truth
is uppercase) and 2,400 synthetic eng lines (8 conditions):
  engine  the exact engine (server/torch_engine.py), tessgpu.lineprep.clean_line first, beam search with and
          without the model's word lists (tessgpu.beam / beam_dict)
  cli     the stock Tesseract 5.3.4 CLI with the file dropped into a tessdata dir (--oem 1 --psm 7, raw crops, as
          run/tess_baseline.py), which is how anyone else would use a fine-tuned model
None of these lines were trained on: the fine-tuning data is generated text (eval texts excluded by hash) and SROIE
receipts 30 onward, while the SROIE evaluation lines come from receipts 0 to 8.
"""
import json
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
from rapidfuzz.distance import Levenshtein

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def norm(s):
    return " ".join(s.split())


def cer(gt, pred, fold):
    g, p = norm(gt), norm(pred)
    if fold:
        g, p = g.casefold(), p.casefold()
    return Levenshtein.distance(g, p) / max(1, len(g))


def rows():
    out = []
    for r in map(json.loads, open(ROOT / "data/real/labels.jsonl", encoding="utf-8")):
        out.append(dict(group=f"real_{r['set']}", cond="real", path=str(ROOT / r["file"]), text=r["text"], fold=True))
    for r in map(json.loads, open(ROOT / "data/lines/labels.jsonl", encoding="utf-8")):
        if r["lang"] == "eng":
            out.append(dict(group="synth_eng", cond=r["cond"], path=str(ROOT / "data/lines" / r["file"]), text=r["text"], fold=False))
    return out


def summarise(rs, preds):
    groups = {}
    for r, p in zip(rs, preds):
        if p is None:
            continue
        c = cer(r["text"], p, r["fold"])
        for k in (r["group"], f"{r['group']}:{r['cond']}" if r["group"] == "synth_eng" else None):
            if k:
                groups.setdefault(k, []).append(c)
    out = {k: round(float(np.mean(v)), 5) for k, v in sorted(groups.items())}
    out["n_failed"] = sum(p is None for p in preds)
    real = [cer(r["text"], p, True) for r, p in zip(rs, preds) if p is not None and r["cond"] == "real"]
    out["real_all"] = round(float(np.mean(real)), 5)
    return out


def engine_preds(path, rs):
    import torch
    from server.torch_engine import TorchTess
    from tessgpu import beam, beam_dict, lineprep
    torch.set_num_threads(4)
    eng = TorchTess(path, mode="exact")
    d = beam_dict.load_dict(eng.model)
    grays = [lineprep.clean_line(np.asarray(Image.open(r["path"]).convert("L"))) for r in rs]
    off, on = [], []
    for c0 in range(0, len(grays), 200):
        for p in eng.recognize(grays[c0:c0 + 200]):
            for dd, acc in ((None, off), (d, on)):
                ids = beam.decode(p, eng.model, dict=dd) if len(p) else []
                acc.append(norm("".join(eng.model.unicharset[u] for u in ids)))
    return off, on


def cli_preds(path, rs, jobs=4):
    tmp = tempfile.mkdtemp()
    shutil.copy(path, Path(tmp) / "eng.traineddata")

    def one(r):
        p = subprocess.run(["tesseract", r["path"], "stdout", "--oem", "1", "--psm", "7", "--tessdata-dir", tmp, "-l", "eng"],
                           capture_output=True, text=True, timeout=120, env={"OMP_THREAD_LIMIT": "1", "PATH": "/usr/bin:/bin"})
        return p.stdout.strip() if p.returncode == 0 else None

    with ThreadPoolExecutor(jobs) as ex:
        out = list(ex.map(one, rs))
    shutil.rmtree(tmp)
    return out


def main():
    models = dict(a.split("=", 1) for a in sys.argv[1:])
    rs = rows()
    out_path = ROOT / "results/ft_eval.json"
    res = json.loads(out_path.read_text()) if out_path.exists() else {}
    res["note"] = __doc__.strip().split("\n\n")[0] + " See run/ft/eval_models.py."
    for name, path in models.items():
        print(f"== {name}: {path}", flush=True)
        off, on = engine_preds(path, rs)
        cli = cli_preds(path, rs)
        res[name] = {"file": str(path), "engine_nodict": summarise(rs, off), "engine_dict": summarise(rs, on),
                     "cli": summarise(rs, cli)}
        for k in ("engine_nodict", "engine_dict", "cli"):
            s = res[name][k]
            print(f"  {k:14} real {100*s['real_all']:.2f}% (funsd {100*s['real_funsd']:.2f}, sroie {100*s['real_sroie']:.2f})  synth eng {100*s['synth_eng']:.3f}%", flush=True)
        out_path.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
