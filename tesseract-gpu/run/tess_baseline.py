#!/usr/bin/env python3
"""Stock Tesseract baseline on the line benchmark.

Runs `tesseract <png> stdout --oem 1 --psm 7 --tessdata-dir models/<set> -l <lang>`
on every image in a label file, for each model set (best, fast), four
processes at a time, and scores the output by character error rate.

    python3 run/tess_baseline.py                       # data/lines/labels.jsonl
    python3 run/tess_baseline.py --labels data/real/labels.jsonl
    python3 run/tess_baseline.py --summary              # table from results json

Per-image outputs go to <labels dir>/tess_<model>.jsonl (data/lines/tess_best.jsonl
for the default labels). The summary goes to results/tess_baseline.json, or
results/tess_baseline_<dir>.json for other label files (data/real -> _real).

A failed tesseract call is recorded with pred null and an error string. It is
excluded from every statistic and counted in n_failed; it is never scored as an
empty prediction.
"""

import argparse
import json
import statistics
import subprocess
import sys
import time
import unicodedata
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from rapidfuzz.distance import Levenshtein

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LABELS = ROOT / "data" / "lines" / "labels.jsonl"
MODEL_SETS = ["best", "fast"]
COND_ORDER = ["clean", "blur", "noise", "jpeg", "lowres", "rotate", "shade", "invert", "real"]


def norm(s: str) -> str:
    return " ".join(unicodedata.normalize("NFC", s).split())


def cer(gt: str, pred: str) -> float:
    g = norm(gt)
    return Levenshtein.distance(g, norm(pred)) / len(g)


def run_one(row, base, tessdata):
    cmd = ["tesseract", str(base / row["file"]), "stdout", "--oem", "1", "--psm", "7",
           "--tessdata-dir", str(tessdata), "-l", row["lang"]]
    t0 = time.perf_counter()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=120,
                           env={"OMP_THREAD_LIMIT": "1", "PATH": "/usr/bin:/bin"})
        sec = time.perf_counter() - t0
        if p.returncode != 0:
            return {"pred": None, "sec": sec, "error": f"exit {p.returncode}: {p.stderr.strip()[-200:]}"}
        return {"pred": p.stdout.strip(), "sec": sec}
    except (OSError, subprocess.SubprocessError) as e:
        return {"pred": None, "sec": time.perf_counter() - t0, "error": repr(e)[:200]}


def group_key(r):
    k = [r["lang"], r["cond"]]
    if "set" in r:
        k.append(r["set"])
    return tuple(k)


def stats(rows):
    scored = [r for r in rows if r["pred"] is not None]
    cers = [cer(r["text"], r["pred"]) for r in scored]
    exact = [norm(r["text"]) == norm(r["pred"]) for r in scored]
    secs = [r["sec"] for r in rows]
    out = {"n": len(rows), "n_failed": len(rows) - len(scored)}
    if scored:
        out.update({
            "mean_cer": round(statistics.fmean(cers), 5),
            "median_cer": round(statistics.median(cers), 5),
            "exact_line_rate": round(sum(exact) / len(exact), 5),
        })
    else:
        out.update({"mean_cer": None, "median_cer": None, "exact_line_rate": None})
    out["sec_per_line"] = round(statistics.fmean(secs), 4) if secs else None
    return out


def results_path(labels: Path) -> Path:
    if labels.parent.name in ("lines", "") and labels.name == "labels.jsonl":
        return ROOT / "results" / "tess_baseline.json"
    return ROOT / "results" / f"tess_baseline_{labels.parent.name}.json"


def tesseract_version():
    p = subprocess.run(["tesseract", "--version"], capture_output=True, text=True)
    return (p.stdout or p.stderr).splitlines()[0] if (p.stdout or p.stderr) else None


def run(labels: Path, jobs: int):
    rows = [json.loads(line) for line in labels.open(encoding="utf8") if line.strip()]
    outdir = labels.parent
    summary = {"labels": str(labels.relative_to(ROOT)), "tesseract": tesseract_version(),
               "psm": 7, "oem": 1, "jobs": jobs, "n_images": len(rows), "rows": [], "overall": []}
    for model in MODEL_SETS:
        tessdata = ROOT / "models" / model
        print(f"[{model}] {len(rows)} images, {jobs} processes", file=sys.stderr)
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=jobs) as ex:
            results = list(ex.map(lambda r: run_one(r, outdir, tessdata), rows))
        print(f"[{model}] done in {time.perf_counter() - t0:.0f}s", file=sys.stderr)

        merged = []
        for r, res in zip(rows, results):
            m = {**r, "model": model, **res}
            merged.append(m)
        with (outdir / f"tess_{model}.jsonl").open("w", encoding="utf8") as f:
            for m in merged:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")

        groups = defaultdict(list)
        for m in merged:
            groups[group_key(m)].append(m)
        for key in sorted(groups, key=lambda k: (k[0], COND_ORDER.index(k[1]) if k[1] in COND_ORDER else 99, k[2:])):
            entry = dict(zip(["lang", "cond", "set"][:len(key)], key), model=model, **stats(groups[key]))
            summary["rows"].append(entry)
        summary["overall"].append({"model": model, "lang": "all", "cond": "all", **stats(merged)})

    out = results_path(labels)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n", encoding="utf8")
    print(f"wrote {out}", file=sys.stderr)
    return out


def summary(path: Path):
    if not path.exists():
        sys.exit(f"no results at {path}; run tess_baseline.py first")
    d = json.loads(path.read_text(encoding="utf8"))
    print(f"{d['labels']}  {d['tesseract']}  n_images={d['n_images']}")
    print(f"{'model':<6}{'lang':<6}{'cond':<9}{'n':>5}{'fail':>6}{'meanCER':>9}{'medCER':>9}{'exact':>8}{'s/line':>8}")

    def line(e):
        f = lambda v, fmt: "-" if v is None else format(v, fmt)
        print(f"{e['model']:<6}{e.get('lang', ''):<6}{e.get('cond', ''):<9}{e['n']:>5}{e['n_failed']:>6}"
              f"{f(e['mean_cer'], '.4f'):>9}{f(e['median_cer'], '.4f'):>9}"
              f"{f(e['exact_line_rate'], '.3f'):>8}{f(e['sec_per_line'], '.3f'):>8}")

    for e in d["overall"]:
        line(e)
    for e in d["rows"]:
        if e.get("cond") in ("clean", "lowres"):
            line(e)
    print("(all lang x cond rows are in the json)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", type=Path, default=DEFAULT_LABELS,
                    help="label file in the labels.jsonl schema (default data/lines/labels.jsonl)")
    ap.add_argument("--jobs", type=int, default=4, help="parallel tesseract processes (default 4)")
    ap.add_argument("--summary", action="store_true",
                    help="print the table from the results json for --labels and exit")
    args = ap.parse_args()
    labels = args.labels if args.labels.is_absolute() else ROOT / args.labels
    if args.summary:
        summary(results_path(labels))
        return
    run(labels, args.jobs)


if __name__ == "__main__":
    main()
