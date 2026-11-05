"""Build the demo page's model directory, web/models/decider/, from a packed model.

    python3 -I run/build_page.py PACKED_DIR_NAME [--cases-per-workflow 25]

Copies weights.bin, manifest.json and tokenizer.json from out/web/<PACKED_DIR_NAME>/, writes
cases.json (test cases with state, questions and gold, for the page's "random test case") and
about.json (brotli-11 download size; test accuracy of the packed model from
results/packed_<PACKED_DIR_NAME>.json, which run/pack.py --score writes).
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import brotli
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, WORKFLOWS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("packed")
    ap.add_argument("--cases-per-workflow", type=int, default=25)
    a = ap.parse_args()
    src = ROOT / "out" / "web" / a.packed
    dst = ROOT / "web" / "models" / "decider"
    dst.mkdir(parents=True, exist_ok=True)
    for f in ("weights.bin", "manifest.json", "tokenizer.json"):
        shutil.copy(src / f, dst / f)
    cases = []
    for w in WORKFLOWS:
        df = pd.read_parquet(ROOT / "data" / "td" / w / "test.parquet")
        for r in df.head(a.cases_per_workflow).itertuples():
            cases.append({"id": r.id, "workflow": w, "state": r.state,
                          "questions": json.loads(r.questions), "gold": json.loads(r.gold)})
    (dst / "cases.json").write_text(json.dumps(cases))
    sizes = {f: len(brotli.compress((dst / f).read_bytes(), quality=11)) for f in ("weights.bin", "tokenizer.json", "manifest.json")}
    js = sum(len(brotli.compress(p.read_bytes(), quality=11)) for p in
             [ROOT / "web" / "decider.js", ROOT / "web" / "engine.js", ROOT / "web" / "wgsl" / "runtime.js", ROOT / "web" / "wgsl" / "kernels.js"])
    packed = json.loads((ROOT / "results" / f"packed_{a.packed}.json").read_text())
    about = {"packed": a.packed, "brotliBytes": sizes, "jsBrotliBytes": js,
             "downloadMB": (sum(sizes.values()) + js) / 1e6, "testAccuracy": packed["accuracy"],
             "testKL": packed["kl_from_gold"], "testBrier": packed["brier"], "testECE": packed["ece"]}
    (dst / "about.json").write_text(json.dumps(about, indent=1))
    (ROOT / "results" / "page_build.json").write_text(json.dumps(about, indent=1))
    print(json.dumps(about, indent=1))


if __name__ == "__main__":
    main()
