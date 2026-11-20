"""Stock Tesseract on the OPB text subset, two ways: the benchmark's published verdicts, and this run.

    python3 run/opb/baseline.py          # after prepare.py and tesseract_cli.py

(a) The published Tesseract row of results.parquet, for each of the subset's text-family tests (gold
    ids; a test with no published row counts as missing). Its pass rate counts every subset test in the
    denominator, as the benchmark's summarize does, so an `unsupported` verdict is not a pass.
(b) The local run data/opb/out/tess_cli, scored by run/opb/score.py (inside the jail).

Writes results/opb_baseline.json. Reads data only; it imports nothing from the benchmark package.
"""
from __future__ import annotations

import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "opb"
TEXT_TYPES = frozenset({"present", "order", "repeat", "page_near_empty"})
RUN = DATA / "out" / "tess_cli"
TESSDATA = Path("/usr/share/tesseract-ocr/5/tessdata/eng.traineddata")


def gold_text_tests(ds: Path, subset: dict) -> dict[str, tuple[str, str]]:
    """test_id -> (page_id, suite) for every text-family test of the subset, from the gold files."""
    out = {}
    for p in subset["pages"]:
        for t in json.loads((ds / p["gt_path"]).read_text())["tests"]:
            if t["test_type"] in TEXT_TYPES:
                out[t["id"]] = (p["page_id"], p["suite"])
    return out


def rate(passed: int, total: int) -> float | None:
    return round(passed / total, 4) if total else None


def published(ds: Path, tests: dict[str, tuple[str, str]]) -> dict:
    df = pq.read_table(ds / "results.parquet", columns=["system", "test_id", "result", "why"],
                       filters=[("system", "=", "Tesseract")]).to_pandas()
    got = dict(zip(df.test_id, df.result))
    res = {tid: got.get(tid, "missing") for tid in tests}
    by_suite: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for tid, (_, suite) in tests.items():
        by_suite[suite][1] += 1
        by_suite[suite][0] += res[tid] == "pass"
    vals = list(res.values())
    n, p = len(tests), vals.count("pass")
    return {"source": "results.parquet, system Tesseract (the benchmark's own run: its adapter, "
                      "the page's script detected with --psm 0, PDFs at 300 DPI, no text layer)",
            "tests": n, "pass": p, "pass_rate": rate(p, n),
            "fail": vals.count("fail"), "unsupported": vals.count("unsupported"),
            "missing": vals.count("missing"),
            "by_suite": {s: {"tests": t, "pass_rate": rate(ps, t)} for s, (ps, t) in sorted(by_suite.items())}}


def local(tests: dict[str, tuple[str, str]]) -> dict:
    proc = subprocess.run([sys.executable, str(REPO / "run" / "opb" / "score.py"), "--run", str(RUN)],
                          capture_output=True, text=True)
    if proc.returncode:
        sys.exit(f"baseline.py: score.py failed:\n{proc.stderr[-2000:]}\n{proc.stdout[-500:]}")
    s = json.loads((RUN / "scored" / "summary.json").read_text())
    scored_ids = {ln["test_id"] for f in (RUN / "scored" / "pages").glob("*.json")
                  for ln in json.loads(f.read_text())["lines"]}
    version = subprocess.run(["tesseract", "--version"], capture_output=True, text=True).stdout.splitlines()[0]
    return {"command": "tesseract <page>.png - --psm 3 --oem 1 -l eng (OMP_THREAD_LIMIT=1)",
            "version": version,
            "tessdata": {"file": str(TESSDATA), "bytes": TESSDATA.stat().st_size},
            "tests": s["tests"], "pass": s["passed"], "pass_rate": round(s["pass_rate"], 4),
            "pages": s["pages"], "pages_missing": s["pages_missing"], "failed_pages": s["failed_pages"],
            "same_test_set_as_gold": scored_ids == set(tests),
            "by_suite": {k: {"tests": v["tests"], "pass_rate": round(v["score"], 4)}
                         for k, v in s["per_suite"].items()}}


def main():
    ds = DATA / "ds"
    subset = json.loads((DATA / "subset.json").read_text())
    tests = gold_text_tests(ds, subset)
    a = published(ds, tests)
    b = local(tests)
    gap = {s: round(100 * (b["by_suite"][s]["pass_rate"] - a["by_suite"][s]["pass_rate"]), 2)
           for s in sorted(set(a["by_suite"]) & set(b["by_suite"]))}
    out = {"dataset": {"repo": "datalab-to/omni_parse_bench", "revision": subset["revision"]},
           "subset": {"pages": subset["n_pages"], "text_tests": len(tests),
                      "text_types": sorted(TEXT_TYPES), "seed": subset["seed"]},
           "a_published_tesseract": a,
           "b_local_tesseract_cli": b,
           "gap_b_minus_a_pp": round(100 * (b["pass_rate"] - a["pass_rate"]), 2),
           "gap_by_suite_pp": gap}
    (REPO / "results" / "opb_baseline.json").write_text(json.dumps(out, indent=1))
    print(f"(a) published Tesseract: {a['pass_rate']} ({a['pass']}/{a['tests']})")
    print(f"(b) local tesseract CLI: {b['pass_rate']} ({b['pass']}/{b['tests']}), "
          f"{b['pages_missing']} missing pages")
    print(f"gap b - a: {out['gap_b_minus_a_pp']} pp; written results/opb_baseline.json")


if __name__ == "__main__":
    main()
