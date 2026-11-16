#!/usr/bin/env python3
"""Rerun every check in RESULTS.md and write results/summary.json.

Runs the suites against bm25_v0.2.1.py (the skill as it was) and against that
file with bm25-fix.diff applied, plus the two planted mutants and the coverage
event counts. About 2 minutes.
"""
import json, os, re, subprocess, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
OLD = HERE / "bm25_v0.2.1.py"

def pytest(*args, **env):
    e = {**os.environ, **{k: str(v) for k, v in env.items()}}
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                        "-p", "no:logging", "-W", "ignore", *args],
                       cwd=HERE, env=e, capture_output=True, text=True)
    tail = r.stdout.strip().splitlines()[-1]
    counts = {k: int(v) for v, k in re.findall(r"(\d+) (passed|failed)", tail)}
    return {"passed": counts.get("passed", 0), "failed": counts.get("failed", 0)}, r.stdout

def main():
    fixed = Path(tempfile.mkdtemp()) / "bm25_fixed.py"
    subprocess.run(["patch", "-s", "-o", str(fixed), str(OLD), str(HERE / "bm25-fix.diff")], check=True)
    out = {}
    out["regressions_old"], _ = pytest("test_regressions.py", BM25_PY=OLD)
    out["regressions_fixed"], _ = pytest("test_regressions.py", BM25_PY=fixed)
    out["nocache_old_1500"], _ = pytest("test_bm25_model.py", "-k", "NoCache", BM25_PY=OLD, N=1500)
    out["all_fixed_2000"], _ = pytest("test_bm25_model.py", BM25_PY=fixed, N=2000, SKIP_KNOWN=0)
    out["mutants"] = {}
    for m in ("drop_last", "query_case"):
        res, text = pytest("test_bm25_model.py", "-k", "NoCache", BM25_PY=fixed, N=1500,
                           SKIP_KNOWN=0, BM25_MUTANT=m)
        errs = sorted(set(re.findall(r"^E\s+(\w+Error)", text, re.M)))
        out["mutants"][m] = {"caught": res["failed"] > 0, "errors": errs,
                             "by_assertion": errs == ["AssertionError"]}
    _, text = pytest("test_bm25_model.py", "-k", "NoCache", "--hypothesis-show-statistics",
                     BM25_PY=OLD, N=500)
    ev = {}
    for pct, label in re.findall(r"\* ([\d.]+)%, (.+)", text):
        ev[label.strip()] = float(pct)
    k_lt_n = sum(v for k, v in ev.items() if k.endswith("k<N=True"))
    multi = sum(v for k, v in ev.items() if k.startswith("nonzero-scored docs=") and int(k.split("=")[1]) >= 2)
    out["coverage_pct"] = {"k_lt_n": round(k_lt_n), "two_plus_nonzero": round(multi),
                           "no_match": round(ev.get("nonzero-scored docs=0", 0)), "raw": ev}
    out["regression_tests"] = {"functions": 4, "cases": 5}  # 3 tests + 1 parametrized twice
    out["property_file"] = {"state_machines": 2, "plain_tests": 2}
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "summary.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "coverage_pct"}, indent=1))

if __name__ == "__main__":
    main()
