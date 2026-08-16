#!/usr/bin/env python3
"""Grade candidate solutions against hidden test suites.

Usage:
  python3 grade.py --solutions solutions.json [--out results.json]
  python3 grade.py --self-test          # grade the reference solutions

solutions.json: {"<arm>": {"<task>": "<python source>", ...}, ...}
Output JSON:    {"<arm>": {"<task>": {"passed": bool, "n_passed": int, "n_total": int,
                                      "failures": "<trimmed pytest output>"}}}

Each solution runs in a throwaway directory with the task's tests.py; pytest runs with a
30 s timeout per task. Portable: paths are resolved relative to this file.
"""
import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TASKS_DIR = ROOT / "tasks"

_SUMMARY = re.compile(r"(\d+) passed")
_FAILED = re.compile(r"(\d+) failed")
_ERRORS = re.compile(r"(\d+) errors?")


def grade_one(task: str, source: str) -> dict:
    tests = (TASKS_DIR / task / "tests.py").read_text()
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        (tdp / "solution.py").write_text(source)
        (tdp / "test_hidden.py").write_text(tests)
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "pytest", "test_hidden.py", "-q", "--tb=line", "-p", "no:cacheprovider"],
                cwd=td, capture_output=True, text=True, timeout=30,
            )
            out = proc.stdout + proc.stderr
        except subprocess.TimeoutExpired:
            return {"passed": False, "n_passed": 0, "n_total": -1, "failures": "TIMEOUT (30s)"}
    n_passed = int(m.group(1)) if (m := _SUMMARY.search(out)) else 0
    n_failed = int(m.group(1)) if (m := _FAILED.search(out)) else 0
    n_err = int(m.group(1)) if (m := _ERRORS.search(out)) else 0
    passed = proc.returncode == 0 and n_passed > 0
    failures = "" if passed else out[-2000:]
    return {"passed": passed, "n_passed": n_passed,
            "n_total": n_passed + n_failed + n_err, "failures": failures}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solutions")
    ap.add_argument("--out")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        sols = {"reference": {t.name: (t / "reference.py").read_text()
                              for t in sorted(TASKS_DIR.iterdir()) if t.is_dir()}}
    else:
        sols = json.loads(Path(args.solutions).read_text())

    results = {}
    for arm, per_task in sols.items():
        results[arm] = {}
        for task, source in per_task.items():
            r = grade_one(task, source)
            results[arm][task] = r
            status = "PASS" if r["passed"] else f"FAIL ({r['n_passed']}/{r['n_total']})"
            print(f"{arm:24s} {task:18s} {status}")

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=1))
    ok_all = all(r["passed"] for pt in results.values() for r in pt.values())
    sys.exit(0 if (not args.self_test or ok_all) else 1)


if __name__ == "__main__":
    main()
