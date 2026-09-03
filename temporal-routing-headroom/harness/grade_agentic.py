#!/usr/bin/env python3
"""Grade agent-edited repositories against the hidden suites.

Input is a runs manifest mapping arm -> task -> the directory the agent worked in:

    {"weak": {"parse_range": "/path/to/run/weak/parse_range", ...}, "strong": {...}}

Grading copies the worktree, restores `tests/` and `tests_hidden.py` from the pristine
task (so a run that edited its own tests cannot buy a pass), then runs the hidden suite.
Tampering is recorded rather than silently repaired: a run that edits tests is a result
about the arm, not a harness error.

Usage:
  python3 grade_agentic.py --runs runs.json --out data/results_stage1.json
  python3 grade_agentic.py --self-test      # grade the pristine (bugged) repos
"""
import argparse
import filecmp
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TASKS = ROOT / "tasks"
PYTEST_TIMEOUT = 120

_PASSED = re.compile(r"(\d+) passed")
_FAILED = re.compile(r"(\d+) failed")
_ERROR = re.compile(r"(\d+) error")


# issue.md is placed in the worktree by emit_prompts.stage(), not by the run.
IGNORE = {"__pycache__", ".pytest_cache", "issue.md"}


def out_of_bounds_edits(worktree, task_dir):
    """Files the run changed outside its package -- the only place it was told to edit.

    Checking `tests/` alone is not enough: two pilot runs answered a broken visible suite
    by writing into the repo's root conftest.py, which is neither a test file nor part of
    the package. Grading rebuilds from the pristine repo plus the run's package directory,
    so anything named here was discarded rather than scored.
    """
    pristine, live = task_dir / "repo", Path(worktree)
    pkg = json.loads((task_dir / "meta.json").read_text())["package"]
    changed = []
    for f in pristine.rglob("*"):
        rel = f.relative_to(pristine)
        if f.is_dir() or rel.parts[0] in (pkg, *IGNORE) or set(rel.parts) & IGNORE:
            continue
        target = live / rel
        if not target.exists() or not filecmp.cmp(f, target, shallow=False):
            changed.append(str(rel))
    for f in live.rglob("*"):
        rel = f.relative_to(live)
        if f.is_dir() or rel.parts[0] in (pkg, *IGNORE) or set(rel.parts) & IGNORE:
            continue
        if not (pristine / rel).exists():
            changed.append(f"+{rel}")
    return sorted(changed)


def grade_one(worktree, task):
    task_dir = TASKS / task
    hidden = (task_dir / "tests_hidden.py").read_text()
    pkg = json.loads((task_dir / "meta.json").read_text())["package"]
    outside = out_of_bounds_edits(worktree, task_dir)
    result = {"edits_outside_package": outside, "tampered_tests": bool(outside)}
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / "w"
        # Rebuild from pristine, then overlay only the package the run was told to edit.
        shutil.copytree(task_dir / "repo", work, ignore=shutil.ignore_patterns(
            "__pycache__", ".pytest_cache", ".git"))
        shutil.rmtree(work / pkg, ignore_errors=True)
        shutil.copytree(Path(worktree) / pkg, work / pkg, ignore=shutil.ignore_patterns(
            "__pycache__", ".pytest_cache", ".git"))
        (work / "tests_hidden.py").write_text(hidden)
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "pytest", "tests_hidden.py", "-q", "--no-header",
                 "-p", "no:cacheprovider"],
                cwd=work, capture_output=True, text=True, timeout=PYTEST_TIMEOUT)
            out = proc.stdout + proc.stderr
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            out, rc = f"TIMEOUT after {PYTEST_TIMEOUT}s", 1
    n_pass = int(m.group(1)) if (m := _PASSED.search(out)) else 0
    n_fail = int(m.group(1)) if (m := _FAILED.search(out)) else 0
    n_err = int(m.group(1)) if (m := _ERROR.search(out)) else 0
    result.update({
        "passed": rc == 0,
        "n_passed": n_pass,
        "n_total": n_pass + n_fail + n_err,
        "failures": trim(out) if rc != 0 else "",
    })
    return result


def trim(out, keep=40):
    lines = [l for l in out.splitlines() if l.strip()]
    return "\n".join(lines if len(lines) <= keep else lines[:keep - 3] + ["..."] + lines[-2:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs")
    ap.add_argument("--out")
    ap.add_argument("--self-test", action="store_true",
                    help="grade the pristine bugged repos; every task must FAIL")
    args = ap.parse_args()

    if args.self_test:
        bad = []
        for task_dir in sorted(TASKS.iterdir()):
            if not (task_dir / "meta.json").exists():
                continue
            r = grade_one(task_dir / "repo", task_dir.name)
            state = "FAIL(expected)" if not r["passed"] else "PASS(UNEXPECTED)"
            print(f"  {task_dir.name:16s} {state:16s} {r['n_passed']}/{r['n_total']}")
            if r["passed"]:
                bad.append(task_dir.name)
        if bad:
            raise SystemExit(f"tasks that do not start red: {bad}")
        print("all task repos start red under the hidden suite")
        return 0

    if not args.runs:
        raise SystemExit("--runs or --self-test required")
    runs = json.loads(Path(args.runs).read_text())
    results = {}
    for arm, per_task in runs.items():
        results[arm] = {}
        for task, worktree in per_task.items():
            results[arm][task] = grade_one(worktree, task)
            r = results[arm][task]
            flag = (f" OUTSIDE:{','.join(r['edits_outside_package'])}"
                    if r["edits_outside_package"] else "")
            print(f"  {arm:14s} {task:16s} {'pass' if r['passed'] else 'fail'} "
                  f"{r['n_passed']}/{r['n_total']}{flag}")
    text = json.dumps(results, indent=2) + "\n"
    if args.out:
        Path(args.out).write_text(text)
        print(f"wrote {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
