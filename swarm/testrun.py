"""Adapter around tests/run.py plus the per-commit results cache.

tests/run.py contract (owned by tests/, not by the swarm):
    python3 tests/run.py --suite NAME [--jobs N] [--filter GLOB] [--json OUT] [--cc CC]
    --json OUT writes {test_id: {status, reason, stage}}; test_id = "<suite>/<relpath>";
    status in PASS | FAIL | TIMEOUT | CRASH (anything else is treated as not passing).

Results cache (all under swarm/state/):
    results/<commit>.json   every result recorded at that main commit (partial or full)
    known.json              {test_id: {status, reason, stage, commit}}: latest known
                            status of each test along main; the gate compares to this
    baseline.json           {commit, time, suites: {name: {pass, total}}}: last full run
"""

from __future__ import annotations

import concurrent.futures as cf
import fnmatch
import os
import subprocess
import tempfile
import time
from pathlib import Path

from swarm.common import Ctx, log, read_json, write_json

TEST_TIMEOUT = 3600   # whole tests/run.py invocation
# Above this fraction of a suite, run the whole suite instead of per-test filters.
WHOLE_SUITE_FRACTION = 0.25


def is_pass(rec: dict | None) -> bool:
    return bool(rec) and rec.get("status") == "PASS"


def suite_of(test_id: str) -> str:
    return test_id.split("/", 1)[0]


def list_suites(ctx: Ctx, tree: Path) -> list[str]:
    d = tree / "tests" / "suites"
    found = sorted(p.name for p in d.iterdir() if p.is_dir()) if d.is_dir() else []
    return [s for s in found if not ctx.suites or s in ctx.suites]


def _env(tree: Path) -> dict:
    # `python3 -m hcc` must resolve to *this* tree's hcc, whatever cwd run.py uses.
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    env["PYTHONPATH"] = str(tree) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


def run_suite(tree: Path, suite: str, jobs: int, filt: str | None = None) -> dict:
    """Run one suite (optionally filtered) in `tree`; returns {test_id: rec}."""
    with tempfile.TemporaryDirectory(prefix="swarm-tr-") as td:
        out = Path(td) / "r.json"
        cmd = ["python3", "tests/run.py", "--suite", suite, "--jobs", str(jobs), "--json", str(out)]
        if filt:
            cmd += ["--filter", filt]
        try:
            p = subprocess.run(cmd, cwd=tree, env=_env(tree), capture_output=True, text=True,
                               timeout=TEST_TIMEOUT, check=False)
        except subprocess.TimeoutExpired:
            log(f"tests/run.py --suite {suite} timed out in {tree}")
            return {}
        res = read_json(out, None)
        if res is None:
            log(f"tests/run.py --suite {suite} wrote no JSON (rc={p.returncode}): {p.stderr[-300:]}")
            return {}
        return res


def suite_ids(ctx: Ctx, suite: str) -> list[str]:
    """Known test ids of a suite (from known.json)."""
    return [t for t in known(ctx) if suite_of(t) == suite]


def run_ids(ctx: Ctx, tree: Path, ids: list[str], jobs: int | None = None) -> dict:
    """Run exactly these test ids in `tree`. Missing results come back as status MISSING."""
    jobs = jobs or ctx.jobs
    by_suite: dict[str, list[str]] = {}
    for t in sorted(set(ids)):
        by_suite.setdefault(suite_of(t), []).append(t)
    out: dict = {}
    for suite, tids in by_suite.items():
        total = max(1, len(suite_ids(ctx, suite)))
        if len(tids) / total > WHOLE_SUITE_FRACTION:
            res = run_suite(tree, suite, jobs)
            out.update({t: res[t] for t in tids if t in res})
            continue
        # One filtered invocation per test, `jobs` at a time, each single-job.
        def one(tid: str) -> dict:
            rel = tid.split("/", 1)[1] if "/" in tid else tid
            pat = "*" + "".join(f"[{c}]" if c in "*?[]" else c for c in rel)
            return {k: v for k, v in run_suite(tree, suite, 1, pat).items() if k == tid}
        with cf.ThreadPoolExecutor(jobs) as ex:
            for r in ex.map(one, tids):
                out.update(r)
    for t in ids:
        out.setdefault(t, {"status": "MISSING", "reason": "no result from tests/run.py", "stage": "run"})
    return out


def run_full(ctx: Ctx, tree: Path) -> dict:
    res: dict = {}
    for s in list_suites(ctx, tree):
        t0 = time.time()
        r = run_suite(tree, s, ctx.jobs)
        log(f"full run: suite {s}: {sum(map(is_pass, r.values()))}/{len(r)} pass ({time.time() - t0:.0f}s)")
        res.update(r)
    return res


# ---------------------------------------------------------------- cache

def known(ctx: Ctx) -> dict:
    return read_json(ctx.state / "known.json", {}) or {}


def record(ctx: Ctx, commit: str, results: dict, replace_known: bool = False) -> None:
    """Store results observed at `commit` and fold them into known.json."""
    path = ctx.state / "results" / f"{commit}.json"
    cur = read_json(path, {}) or {}
    cur.update(results)
    write_json(path, cur)
    k = {} if replace_known else known(ctx)
    for t, r in results.items():
        if r.get("status") == "MISSING":
            continue
        k[t] = {"status": r.get("status"), "reason": r.get("reason", ""), "stage": r.get("stage", ""),
                "commit": commit}
    write_json(ctx.state / "known.json", k)


def suite_counts(results: dict) -> dict:
    c: dict = {}
    for t, r in results.items():
        s = c.setdefault(suite_of(t), {"pass": 0, "total": 0})
        s["total"] += 1
        s["pass"] += is_pass(r)
    return c


def sample_ids(ctx: Ctx, n: int, seed: str) -> list[str]:
    """Fixed pseudo-random sample of known-passing tests, stable for a given seed
    (the baseline commit), spread across suites."""
    import hashlib
    passing = sorted(t for t, r in known(ctx).items() if is_pass(r))
    key = lambda t: hashlib.sha1(f"{seed}:{t}".encode()).hexdigest()
    return sorted(passing, key=key)[:n]


def matches(test_id: str, glob: str) -> bool:
    return fnmatch.fnmatchcase(test_id, glob)
