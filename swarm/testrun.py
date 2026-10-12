"""Adapter around tests/run.py plus the per-commit results cache.

tests/run.py contract (owned by tests/, not by the swarm):
    python3 tests/run.py --suite NAME[,NAME] [--jobs N] [--filter GLOB]... [--json OUT] [--quiet]
    --filter is a repeatable glob on the test id; tests in a suite's skip.txt are not run.
    --json OUT writes {test_id: {status, reason, stage}}; test_id = "<suite>/<relpath>";
    status in PASS | FAIL | TIMEOUT | CRASH (anything else is treated as not passing).

Results cache (all under swarm/state/):
    results/<commit>.json   every result recorded at that main commit (partial or full)
    known.json              {test_id: {status, reason, stage, commit}}: latest known
                            status of each test along main; the gate compares to this
    baseline.json           {commit, time, suites: {name: {pass, total}}}: last full run
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import time
from pathlib import Path

from swarm.common import Ctx, log, read_json, write_json

TEST_TIMEOUT = 3600   # whole tests/run.py invocation


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


def _invoke(tree: Path, suites: list[str], jobs: int, filters: list[str]) -> dict:
    """One tests/run.py invocation in `tree`; returns {test_id: rec} ({} on failure)."""
    with tempfile.TemporaryDirectory(prefix="swarm-tr-") as td:
        out = Path(td) / "r.json"
        cmd = ["python3", "tests/run.py", "--suite", ",".join(suites), "--jobs", str(jobs),
               "--json", str(out), "--quiet"]
        for f in filters:
            cmd += ["--filter", f]
        try:
            p = subprocess.run(cmd, cwd=tree, env=_env(tree), capture_output=True, text=True,
                               timeout=TEST_TIMEOUT, check=False)
        except subprocess.TimeoutExpired:
            log(f"tests/run.py --suite {','.join(suites)} timed out in {tree}")
            return {}
        res = read_json(out, None)
        if res is None:
            log(f"tests/run.py wrote no JSON (rc={p.returncode}): {p.stderr[-300:]}")
            return {}
        return res


def run_suite(tree: Path, suite: str, jobs: int) -> dict:
    return _invoke(tree, [suite], jobs, [])


def glob_escape(test_id: str) -> str:
    return "".join(f"[{c}]" if c in "*?[]" else c for c in test_id)


def run_ids(ctx: Ctx, tree: Path, ids: list[str], jobs: int | None = None) -> dict:
    """Run exactly these test ids in `tree` (one invocation, one --filter per id).
    Ids with no result come back as status MISSING."""
    ids = sorted(set(ids))
    out: dict = {}
    for i in range(0, len(ids), 400):         # keep argv a sane length
        chunk = ids[i:i + 400]
        suites = sorted({suite_of(t) for t in chunk})
        res = _invoke(tree, suites, jobs or ctx.jobs, [glob_escape(t) for t in chunk])
        out.update({t: res[t] for t in chunk if t in res})
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
