"""Self-test of the swarm machinery on a throwaway fake repo (no model calls, ~30 s).

    python3 swarm/selftest.py [--keep]

Builds a git repo in a temp dir with a fake tests/run.py (a test passes iff
hcc/_fake_fixes/<suite>__<name> exists) and this swarm/ copied in, then drives
run.py --fake through every gate path and asserts the ledger outcome:
fix -> merged, tests/ edit -> rejected, noop -> failed, break -> rejected by
the gate sample or reverted by the full refresh, two conflicting candidates -> one merged,
two candidates with conflicting doc edits -> both merged.
"""

from __future__ import annotations

if __name__ == "__main__":
    import _boot  # noqa: F401

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from swarm.common import read_jsonl

HERE = Path(__file__).resolve().parent

FAKE_RUN = '''"""Fake tests/run.py: a test passes iff hcc/_fake_fixes/<suite>__<name> exists."""
import argparse, fnmatch, json
from pathlib import Path
ap = argparse.ArgumentParser()
for f in ("--suite", "--jobs", "--json", "--cc", "--baseline"): ap.add_argument(f)
ap.add_argument("--filter", action="append"); ap.add_argument("--quiet", action="store_true")
ap.add_argument("-v", action="store_true")
a = ap.parse_args()
root = Path(__file__).resolve().parent.parent
res = {}
for suite in (a.suite or "smoke").split(","):
    for f in sorted((root / "tests/suites" / suite).iterdir()):
        tid = f"{suite}/{f.name}"
        if a.filter and not any(fnmatch.fnmatch(tid, x) for x in a.filter): continue
        ok = (root / "hcc/_fake_fixes" / f"{suite}__{f.name}").exists()
        n = int("".join(c for c in f.name if c.isdigit()))
        reason = ["unexpected token 'foo%d' at %s:3:%d" % (n, f.name, n),
                  "unsupported node BinOp line %d" % n, "exit code %d expected 0" % n][n % 3]
        res[tid] = ({"status": "PASS", "reason": "", "stage": ""} if ok else
                    {"status": "FAIL", "reason": reason, "stage": ["parse", "codegen", "run"][n % 3]})
if a.json: Path(a.json).write_text(json.dumps(res))
'''


def sh(cwd: Path, *cmd: str) -> str:
    return subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True).stdout


def build_repo(root: Path) -> None:
    for suite, n in (("smoke", 4), ("c-testsuite", 12), ("gcc-torture", 18)):
        d = root / "tests" / "suites" / suite
        d.mkdir(parents=True)
        for i in range(1, n + 1):
            (d / f"{suite[0]}{i}.c").write_text("int main(){return 0;}\n")
    fixes = root / "hcc" / "_fake_fixes"
    fixes.mkdir(parents=True)
    passing = [f"smoke__s{i}.c" for i in range(1, 5)] + [f"c-testsuite__c{i}.c" for i in range(1, 7)]
    for name in passing:
        (fixes / name).write_text("fixed\n")
    (root / "tests" / "run.py").write_text(FAKE_RUN)
    (root / "docs").mkdir()
    (root / "docs" / "CONTRACT.md").write_text("# fake\n")
    shutil.copytree(HERE, root / "swarm", ignore=shutil.ignore_patterns("state", "rejected", "__pycache__"))
    sh(root, "git", "init", "-q", "-b", "main")
    sh(root, "git", "add", "-A")
    sh(root, "git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init")


def run(root: Path, wt: Path, *args: str) -> str:
    cmd = [sys.executable, str(root / "swarm" / "run.py"), "run", "--fake", "--root", str(root),
           "--wt-root", str(wt), "--poll", "0.3", "--sample", "4", *args]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=300, check=False)
    return p.stdout + p.stderr


def statuses(root: Path) -> dict[str, str]:
    last: dict[str, dict] = {}
    for r in read_jsonl(root / "swarm" / "state" / "ledger.jsonl"):
        last[r["id"]] = r
    return {k: v["status"] for k, v in last.items()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--keep", action="store_true", help="keep the temp repo for inspection")
    a = ap.parse_args(argv)
    tmp = Path(tempfile.mkdtemp(prefix="swarm-selftest-"))
    root, wt = tmp / "repo", tmp / "wt"
    root.mkdir()
    build_repo(root)
    failures = []

    def check(name: str, out: str, want: dict[str, str]) -> None:
        got = statuses(root)
        bad = {k: (got.get(k), v) for k, v in want.items() if got.get(k) not in v.split("|")}
        print(f"{'ok  ' if not bad else 'FAIL'} {name}" + (f"  (got, want): {bad}" if bad else ""))
        if bad:
            failures.append(name)
            print(out[-2000:])

    out = run(root, wt, "--max-workers", "2", "--concurrency", "2", "--full-every", "99")
    check("fix -> merged", out, {"w00001": "merged", "w00002": "merged"})
    out = run(root, wt, "--fake-mode", "tests", "--max-workers", "1", "--concurrency", "1")
    check("tests/ edit -> rejected", out, {"w00003": "rejected"})
    out = run(root, wt, "--fake-mode", "noop", "--max-workers", "1", "--concurrency", "1")
    check("noop -> failed", out, {"w00004": "failed"})
    out = run(root, wt, "--fake-mode", "break", "--max-workers", "1", "--concurrency", "1", "--full-every", "1")
    # caught by the gate's sample (rejected) or else by the full refresh (reverted)
    check("break -> rejected by gate or reverted by refresh", out, {"w00005": "rejected|reverted"})
    out = run(root, wt, "--fake-mode", "conflict", "--max-workers", "2", "--concurrency", "2",
              "--batch", "2", "--poll", "2", "--full-every", "99")
    got = statuses(root)
    pair = sorted([got.get("w00006"), got.get("w00007")])
    ok = pair == ["merged", "rejected"]
    print(f"{'ok  ' if ok else 'FAIL'} conflicting pair -> one merged, one rejected  {pair}")
    if not ok:
        failures.append("conflict")
        print(out[-2000:])
    out = run(root, wt, "--fake-mode", "docs", "--max-workers", "2", "--concurrency", "2",
              "--batch", "2", "--poll", "2", "--full-every", "99")
    check("conflicting doc edits -> both merged (code only for one)", out,
          {"w00008": "merged", "w00009": "merged"})
    log = sh(root, "git", "log", "--format=%an %s")
    if "Oskar Austegard" not in log:
        failures.append("author")
    print(f"{'ok  ' if 'Oskar Austegard' in log else 'FAIL'} merge commits authored by Oskar Austegard")
    print(f"{len(failures)} failures" + (f": {failures}" if failures else ""))
    if a.keep:
        print(f"kept {tmp}")
    else:
        subprocess.run(["git", "worktree", "prune"], cwd=root, check=False)
        shutil.rmtree(tmp, ignore_errors=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
