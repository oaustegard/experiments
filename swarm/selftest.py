"""Self-test of the swarm machinery on a throwaway fake repo (no model calls, ~30 s).

    python3 swarm/selftest.py [--keep]

Builds a git repo in a temp dir with a fake tests/run.py (a test passes iff
hcc/_fake_fixes/<suite>__<name> exists) and this swarm/ copied in, then drives
run.py --fake through every gate path and asserts the ledger outcome:
fix -> merged, tests/ edit -> rejected, noop -> failed, break -> rejected by
the gate sample or reverted by the full refresh, two conflicting candidates -> one merged,
two candidates with conflicting doc edits -> both merged.

Federation (swarm/federation.py), with a local bare repo as the transport: the
hub publishes hcc/main + hcc/state; a sibling clone seeds its state from them,
merges two fakes locally and pushes hcc/cand/s1 (with Fixed: trailers); the hub
ingests and merges both; after the hub moves on, the sibling rebases (dropping
what the hub took) and re-stacks its branch fast-forward; a sibling commit that
conflicts with the hub is rejected there, dropped by the sibling's next rebase
and re-dispatched as a REBASE worker.
"""

from __future__ import annotations

if __name__ == "__main__":
    import _boot  # noqa: F401

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from swarm.common import read_json, read_jsonl, write_json

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
           "--wt-root", str(wt), "--poll", "0.3", "--sample", "4", "--idle-max", "5", *args]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=300, check=False)
    return p.stdout + p.stderr


def records(root: Path) -> dict[str, dict]:
    last: dict[str, dict] = {}
    for r in read_jsonl(root / "swarm" / "state" / "ledger.jsonl"):
        last[r["id"]] = r
    return last


def statuses(root: Path) -> dict[str, str]:
    return {k: v["status"] for k, v in records(root).items()}


def federation(tmp: Path, root: Path, wt: Path, expect) -> None:
    """Hub `root` + sibling clone `sib` talking through the bare repo `bare`."""
    bare, sib, swt = tmp / "remote.git", tmp / "sib", tmp / "wt-s1"
    sh(tmp, "git", "init", "-q", "--bare", str(bare))
    F = ("--fed-remote", str(bare))

    def fed(repo: Path, *args: str) -> str:
        p = subprocess.run([sys.executable, str(repo / "swarm" / "run.py"), *args, "--root", str(repo), *F],
                           capture_output=True, text=True, timeout=300, check=False)
        return p.stdout + p.stderr

    def remote(ref: str) -> str:
        return subprocess.run(["git", "rev-parse", "-q", "--verify", ref], cwd=bare, capture_output=True,
                              text=True).stdout.strip()

    def hub_ingest(*extra: str) -> str:
        return run(root, wt, "--federate", "--no-launch", "--fed-poll", "0", "--suites", "smoke,c-testsuite", *F, *extra)

    def sibling(*extra: str) -> str:
        return run(sib, swt, "--sibling", "s1", "--only", "gcc-torture/*", "--suites", "smoke,c-testsuite,gcc-torture",
                   "--full-every", "99", *F, *extra)

    out = fed(root, "fed-publish")
    hub_main = sh(root, "git", "rev-parse", "main").strip()
    state_files = sh(bare, "git", "ls-tree", "--name-only", "-r", "hcc/state") if remote("hcc/state") else ""
    expect("hub publishes hcc/main and hcc/state", remote("hcc/main") == hub_main and "federation/known.json" in state_files,
           out)

    sh(tmp, "git", "clone", "-q", "-b", "hcc/main", str(bare), str(sib))
    sh(sib, "git", "checkout", "-q", "-b", "fed-s1")
    out = sibling("--max-workers", "2", "--concurrency", "2")
    st = statuses(sib)
    tip = remote("hcc/cand/s1")
    msg = sh(sib, "git", "log", "-1", "--format=%B", tip) if tip else ""
    expect("sibling seeds from hcc/state, merges locally, pushes hcc/cand/s1 with Fixed: trailer",
           st.get("s1-w00001") == "merged" and st.get("s1-w00002") == "merged" and "Fixed: gcc-torture/" in msg
           and (sib / "swarm/state/known.json").exists(), out)

    out = hub_ingest()
    hst = statuses(root)
    log = sh(root, "git", "log", "--format=%B", "-5")
    expect("hub ingests both candidates, merges them and pushes hcc/main",
           hst.get("s1-w00001") == "merged" and hst.get("s1-w00002") == "merged" and "Swarm-Task: s1-w00002" in log
           and remote("hcc/main") == sh(root, "git", "rev-parse", "main").strip(), out)

    # Sibling round 2: rebase drops what the hub took; its new conflicting commit is re-stacked (fast-forward).
    old_tip = remote("hcc/cand/s1")
    out = sibling("--max-workers", "1", "--concurrency", "1", "--fake-mode", "conflict")
    new_tip = remote("hcc/cand/s1")
    stack = sh(sib, "git", "rev-list", "--first-parent", "--no-merges", new_tip, "^" + remote("hcc/main")).split()
    ff = subprocess.run(["git", "merge-base", "--is-ancestor", old_tip, new_tip], cwd=bare).returncode == 0
    expect("sibling rebase drops hub-merged commits; hcc/cand/s1 only fast-forwards",
           statuses(sib).get("s1-w00003") == "merged" and len(stack) == 1 and ff, out)

    # The hub merges its own conflicting change first, so the sibling's candidate conflicts there.
    (root / "hcc" / "_fake_conflict.txt").write_text("hub change\n")
    sh(root, "git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qam", "hub: conflicting change")
    out = hub_ingest()
    r = records(root).get("s1-w00003") or {}
    expect("hub: conflicting candidate rejected (once per sha), nothing re-ingested",
           r.get("status") == "rejected" and str(r.get("reason", "")).startswith("conflict")
           and not any(k in ("s1-w00001", "s1-w00002") and v["status"] != "merged" for k, v in records(root).items()),
           out)

    out = sibling("--no-launch")
    r = records(sib).get("s1-w00003") or {}
    known = __import__("json").loads((sib / "swarm/state/known.json").read_text())
    refail = [t for t in r.get("gate_fixed") or [] if known.get(t, {}).get("status") != "PASS"]
    expect("sibling rebase drops the conflicting commit, marks it a conflict, re-measures its tests",
           r.get("status") == "rejected" and str(r.get("reason", "")).startswith("conflict") and refail, out)

    out = sibling("--max-workers", "1", "--concurrency", "1", "--only", ",".join(r.get("tests") or []))
    r4 = records(sib).get("s1-w00004") or {}
    out += hub_ingest()
    expect("conflicted cluster re-dispatched as a REBASE worker, then ingested by the hub",
           r4.get("rebase_of") == "s1-w00003" and statuses(root).get("s1-w00004") == "merged", out)

    out = fed(root, "fed-publish")
    ingest = __import__("json").loads(sh(bare, "git", "show", "hcc/state:federation/ingest.json"))
    expect("hcc/state carries the hub's ingest outcomes",
           ingest.get("s1-w00003", {}).get("status") == "rejected" and ingest.get("s1-w00004", {}).get("status") == "merged",
           out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--keep", action="store_true", help="keep the temp repo for inspection")
    a = ap.parse_args(argv)
    tmp = Path(tempfile.mkdtemp(prefix="swarm-selftest-"))
    root, wt = tmp / "repo", tmp / "wt"
    root.mkdir()
    build_repo(root)
    failures = []
    # the gate's hcc selftest stand-in (the real 710 take minutes): one fake selftest that fails
    # while the tree carries hcc/_fake_selftest_break.txt
    fake_st = tmp / "fake_selftest.py"
    fake_st.write_text("import os\nbad = os.path.exists('hcc/_fake_selftest_break.txt')\n"
                       "print('FAIL fake_selftest.c: broken') if bad else None\n"
                       "print(f'{1 - bad}/1 passed')\n")
    os.environ["SWARM_SELFTEST"] = f"{sys.executable} {fake_st}"

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
    os.environ["SWARM_PYCHECK"] = f"{sys.executable} -m compileall -q"   # the real one needs PyPy
    out = run(root, wt, "--fake-mode", "syntax", "--max-workers", "1", "--concurrency", "1")
    r = records(root).get("w00004") or {}
    ok = r.get("status") == "rejected" and str(r.get("reason", "")).startswith("python 3.11 syntax:")
    print(f"{'ok  ' if ok else 'FAIL'} hcc/ that does not byte-compile -> rejected  {r.get('reason', '')[:80]}")
    if not ok:
        failures.append("pycheck")
        print(out[-2000:])
    out = run(root, wt, "--fake-mode", "noop", "--max-workers", "1", "--concurrency", "1")
    check("noop -> failed", out, {"w00005": "failed"})
    out = run(root, wt, "--fake-mode", "break", "--max-workers", "1", "--concurrency", "1", "--full-every", "1")
    # caught by the gate's sample (rejected) or else by the full refresh (reverted)
    check("break -> rejected by gate or reverted by refresh", out, {"w00006": "rejected|reverted"})
    out = run(root, wt, "--fake-mode", "conflict", "--max-workers", "2", "--concurrency", "2",
              "--batch", "2", "--poll", "2", "--full-every", "99")
    got = statuses(root)
    pair = sorted([got.get("w00007"), got.get("w00008")])
    ok = pair == ["merged", "rejected"]
    print(f"{'ok  ' if ok else 'FAIL'} conflicting pair -> one merged, one rejected  {pair}")
    if not ok:
        failures.append("conflict")
        print(out[-2000:])
    out = run(root, wt, "--fake-mode", "docs", "--max-workers", "2", "--concurrency", "2",
              "--batch", "2", "--poll", "2", "--full-every", "99")
    check("conflicting doc edits -> both merged (code only for one)", out,
          {"w00009": "merged", "w00010": "merged"})
    def expect(name: str, ok: bool, out: str) -> None:
        print(f"{'ok  ' if ok else 'FAIL'} {name}")
        if not ok:
            failures.append(name)
            print(out[-3000:])

    federation(tmp, root, wt, expect)
    log = sh(root, "git", "log", "--format=%an %s")
    if "Oskar Austegard" not in log:
        failures.append("author")
    print(f"{'ok  ' if 'Oskar Austegard' in log else 'FAIL'} merge commits authored by Oskar Austegard")
    # quick refresh: a merge triggers it (--quick-every 1); it must run the quick suites and finish
    out = run(root, wt, "--max-workers", "1", "--concurrency", "1", "--full-every", "99",
              "--quick-every", "1", "--quick-suites", "smoke,c-testsuite")
    ok = "quick refresh at" in out and "quick refresh done" in out
    print(f"{'ok  ' if ok else 'FAIL'} quick refresh runs after a merge and completes")
    if not ok:
        failures.append("quick refresh")
        print(out[-2000:])
    # selftest gate: a candidate that fixes its targets but breaks an hcc selftest is rejected
    before = set(records(root))
    out = run(root, wt, "--fake-mode", "selftest", "--max-workers", "1", "--concurrency", "1", "--full-every", "99")
    new = [r for k, r in records(root).items() if k not in before]
    ok = bool(new) and new[0].get("status") == "rejected" and \
        str(new[0].get("reason", "")).startswith("selftest regressions (1): fake_selftest.c")
    print(f"{'ok  ' if ok else 'FAIL'} a candidate that breaks an hcc selftest -> rejected")
    if not ok:
        failures.append("selftest gate")
        print(out[-2000:])
    # idle refresh: once the queue drains, merges since the last full refresh get one
    cf = root / "swarm" / "state" / "counters.json"
    c = read_json(cf, {}) or {}
    c["merges_since_full"] = 3          # merges the gate checked only against its sample
    write_json(cf, c)
    out = run(root, wt, "--max-workers", "50", "--concurrency", "2", "--full-every", "99")
    ok = "merges since the last full refresh, refreshing" in out
    print(f"{'ok  ' if ok else 'FAIL'} empty queue with unrefreshed merges -> full refresh")
    if not ok:
        failures.append("idle refresh")
        print(out[-2000:])
    print(f"{len(failures)} failures" + (f": {failures}" if failures else ""))
    if a.keep:
        print(f"kept {tmp}")
    else:
        subprocess.run(["git", "worktree", "prune"], cwd=root, check=False)
        shutil.rmtree(tmp, ignore_errors=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
