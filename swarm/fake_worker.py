"""Stand-in for headless_fanout.py + a Haiku child, for testing the machinery (--fake).

    python3 swarm/fake_worker.py TASK.jsonl --work DIR --cwd WORKTREE [--mode fix|noop|break|tests]

It writes what the real runner would: DIR/ledger.jsonl (cost 0) and the task's
out JSON. Modes (default from env SWARM_FAKE_MODE, else "fix"):
  fix    create hcc/_fake_fixes/<test name> for each representative test; a fake
         tests/run.py can treat that file as "this test now passes"
  noop   change nothing, report failed
  break  like fix, but also delete one existing fix marker (a regression)
  tests  edit a file under tests/ (must be rejected by the gate)
  conflict  like fix, plus write hcc/_fake_conflict.txt (two of these conflict)
  docs   like fix, plus write docs/STATUS.md (two of these conflict on docs only)
  syntax like fix, plus a .py under hcc/ that does not compile
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tasks", type=Path)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--cwd", type=Path, required=True)
    ap.add_argument("--mode", default=os.environ.get("SWARM_FAKE_MODE", "fix"))
    a, _ = ap.parse_known_args(argv)
    task = json.loads(a.tasks.read_text().splitlines()[0])
    tests = re.findall(r"^  - (\S+)", task["prompt"], re.M)
    mode = a.mode
    m = re.search(r"\[fake:(\w+)\]", task["prompt"])   # per-task override for demos
    if m:
        mode = m.group(1)
    time.sleep(0.5)
    fixdir = a.cwd / "hcc" / "_fake_fixes"
    changed = []
    if mode == "conflict":
        (a.cwd / "hcc" / "_fake_conflict.txt").write_text(task["id"] + "\n")
        changed.append("hcc/_fake_conflict.txt")
    if mode == "docs":
        (a.cwd / "docs").mkdir(exist_ok=True)
        (a.cwd / "docs" / "STATUS.md").write_text(f"done by {task['id']}\n")
        changed.append("docs/STATUS.md")
    if mode in ("fix", "break", "conflict", "docs", "syntax"):
        fixdir.mkdir(parents=True, exist_ok=True)
        for t in tests:
            p = fixdir / t.replace("/", "__")
            p.write_text("fixed\n")
            changed.append(str(p.relative_to(a.cwd)))
    if mode == "break":
        others = sorted(x for x in fixdir.iterdir() if str(x.relative_to(a.cwd)) not in changed)
        if others:
            others[0].unlink()
            changed.append(str(others[0].relative_to(a.cwd)))
    if mode == "syntax":   # a fix that does not byte-compile (the gate's pycheck must reject it)
        (a.cwd / "hcc" / "_fake_bad.py").write_text("def broken(:\n    pass\n")
        changed.append("hcc/_fake_bad.py")
    if mode == "tests":
        p = a.cwd / "tests" / "HACK.txt"
        p.write_text("cheating\n")
        changed.append("tests/HACK.txt")
    status = "failed" if mode == "noop" else "fixed"
    out = {"status": status, "test_ids_fixed": tests if status == "fixed" else [],
           "files_changed": changed, "root_cause": f"fake {mode}: {', '.join(tests)}"[:72],
           "notes": f"fake worker mode={mode}"}
    Path(task["out"]).parent.mkdir(parents=True, exist_ok=True)
    Path(task["out"]).write_text(json.dumps(out))
    a.work.mkdir(parents=True, exist_ok=True)
    (a.work / "results").mkdir(exist_ok=True)
    (a.work / "results" / f"{task['id']}.json").write_text(json.dumps({"total_cost_usd": 0.0, "result": "fake"}))
    with (a.work / "ledger.jsonl").open("a") as f:
        f.write(json.dumps({"id": task["id"], "attempt": 1, "ok": True, "reason": None, "cost": 0.0,
                            "secs": 0.5, "turns": 1, "denials": 0}) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
