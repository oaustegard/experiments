#!/usr/bin/env python3
"""Emit the exact subagent prompt for a Stage-1 run, and stage its worktree.

CCotw carries no ANTHROPIC_API_KEY, so scripts do not call models: this prints prompts
and the parent session dispatches them via the Agent tool. That also keeps the prompt
byte-identical across arms, which is the only way the weak/strong comparison means
anything.

Stage 1 has two arms and neither routes: both run every task to completion. The point is
the oracle bound and the disjointness of the two solve sets, not a routing decision.

Usage:
  python3 emit_prompts.py --arm weak --run-root /tmp/stage1/r1 --stage      # copy repos
  python3 emit_prompts.py --arm weak --run-root /tmp/stage1/r1 --manifest   # runs.json
  python3 emit_prompts.py --arm weak --run-root /tmp/stage1/r1 --task parse_range
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TASKS = ROOT / "tasks"

PROMPT = """You are fixing a bug in a small Python repository.

Repository root: {worktree}
Read `{worktree}/issue.md` first. It is the only description of the problem you get.

Procedure:
1. `cd {worktree} && python3 -m pytest tests/ -q` to see the failure.
2. Read the source under `{worktree}/{package}/` and find the cause.
3. Edit the source in place until `python3 -m pytest tests/ -q` passes.

Rules:
- Do NOT modify, add, or delete anything under `{worktree}/tests/`. Edits there are
  detected and score the run as failed.
- Do NOT add dependencies. The standard library is all you have.
- Keep the public API of the `{package}` package unchanged: {exports}.
- The visible tests are a subset of a larger hidden suite that grades this run. Fixing
  the underlying cause scores; special-casing the visible assertions does not.
- Do not create new files outside `{worktree}/{package}/`.

When the visible suite passes, reply with a one-paragraph summary naming the files you
changed and the cause you found. Do not paste the full diff.
"""


def worktree_for(run_root, task):
    return Path(run_root) / task


def stage(run_root):
    n = 0
    for task_dir in sorted(TASKS.iterdir()):
        if not (task_dir / "meta.json").exists():
            continue
        dest = worktree_for(run_root, task_dir.name)
        shutil.rmtree(dest, ignore_errors=True)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(task_dir / "repo", dest)
        shutil.copy(task_dir / "issue.md", dest / "issue.md")
        n += 1
    print(f"staged {n} worktrees under {run_root}")


def prompt_for(run_root, task):
    meta = json.loads((TASKS / task / "meta.json").read_text())
    return PROMPT.format(worktree=worktree_for(run_root, task),
                         package=meta["package"],
                         exports=", ".join(meta["exports"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, help="label for this arm, e.g. weak / strong")
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--task")
    ap.add_argument("--stage", action="store_true")
    ap.add_argument("--manifest", action="store_true")
    args = ap.parse_args()

    root = Path(args.run_root) / args.arm
    tasks = sorted(d.name for d in TASKS.iterdir() if (d / "meta.json").exists())

    if args.stage:
        stage(root)
        return 0
    if args.manifest:
        print(json.dumps({args.arm: {t: str(worktree_for(root, t)) for t in tasks}}, indent=2))
        return 0
    if args.task:
        print(prompt_for(root, args.task))
        return 0
    for t in tasks:
        print(f"===== {args.arm} :: {t}\n{prompt_for(root, t)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
