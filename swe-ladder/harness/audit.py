"""Audit a run's transcripts for cross-checkout contamination and tool misuse.

    python harness/audit.py RUN [--tasks-dir DIR]

Reports, per task, any `git stash` use (worktrees of one mirror share refs/stash,
so a stash pop in one checkout can apply another task's changes) and any
`git commit`/`checkout`/`reset` that could move HEAD, plus the mirrors' current
stash lists.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, MIRRORS, git  # noqa: E402
from cost import default_tasks_dir  # noqa: E402

RISKY = re.compile(r"\bgit\b[^\n|;&]*\b(stash|commit|reset --hard|checkout -b|switch|rebase|merge)\b")


def commands(path: Path):
    for line in open(path, errors="replace"):
        try:
            d = json.loads(line)
        except ValueError:
            continue
        m = d.get("message")
        if isinstance(m, dict) and m.get("role") == "assistant" and isinstance(m.get("content"), list):
            for b in m["content"]:
                if isinstance(b, dict) and b.get("type") == "tool_use" and b.get("name") == "Bash":
                    yield b.get("input", {}).get("command", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--tasks-dir", type=Path, default=None)
    a = ap.parse_args()
    src = a.tasks_dir or default_tasks_dir()
    marker = f"/runs/{a.run}/prompts/"
    hits = {}
    for t in src.glob("*.output"):
        if not t.exists():   # dangling symlink to a finished background command
            continue
        head = t.open(errors="replace").read(4000)
        i = head.find(marker)
        if i < 0:
            continue
        iid = head[i + len(marker):].split(".md", 1)[0]
        risky = [c for c in commands(t) if RISKY.search(c)]
        if risky:
            hits[iid] = [c[:160] for c in risky]
    for iid, cs in sorted(hits.items()):
        print(iid)
        for c in cs:
            print("   ", c.replace("\n", " "))
    for m in sorted(MIRRORS.glob("*.git")):
        out = git("--git-dir", str(m), "stash", "list", check=False).strip()
        print(f"{m.name} stash: {out or '(empty)'}")
    (DATA / "runs" / a.run / "audit.json").write_text(json.dumps(hits, indent=1))


if __name__ == "__main__":
    main()
