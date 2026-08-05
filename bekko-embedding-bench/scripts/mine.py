"""Mine mini-CTXBench instances from a scikit-learn clone.

Deterministic and API-free: scikit-learn squash-merges, so each PR is one
commit on main whose subject ends in ``(#NNNNN)`` and whose body may say
``Fixes #MMMMM``. The commit's diff file set IS the PR diff file set (the gold),
and no GitHub API call is needed to get it — which matters here because the
CCotw agent proxy 403s api.github.com for out-of-scope repos.

Issue *bodies* still come from GitHub, via ``mcp__github__search_issues``
(global, unscoped), fetched separately and cached to instances.json.

Usage:
    python3 mine.py --limit 600 > candidates.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

REPO = Path(os.environ.get("BEKKO_BENCH_REPO", "/home/user/sklearn-bench"))

# "ENH add foo (#12345)" -> PR number
PR_RE = re.compile(r"\(#(\d{4,6})\)\s*$")
# "Fixes #123", "closes #123", "fix #123"
FIX_RE = re.compile(r"\b(?:fix(?:e[sd])?|close[sd]?|resolve[sd]?)\s+#(\d{3,6})\b", re.I)

# Code-shaped tokens an `rg` baseline could plausibly extract from issue text.
IDENT_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\b")
CAMEL_RE = re.compile(r"\b[A-Z][a-z]+[A-Z][A-Za-z0-9]*\b")
SNAKE_RE = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b")
BACKTICK_RE = re.compile(r"`([^`\n]{2,60})`")


def git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True
    ).stdout


def mine(limit: int) -> list[dict]:
    """Return merged-PR records that reference a fixed issue, newest first."""
    sep = "\x1e"
    log = git(
        "log",
        f"-n{limit * 4}",
        "--first-parent",
        f"--format={sep}%H%x1f%s%x1f%b",
    )
    out = []
    for rec in log.split(sep):
        if not rec.strip():
            continue
        parts = rec.split("\x1f")
        if len(parts) < 2:
            continue
        sha, subject, body = parts[0].strip(), parts[1], (parts[2] if len(parts) > 2 else "")
        m = PR_RE.search(subject)
        if not m:
            continue
        fix = FIX_RE.search(subject + "\n" + body)
        if not fix:
            continue
        files = [f for f in git("show", "--name-only", "--format=", sha).split("\n") if f.strip()]
        # Gold = source files touched. Drop changelog fragments: they are
        # per-PR files named after the PR itself, so any method that knows the
        # PR number scores them trivially.
        gold = [
            f
            for f in files
            if not f.startswith("doc/whats_new")
            and "/changelog.d/" not in f
            and not f.endswith(".rst.template")
        ]
        if not gold:
            continue
        out.append(
            {
                "pr": int(m.group(1)),
                "issue": int(fix.group(1)),
                "sha": sha,
                "subject": subject,
                "gold": gold,
                "n_gold": len(gold),
            }
        )
        if len(out) >= limit:
            break
    return out


def identifiers(text: str) -> list[str]:
    """Code-shaped tokens an rg baseline would extract from an issue."""
    toks: list[str] = []
    for m in BACKTICK_RE.findall(text):
        toks += IDENT_RE.findall(m)
    toks += CAMEL_RE.findall(text)
    toks += SNAKE_RE.findall(text)
    seen, out = set(), []
    for t in toks:
        if t.lower() in seen or len(t) < 4:
            continue
        seen.add(t.lower())
        out.append(t)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=600)
    a = ap.parse_args()
    recs = mine(a.limit)
    print(json.dumps(recs, indent=1))
