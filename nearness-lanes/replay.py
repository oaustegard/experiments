#!/usr/bin/env python3
"""Replay hcc/main as a concurrent swarm and score nearness predictors against real git conflicts.

For each code commit c and window W, pretend c was developed on start = the commit W-1 steps
before c^, while the W-1 commits in between landed on main. Ground truth: does git's 3-way
merge of c's change onto start conflict (`git merge-tree --merge-base=c^ start c`)? That is the
same textual question the gate asks when a stale candidate meets a moved main.

Predictors compare c's change (c^ -> c) with the window's change (c^ -> start, i.e. what moved),
both in c^ line coordinates:
  file      any file both touched
  hunk:k    a changed line of c within k lines of a line the window changed, same file
  func      both changed lines inside the same top-level def/class of the c^ file (Python only)

    python3 replay.py --repo <experiments checkout> --ref origin/hcc/main --out replay.json
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
from functools import lru_cache

HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def git(repo: str, *a: str, ok=(0,)) -> tuple[int, str]:
    r = subprocess.run(["git", "-C", repo, *a], capture_output=True, text=True, errors="replace")
    if r.returncode not in ok:
        raise RuntimeError(f"git {' '.join(a)}: {r.stderr.strip()}")
    return r.returncode, r.stdout


def old_lines(repo: str, a: str, b: str, paths: list[str] | None = None) -> dict[str, set[int]]:
    """{file: lines of `a` that the diff a->b changes}; a pure insertion marks the line it follows."""
    _, out = git(repo, "diff", "-U0", "--no-renames", a, b, "--", *(paths or []))
    res: dict[str, set[int]] = {}
    f = None
    for line in out.splitlines():
        if line.startswith("--- "):
            f = line[6:] if line.startswith("--- a/") else None
        elif line.startswith("+++ ") and f is None:
            f = line[6:] if line.startswith("+++ b/") else None  # new file: nothing to collide with in a
            if f:
                res.setdefault(f, set()).add(0)
        elif (m := HUNK.match(line)) and f:
            start, n = int(m[1]), int(m[2] if m[2] is not None else 1)
            res.setdefault(f, set()).update(range(start, start + n) if n else {start})
    return res


@lru_cache(maxsize=4096)
def spans(repo: str, commit: str, path: str) -> tuple[tuple[int, int, str], ...]:
    """Top-level def/class spans of `path` at `commit`, plus methods as Class.name."""
    if not path.endswith(".py"):
        return ()
    rc, src = git(repo, "show", f"{commit}:{path}", ok=(0, 128))
    if rc:
        return ()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return ()
    out = []
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.append((n.lineno - len(n.decorator_list), n.end_lineno, n.name))
            if isinstance(n, ast.ClassDef):
                out += [(m.lineno, m.end_lineno, f"{n.name}.{m.name}") for m in n.body
                        if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))]
    return tuple(out)


def units(repo: str, commit: str, path: str, lines: set[int]) -> set[str]:
    """Innermost def/class names covering `lines`; '<module>' for module-level lines."""
    sp = spans(repo, commit, path)
    res = set()
    for ln in lines:
        inner = [s for s in sp if s[0] <= ln <= s[1]]
        res.add(min(inner, key=lambda s: s[1] - s[0])[2] if inner else "<module>")
    return res


def predictors(repo: str, base: str, mine: dict, theirs: dict) -> dict:
    shared = set(mine) & set(theirs)
    def gap(f):
        a, b = sorted(mine[f]), sorted(theirs[f])
        return min(abs(x - y) for x in a for y in b)
    gaps = [gap(f) for f in shared]
    func = any(units(repo, base, f, mine[f]) & units(repo, base, f, theirs[f]) for f in shared if f.endswith(".py"))
    return {"file": bool(shared), "gap": min(gaps) if gaps else None, "func": func,
            "files": sorted(shared)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--ref", default="origin/hcc/main")
    ap.add_argument("--path", default="hcc")
    ap.add_argument("--windows", default="2,4,8,16,32")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    _, out = git(a.repo, "rev-list", "--first-parent", "--reverse", a.ref)
    chain = out.split()
    _, out = git(a.repo, "rev-list", "--reverse", a.ref, "--", a.path)
    code = set(out.split())
    idx = {c: i for i, c in enumerate(chain)}
    rows = []
    for c in chain:
        if c not in code or idx[c] == 0:
            continue
        parent = chain[idx[c] - 1]
        mine = old_lines(a.repo, parent, c)
        for w in map(int, a.windows.split(",")):
            j = idx[c] - w
            if j < 0:
                continue
            start = chain[j]
            theirs = old_lines(a.repo, parent, start, sorted(mine))  # what moved under c's files, in c^ coordinates
            rc, _ = git(a.repo, "merge-tree", "--write-tree", f"--merge-base={parent}", start, c, ok=(0, 1))
            rows.append({"commit": c[:10], "w": w, "conflict": rc == 1,
                         "nfiles": len(mine), **predictors(a.repo, parent, mine, theirs)})
    json.dump(rows, open(a.out, "w"), indent=0)
    print(f"{len(rows)} rows -> {a.out}")


if __name__ == "__main__":
    main()
