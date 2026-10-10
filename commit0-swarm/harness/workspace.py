"""Checkouts and test runs: the pieces the grader, certifier and agents share.

A checkout is a `git clone --shared` of the bare mirror (not a worktree: worktrees
of one mirror share refs/stash, which contaminated 30 concurrent swe-ladder tasks),
with .venv pointing at the library's env and a .c0-task.json marker.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ENVS, TASK_MARKER, git, jailed, mirror, test_argv  # noqa: E402

TEST_TIMEOUT = 1800


def checkout(task: dict, dest: Path, commit: str | None = None, marker: dict | None = None,
             history: bool = True) -> Path:
    """history=False is for agent trees: the commit's files in a fresh one-commit repo.

    Commit0's stub commit is a child of the reference commit in all 16 lite repos,
    so in a clone `git diff HEAD~1` prints the original library. The pilot's agents
    never ran it (audited), but nothing stopped them."""
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if history:
        git("clone", "-q", "--shared", "--no-checkout", str(mirror(task["name"])), str(dest))
        git("checkout", "-q", commit or task["base_commit"], cwd=dest)
    else:
        dest.mkdir()
        arch = subprocess.run(["git", "--git-dir", str(mirror(task["name"])), "archive",
                               commit or task["base_commit"]], stdout=subprocess.PIPE, check=True)
        subprocess.run(["tar", "-x", "-C", str(dest)], input=arch.stdout, check=True)
        git("init", "-q", cwd=dest)
    git("config", "user.email", "c0@harness", cwd=dest)
    git("config", "user.name", "c0", cwd=dest)
    if not history:
        git("add", "-A", cwd=dest)
        git("commit", "-q", "-m", "skeleton", cwd=dest)
    # A --shared clone does not inherit the mirror's info/exclude.
    (dest / ".git" / "info").mkdir(parents=True, exist_ok=True)
    (dest / ".git" / "info" / "exclude").write_text(
        ".venv\n.c0-task.json\n.c0-junit.xml\n.claims/\n__pycache__/\n*.pyc\n.pytest_cache/\n")
    (dest / ".venv").symlink_to(ENVS / task["name"])
    (dest / TASK_MARKER).write_text(json.dumps(marker or {"name": task["name"]}) + "\n")
    babel_data(task, dest)
    opener(dest)
    return dest


def opener(path: Path) -> None:
    """The jail runs as nobody; root-owned agents edit. Both need the tree."""
    subprocess.run(["chmod", "-R", "a+rwX", str(path)], check=True)
    for p in (path, *path.parents):
        if p == Path("/"):
            break
        p.chmod(p.stat().st_mode | 0o055)


BABEL_DATA = ENVS / "_babel-locale-data"


def babel_data(task: dict, dest: Path) -> None:
    """Babel's tests need CLDR data the repo builds with a networked script. The
    PyPI wheel ships the same .dat files; copy them in (data, not code)."""
    if task["name"] != "babel":
        return
    if not (BABEL_DATA / "global.dat").exists():
        raise FileNotFoundError(f"{BABEL_DATA} missing: run harness/fetch_babel_data.py")
    shutil.copytree(BABEL_DATA / "locale-data", dest / "babel" / "locale-data", dirs_exist_ok=True)
    shutil.copy2(BABEL_DATA / "global.dat", dest / "babel" / "global.dat")


def overlay_src(task: dict, src_tree: Path, dest: Path) -> list[str]:
    """Copy src_dir from src_tree onto dest. Returns files changed outside src_dir
    in src_tree relative to base (reported, never copied)."""
    src = task["src_dir"]
    a, b = src_tree / src, dest / src
    if b.exists():
        shutil.rmtree(b)
    shutil.copytree(a, b, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"), symlinks=True)
    babel_data(task, dest)
    out = git("status", "--porcelain", "--untracked-files=all", cwd=src_tree)
    return [l[3:] for l in out.splitlines() if l[3:].strip() and not l[3:].startswith(src)
            and not l[3:].startswith(("babel/locale-data", "babel/global.dat"))]


def run_tests(task: dict, tree: Path, junit_name: str = ".c0-junit.xml") -> dict:
    """Run the task's suite in the jail; per-test outcomes from junit xml."""
    junit = tree / junit_name
    if junit.exists():
        junit.unlink()
    opener(tree)
    p = subprocess.run(jailed(task, tree, test_argv(task, [], junit=str(junit)), TEST_TIMEOUT),
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
    outcomes = parse_junit(junit) if junit.exists() else None
    return {"exit": p.returncode, "tail": p.stdout[-3000:], "outcomes": outcomes}


def parse_junit(path: Path) -> dict[str, str] | None:
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError:
        return None
    out = {}
    for tc in root.iter("testcase"):
        key = f"{tc.get('classname', '')}::{tc.get('name', '')}"
        tags = {c.tag for c in tc}
        if "failure" in tags or "error" in tags:
            out[key] = "fail"
        elif "skipped" in tags:
            out[key] = "skip"
        else:
            out[key] = "pass"
    return out
