"""Task checkouts: one git worktree per (run, instance) off the bare mirror.

A checkout is root-owned and world-writable so the jail's uid can write caches
and test output while root-side tools (git, the agents' Edit) need no
safe.directory exemption. `.venv` links to the shared dependency env.
"""
from __future__ import annotations

import fcntl
import json
import shutil
import subprocess
from contextlib import contextmanager
from pathlib import Path

from common import ENVS, TASK_MARKER, WORK, env_key, git, mirror


@contextmanager
def _locked(repo: str):
    lock = mirror(repo).with_suffix(".lock")
    with open(lock, "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def checkout(task: dict, run: str, name: str | None = None) -> Path:
    """Fresh worktree of task's base commit at WORK/run/name (default: instance id)."""
    path = WORK / run / (name or task["instance_id"])
    remove(task, path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.parent.chmod(0o755)
    path.parent.parent.chmod(0o755)
    with _locked(task["repo"]):
        git("--git-dir", str(mirror(task["repo"])), "worktree", "add", "-q", "--detach",
            str(path), task["base_commit"])
    (path / ".venv").symlink_to(ENVS / env_key(task))
    (path / TASK_MARKER).write_text(json.dumps({
        "instance_id": task["instance_id"], "repo": task["repo"],
        "version": task["version"], "base_commit": task["base_commit"]}))
    subprocess.run(["chmod", "-R", "a+rwX", str(path)], check=True)
    return path


def remove(task: dict, path: Path) -> None:
    if not path.exists():
        return
    # Deleting 64 MB of files is the slow part and needs no lock; only the
    # prune touches the mirror's shared worktree list.
    shutil.rmtree(path)
    with _locked(task["repo"]):
        git("--git-dir", str(mirror(task["repo"])), "worktree", "prune", check=False)


def diff(path: Path, base: str) -> str:
    """Everything the checkout changed against base, new files included."""
    git("add", "-A", cwd=path)
    return git("diff", "--cached", "--binary", base, cwd=path)


def apply(path: Path, patch: str) -> tuple[bool, str]:
    if not patch.strip():
        return True, "empty patch"
    for cmd in (["git", "apply", "--binary", "-"],
                ["git", "apply", "--binary", "--reject", "--whitespace=fix", "-"],
                ["patch", "-p1", "--batch", "--fuzz=5"]):
        p = subprocess.run(cmd, cwd=path, input=patch, text=True,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if p.returncode == 0:
            return True, " ".join(cmd[:2])
        git("checkout", "-q", "--", ".", cwd=path, check=False)
        git("clean", "-fdq", "-e", ".venv", "-e", TASK_MARKER, cwd=path, check=False)
    return False, p.stdout[-2000:]
