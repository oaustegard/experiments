"""Shared plumbing for the swarm: paths, git helpers, JSON state files, ledger.

Everything takes a `Ctx` (paths + settings) instead of reading globals, so the
whole swarm can be pointed at a scratch repo for testing (`--root`).
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_ROOT = Path(__file__).resolve().parent.parent
AUTHOR_NAME = "Oskar Austegard"
AUTHOR_EMAIL = "oskar@austegard.com"
FANOUT = Path("/mnt/skills/user/dispatching-subagents/scripts/headless_fanout.py")


@dataclass
class Ctx:
    root: Path = DEFAULT_ROOT                     # main checkout (owns the branch)
    wt_root: Path = Path("/home/user/haiku-cc-wt")  # one worktree per worker + _gate
    branch: str = ""                              # detected from root when empty
    jobs: int = 4                                 # test parallelism for gate / baseline
    suites: list[str] = field(default_factory=list)  # [] = every dir in tests/suites
    sample_size: int = 150
    full_every: int = 25                          # full-suite refresh every N merges
    batch: int = 4                                # candidates gated together

    @property
    def swarm(self) -> Path:
        return self.root / "swarm"

    @property
    def state(self) -> Path:
        p = self.swarm / "state"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def rejected(self) -> Path:
        p = self.swarm / "rejected"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def gate_tree(self) -> Path:
        return self.wt_root / "_gate"

    def worktree(self, task_id: str) -> Path:
        return self.wt_root / task_id

    def task_dir(self, task_id: str) -> Path:
        p = self.state / "tasks" / task_id
        p.mkdir(parents=True, exist_ok=True)
        return p


# ---------------------------------------------------------------- git

class GitError(RuntimeError):
    pass


def git(cwd: Path, *args: str, check: bool = True, input: str | None = None) -> str:
    """Run git in `cwd`; retries briefly on index.lock contention."""
    env = dict(os.environ, GIT_AUTHOR_NAME=AUTHOR_NAME, GIT_AUTHOR_EMAIL=AUTHOR_EMAIL,
               GIT_COMMITTER_NAME=AUTHOR_NAME, GIT_COMMITTER_EMAIL=AUTHOR_EMAIL)
    for attempt in range(20):
        p = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True,
                           input=input, env=env, check=False)
        if p.returncode != 0 and "index.lock" in p.stderr and attempt < 19:
            time.sleep(0.5 + attempt * 0.25)
            continue
        if check and p.returncode != 0:
            raise GitError(f"git {' '.join(args)} (in {cwd}): {p.stderr.strip()[:500]}")
        return p.stdout
    raise AssertionError("unreachable")


def head(ctx: Ctx) -> str:
    return git(ctx.root, "rev-parse", f"refs/heads/{branch(ctx)}").strip()


def branch(ctx: Ctx) -> str:
    if not ctx.branch:
        ctx.branch = git(ctx.root, "symbolic-ref", "--short", "HEAD").strip()
    return ctx.branch


def add_worktree(ctx: Ctx, path: Path, commit: str) -> None:
    """Detached worktree at `commit`; untracked test-suite dirs are symlinked in."""
    if path.exists():
        remove_worktree(ctx, path)
    path.parent.mkdir(parents=True, exist_ok=True)
    git(ctx.root, "worktree", "add", "--detach", "--force", str(path), commit)
    link_untracked_suites(ctx, path)


def link_untracked_suites(ctx: Ctx, tree: Path) -> None:
    """Suites/tools that are gitignored in main (downloaded corpora) are symlinked."""
    for rel in ("tests/suites", "tests/tools"):
        src = ctx.root / rel
        if not src.is_dir():
            continue
        for d in src.iterdir():
            dst = tree / rel / d.name
            if not dst.exists() and not dst.is_symlink():
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.symlink_to(d)


def remove_worktree(ctx: Ctx, path: Path) -> None:
    git(ctx.root, "worktree", "remove", "--force", str(path), check=False)
    if path.exists():
        subprocess.run(["rm", "-rf", str(path)], check=False)
    git(ctx.root, "worktree", "prune", check=False)


# ---------------------------------------------------------------- JSON state

def read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return default


def write_json(path: Path, value) -> None:
    """Atomic write (tmp + rename) so an interrupted run never leaves half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=1, sort_keys=True))
    os.replace(tmp, path)


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass  # a torn last line after a crash
    return out


# ---------------------------------------------------------------- ledger
# state/ledger.jsonl is append-only; each line is a full snapshot of one
# worker's record. The last line per id wins.
# Record fields: id, key, signature, tests, base, status, cost_usd, result,
# merged_commit, reason, t_launch, t_done.
# status: launched -> finished -> merged | rejected | failed ; merged -> reverted

def ledger(ctx: Ctx) -> dict[str, dict]:
    recs: dict[str, dict] = {}
    for r in read_jsonl(ctx.state / "ledger.jsonl"):
        recs[r["id"]] = r
    return recs


def ledger_put(ctx: Ctx, rec: dict) -> dict:
    append_jsonl(ctx.state / "ledger.jsonl", rec)
    return rec


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
