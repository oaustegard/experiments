"""Paths, task loading and per-repo test recipes for the SWE-bench Verified ladder.

Every command that executes code from a task repository goes through
claude-workspace's scripts/jail.sh: no network, uid nobody, empty environment.
Dependencies are installed outside the jail from PyPI; the repositories
themselves are never pip-installed, they are put on PYTHONPATH.
"""
from __future__ import annotations

import json
import os
import shlex
import subprocess
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
DATA = EXP / "data"
TASKS = DATA / "tasks.jsonl"          # regenerable, gitignored (fetch_tasks.py)

ROOT = Path(os.environ.get("SWE_LADDER_ROOT", "/tmp/jail/swe"))
MIRRORS = ROOT / "mirrors"
ENVS = ROOT / "envs"
WORK = ROOT / "work"
BIN = ROOT / "bin"
UVPY = Path(os.environ.get("SWE_LADDER_UVPY", "/opt/uvpy"))


def _hub() -> Path:
    for p in (os.environ.get("CLAUDE_WORKSPACE"), Path.home() / "claude-workspace",
              "/home/user/claude-workspace"):
        if p and (Path(p) / "scripts" / "jail.sh").is_file():
            return Path(p)
    raise FileNotFoundError("claude-workspace checkout with scripts/jail.sh not found; set CLAUDE_WORKSPACE")


JAIL = _hub() / "scripts" / "jail.sh"

# Repos whose code is pure Python and whose tests run without Docker.
PURE = ("django/django", "sympy/sympy")

TASK_MARKER = ".swe-task.json"


def load_tasks(ids: set[str] | None = None) -> dict[str, dict]:
    out = {}
    with open(TASKS) as f:
        for line in f:
            t = json.loads(line)
            if ids is None or t["instance_id"] in ids:
                out[t["instance_id"]] = t
    return out


def mirror(repo: str) -> Path:
    return MIRRORS / (repo.split("/")[1] + ".git")


def python_for(spec_python: str) -> str:
    """Interpreter version actually used. 3.5-3.7 have no uv build; 3.8 stands in,
    and validate.py drops any task the substitution breaks."""
    return "3.8" if spec_python in ("3.5", "3.6", "3.7") else spec_python


def env_key(task: dict) -> str:
    return f"{task['repo'].replace('/', '__')}__{task['version']}"


def test_directives(task: dict) -> list[str]:
    """Test targets from the test patch, as SWE-bench's harness derives them."""
    import re
    files = re.findall(r"diff --git a/.* b/(.*)", task["test_patch"])
    files = [f for f in files if f.endswith(".py")]
    if task["repo"] == "django/django":
        out = []
        for d in files:
            d = d[:-3]
            d = d[len("tests/"):] if d.startswith("tests/") else d
            out.append(d.replace("/", "."))
        return out
    return files


def test_command(task: dict, targets: list[str]) -> tuple[list[str], dict[str, str]]:
    """argv and extra env for running `targets` in a task checkout, inside the jail."""
    repo = task["repo"]
    if repo == "django/django":
        argv = ["python", "./tests/runtests.py", "--verbosity", "2",
                "--settings=test_sqlite", "--parallel", "1", *targets]
        return argv, {"PYTHONIOENCODING": "utf8"}
    if repo == "sympy/sympy":
        argv = ["python", "bin/test", "-C", "--verbose", *targets]
        return argv, {"PYTHONWARNINGS": "ignore::UserWarning,ignore::SyntaxWarning"}
    raise KeyError(repo)


def jailed(checkout: Path, argv: list[str], env: dict[str, str], timeout: int) -> list[str]:
    """The full command line that runs argv in checkout under jail.sh with a time limit."""
    # Unbuffered: with stdout block-buffered behind stderr, Django's banner lands
    # after the results, and `swe-test ... | tail` showed agents only the banner.
    cmd = ["sh", str(JAIL), "-C", str(checkout), "-e", f"PYTHONPATH={checkout}",
           "-e", "PYTHONUNBUFFERED=1"]
    for k, v in env.items():
        cmd += ["-e", f"{k}={v}"]
    return cmd + ["--", "timeout", "-k", "10", str(timeout), *argv]


def run_jailed(checkout: Path, argv: list[str], env: dict[str, str], timeout: int) -> tuple[int, str]:
    p = subprocess.run(jailed(checkout, argv, env, timeout), stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, text=True, errors="replace")
    return p.returncode, p.stdout


def git(*args, cwd: Path | None = None, check: bool = True, input: str | None = None) -> str:
    p = subprocess.run(["git", *args], cwd=cwd, input=input, text=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if check and p.returncode:
        raise RuntimeError(f"git {shlex.join(args)} failed: {p.stderr.strip()[-800:]}")
    return p.stdout
