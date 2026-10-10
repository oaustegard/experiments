"""Paths, task loading and the jailed test recipe for Commit0-lite.

Every command that executes code from a library repo goes through
claude-workspace's scripts/jail.sh: no network, uid nobody, empty environment.
Dependencies come from PyPI outside the jail; the libraries themselves are never
pip-installed, checkouts go on PYTHONPATH (src/ layouts put src/ there).

    python harness/common.py names     # the 16 lite library names, one per line
"""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
DATA = EXP / "data"
PARQUET = DATA / "commit0_combined.parquet"
TASKS = DATA / "tasks.json"

ROOT = Path(os.environ.get("C0_ROOT", "/tmp/jail/c0"))
MIRRORS = ROOT / "mirrors"
ENVS = ROOT / "envs"
WORK = ROOT / "work"
BIN = ROOT / "bin"
UVPY = Path(os.environ.get("C0_UVPY", "/opt/uvpy"))

TASK_MARKER = ".c0-task.json"

# Commit0's lite split (commit0/harness/constants.py SPLIT_LITE).
LITE = ("simpy", "tinydb", "marshmallow", "parsel", "pyjwt", "minitorch", "wcwidth",
        "imapclient", "chardet", "babel", "voluptuous", "jinja", "deprecated",
        "cachetools", "cookiecutter", "portalocker")


def _hub() -> Path:
    for p in (os.environ.get("CLAUDE_WORKSPACE"), Path.home() / "claude-workspace",
              "/home/user/claude-workspace"):
        if p and (Path(p) / "scripts" / "jail.sh").is_file():
            return Path(p)
    raise FileNotFoundError("claude-workspace checkout with scripts/jail.sh not found; set CLAUDE_WORKSPACE")


JAIL = _hub() / "scripts" / "jail.sh"


def load_tasks() -> dict[str, dict]:
    """name -> task dict, built from the parquet on first use."""
    if not TASKS.exists():
        import pandas as pd
        d = pd.read_parquet(PARQUET)
        out = {}
        for _, r in d.iterrows():
            name = r["repo"].split("/")[1]
            if name not in LITE:
                continue
            setup = {k: (list(v) if hasattr(v, "tolist") else v) for k, v in r["setup"].items()}
            out[name] = {"name": name, "repo": r["repo"], "original_repo": r["original_repo"],
                         "base_commit": r["base_commit"], "reference_commit": r["reference_commit"],
                         "setup": setup, "test": dict(r["test"]), "src_dir": r["src_dir"].rstrip("/")}
        TASKS.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    tasks = json.loads(TASKS.read_text())
    control = DATA / "control_tasks.json"   # post-cutoff libraries (control_setup.py)
    if control.exists():
        tasks.update(json.loads(control.read_text()))
    return tasks


def mirror(name: str) -> Path:
    return MIRRORS / f"{name}.git"


def pythonpath(task: dict, checkout: Path) -> str:
    """src/ layouts import from src/; everything else from the checkout root."""
    src = task["src_dir"]
    if src.startswith("src/"):
        return f"{checkout / 'src'}:{checkout}"
    return str(checkout)


def test_argv(task: dict, targets: list[str], junit: str | None = None) -> list[str]:
    """pytest argv for targets (default: the task's test dir), with Commit0's flags."""
    cmd = shlex.split(task["test"]["test_cmd"])
    if cmd[0] == "pytest":
        cmd = ["python", "-m", "pytest", *cmd[1:]]
    # No cache dir: concurrent swarm workers share one tree and one would clobber another's.
    cmd += ["-p", "no:cacheprovider", "-p", "no:cov", "-o", "addopts="]
    if junit:
        cmd += [f"--junitxml={junit}"]
    return cmd + (targets or [task["test"]["test_dir"]])


def jailed(task: dict, checkout: Path, argv: list[str], timeout: int,
           env: dict[str, str] | None = None) -> list[str]:
    cmd = ["sh", str(JAIL), "-C", str(checkout), "-e", f"PYTHONPATH={pythonpath(task, checkout)}",
           "-e", "PYTHONUNBUFFERED=1", "-e", "PYTHONDONTWRITEBYTECODE=1", "-e", "HOME=/tmp"]
    for k, v in (env or {}).items():
        cmd += ["-e", f"{k}={v}"]
    return cmd + ["--", "timeout", "-k", "10", str(timeout), *argv]


def run_jailed(task: dict, checkout: Path, argv: list[str], timeout: int) -> tuple[int, str]:
    p = subprocess.run(jailed(task, checkout, argv, timeout), stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, text=True, errors="replace")
    return p.returncode, p.stdout


def git(*args, cwd: Path | None = None, check: bool = True, input: str | None = None) -> str:
    p = subprocess.run(["git", *args], cwd=cwd, input=input, text=True, errors="replace",
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if check and p.returncode:
        raise RuntimeError(f"git {shlex.join(args)} failed: {p.stderr.strip()[-800:]}")
    return p.stdout


if __name__ == "__main__":
    if sys.argv[1:] == ["names"]:
        print("\n".join(LITE))
