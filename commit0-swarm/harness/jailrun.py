"""The agents' only route to executing library code: c0-test and c0-run.

    c0-test [TARGET ...]     pytest on these targets (default: the whole suite)
    c0-run CMD [ARG ...]     any command, e.g. c0-run python -c 'import tinydb'

Both work from anywhere inside a checkout (found by its .c0-task.json) and run
under claude-workspace's scripts/jail.sh (no network, uid nobody, empty
environment) with the checkout on PYTHONPATH and its env's python first on PATH.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import TASK_MARKER, jailed, load_tasks, test_argv  # noqa: E402

TIMEOUT = {"c0-test": 900, "c0-run": 300}


def find_checkout(start: Path) -> Path | None:
    for p in (start, *start.parents):
        if (p / TASK_MARKER).is_file():
            return p
    return None


def main(prog: str, args: list[str]) -> int:
    root = find_checkout(Path.cwd())
    if root is None:
        print(f"{prog}: run this from inside a library checkout (no {TASK_MARKER} above {Path.cwd()})",
              file=sys.stderr)
        return 2
    task = load_tasks()[json.loads((root / TASK_MARKER).read_text())["name"]]
    if prog == "c0-test":
        argv = test_argv(task, args)
    elif not args:
        print("c0-run: name a command, e.g. c0-run python -c 'import x'", file=sys.stderr)
        return 2
    else:
        argv = list(args)
    cmd = jailed(task, root, argv, TIMEOUT[prog])
    if Path.cwd() != root:
        rel = Path.cwd().relative_to(root)
        i = cmd.index("--")
        cmd = cmd[:i + 1] + ["sh", "-c", 'cd "$1" && shift && exec "$@"', "_", str(rel)] + cmd[i + 1:]
    code = subprocess.run(cmd).returncode
    sys.stdout.flush()
    # Last line: behind `| tail` the shell reports tail's status, not ours.
    print(f"[{prog} exit status {code}{' (timed out)' if code in (124, 137) else ''}]", flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2:]))
