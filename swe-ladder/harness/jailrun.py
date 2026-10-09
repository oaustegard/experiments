"""The agents' only route to executing task-repo code: swe-test and swe-run.

    swe-test TARGET [TARGET ...]    the repo's test runner on these targets
    swe-run CMD [ARG ...]           any command, e.g. swe-run python repro.py

Both work from anywhere inside a task checkout (found by its .swe-task.json),
run under claude-workspace's scripts/jail.sh (no network, uid nobody, empty
environment) with the checkout on PYTHONPATH and its env's python first on PATH.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import TASK_MARKER, jailed, test_command  # noqa: E402

TIMEOUT = {"swe-test": 900, "swe-run": 300}
HINT = {
    "django/django": "e.g. swe-test admin_views.tests  or  swe-test utils_tests.test_text.TestUtilsText",
    "sympy/sympy": "e.g. swe-test sympy/core/tests/test_basic.py  or  swe-test sympy/core/tests/test_basic.py -k test_name",
}


def find_checkout(start: Path) -> Path | None:
    for p in (start, *start.parents):
        if (p / TASK_MARKER).is_file():
            return p
    return None


def main(prog: str, args: list[str]) -> int:
    root = find_checkout(Path.cwd())
    if root is None:
        print(f"{prog}: run this from inside a task checkout (no {TASK_MARKER} above {Path.cwd()})",
              file=sys.stderr)
        return 2
    task = json.loads((root / TASK_MARKER).read_text())
    if not args:
        print(f"{prog}: name what to run. {HINT.get(task['repo'], '')}", file=sys.stderr)
        return 2
    if prog == "swe-test":
        argv, env = test_command(task, args)
    else:
        argv, env = list(args), {}
    cmd = jailed(root, argv, env, TIMEOUT[prog])
    if Path.cwd() != root:
        # jail.sh -C sets the directory; keep the caller's relative position.
        rel = Path.cwd().relative_to(root)
        i = cmd.index("--")
        cmd = cmd[:i + 1] + ["sh", "-c", 'cd "$1" && shift && exec "$@"', "_", str(rel)] + cmd[i + 1:]
    # Print the status as the last line: behind `| tail` the shell reports
    # tail's exit code, and one pilot agent read that 0 as "tests pass".
    code = subprocess.run(cmd).returncode
    sys.stdout.flush()
    print(f"[{prog} exit status {code}{' (timed out)' if code in (124, 137) else ''}]", flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2:]))
