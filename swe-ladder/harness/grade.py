"""Grade a patch for one task the way SWE-bench does, minus Docker.

Fresh checkout at base_commit -> apply the patch -> restore the test files the
test patch touches -> apply the test patch -> run the test-patch modules in the
jail -> parse with SWE-bench's own log parser -> FAIL_TO_PASS and PASS_TO_PASS
must all pass.

    python harness/grade.py INSTANCE_ID PATCH_FILE [--keep]
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import git, load_tasks, run_jailed, test_command, test_directives  # noqa: E402
import workspace  # noqa: E402

GRADE_TIMEOUT = 1800


def parser(repo: str):
    from swebench.harness.log_parsers import MAP_REPO_TO_PARSER  # swebench==4.0.4
    return MAP_REPO_TO_PARSER[repo]


_DJANGO_NAME = re.compile(r"^(\w+ \([\w.]+\))$")


def django_docstring_lines(log: str) -> str:
    """Django at verbosity 2 prints a test's docstring between its name and the
    verdict: 'test_x (app.tests.T)' then 'Docstring line ... ok'. The dataset keys
    such tests by name, SWE-bench's parser by the docstring line. Add a synthetic
    'name ... verdict' line after each such pair so both keys get the status."""
    lines = log.split("\n")
    out = []
    for i, line in enumerate(lines):
        out.append(line)
        m = _DJANGO_NAME.match(line.strip())
        if m and i + 1 < len(lines) and " ... " in lines[i + 1] and not _DJANGO_NAME.match(
                lines[i + 1].split(" ... ")[0].strip()):
            verdict = lines[i + 1].rsplit(" ... ", 1)[1]
            out.append(f"{m.group(1)} ... {verdict}")
    return "\n".join(out)


def status_map(task: dict, log: str) -> dict[str, str]:
    spec = SimpleNamespace(repo=task["repo"], version=task["version"],
                           instance_id=task["instance_id"])
    if task["repo"] == "django/django":
        log = django_docstring_lines(log)
    return parser(task["repo"])(log, spec)


def report(task: dict, sm: dict[str, str]) -> dict:
    ok = lambda t: sm.get(t) in ("PASSED", "XFAIL")
    f2p_fail = [t for t in task["FAIL_TO_PASS"] if not ok(t)]
    p2p_fail = [t for t in task["PASS_TO_PASS"] if not ok(t)]
    return {"resolved": not f2p_fail and not p2p_fail,
            "f2p": [len(task["FAIL_TO_PASS"]) - len(f2p_fail), len(task["FAIL_TO_PASS"])],
            "p2p": [len(task["PASS_TO_PASS"]) - len(p2p_fail), len(task["PASS_TO_PASS"])],
            "f2p_fail": f2p_fail, "p2p_fail": p2p_fail}


def touched_files(patch: str) -> list[str]:
    return re.findall(r"^diff --git a/.* b/(.*)$", patch, flags=re.M)


def grade(task: dict, patch: str, run: str, name: str, log_path: Path | None = None,
          keep: bool = False) -> dict:
    t0 = time.time()
    path = workspace.checkout(task, run, name)
    try:
        applied, how = workspace.apply(path, patch)
        if not applied:
            return {"resolved": False, "error": "patch_apply_failed", "detail": how,
                    "seconds": round(time.time() - t0, 1)}
        base = task["base_commit"]
        for f in touched_files(task["test_patch"]):
            # SWE-bench resets these to base before applying the test patch, so a
            # run that edited the hidden tests' files gains nothing from it.
            if git("ls-tree", base, "--", f, cwd=path).strip():
                git("checkout", base, "--", f, cwd=path)
            elif (path / f).exists():
                (path / f).unlink()
        ok, how = workspace.apply(path, task["test_patch"])
        if not ok:
            return {"resolved": False, "error": "test_patch_apply_failed", "detail": how,
                    "seconds": round(time.time() - t0, 1)}
        argv, env = test_command(task, test_directives(task))
        code, log = run_jailed(path, argv, env, GRADE_TIMEOUT)
        if log_path:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(log)
        out = report(task, status_map(task, log))
        out.update(exit=code, timeout=code in (124, 137), seconds=round(time.time() - t0, 1))
        return out
    finally:
        if not keep:
            workspace.remove(task, path)


def main(argv):
    iid, patch_file = argv[0], argv[1]
    task = load_tasks({iid})[iid]
    print(json.dumps(grade(task, Path(patch_file).read_text(), "adhoc", iid,
                           keep="--keep" in argv), indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
