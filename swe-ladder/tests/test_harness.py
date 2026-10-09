"""Harness invariants that need no network, no agents and no jail.

    .venv/bin/python -m pytest -q tests/
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "harness"))

import common  # noqa: E402
import grade  # noqa: E402
import prompts  # noqa: E402
import workspace  # noqa: E402

DJ = {"repo": "django/django", "instance_id": "django__django-1", "version": "3.0",
      "base_commit": "abcdef1234567890", "problem_statement": "Bug {x} with braces }",
      "FAIL_TO_PASS": ["test_new (app.tests.T)"],
      "PASS_TO_PASS": ["test_old (app.tests.T)", "test_doc (app.tests.T)"],
      "test_patch": "diff --git a/tests/app/tests.py b/tests/app/tests.py\n"
                    "diff --git a/tests/app/data.json b/tests/app/data.json\n"}
SY = dict(DJ, repo="sympy/sympy", instance_id="sympy__sympy-1",
          test_patch="diff --git a/sympy/core/tests/test_basic.py b/sympy/core/tests/test_basic.py\n")


def test_test_directives_follow_swebench():
    assert common.test_directives(DJ) == ["app.tests"]
    assert common.test_directives(SY) == ["sympy/core/tests/test_basic.py"]


def test_django_docstring_line_counts_under_the_test_name():
    log = ("test_old (app.tests.T) ... ok\n"
           "test_doc (app.tests.T)\n"
           "Docstring first line ... ok\n"
           "test_new (app.tests.T) ... FAIL\n")
    sm = grade.status_map(DJ, log) if False else grade.parser("django/django")(
        grade.django_docstring_lines(log), None)
    assert sm["test_doc (app.tests.T)"] == "PASSED"
    assert sm["Docstring first line"] == "PASSED"      # the other key survives too
    r = grade.report(DJ, sm)
    assert r["resolved"] is False and r["f2p_fail"] == ["test_new (app.tests.T)"]
    assert r["p2p"] == [2, 2]


def test_prompts_render_once_with_out_placeholder():
    for task in (DJ, SY):
        p = prompts.rung1(task, "/tmp/jail/swe/work/r/x")
        assert p.count("{out}") == 1 and "{{" not in p and "}}" not in p
        assert "Bug {x} with braces }" in p                 # issue text is not formatted
        assert "/tmp/jail/swe/bin/swe-test" in p
        q = prompts.rung2(task, "/w", "diff --git a/x b/x\n+{y}\n", ["test_new"], "Trace {z}")
        assert q.count("{out}") == 1 and "+{y}" in q and "Trace {z}" in q


def test_failure_excerpt_keeps_only_graded_failures():
    log = ("test_new (app.tests.T) ... FAIL\n"
           "======================================================================\n"
           "FAIL: test_new (app.tests.T)\n"
           "----------------------------------------------------------------------\n"
           "AssertionError: 1 != 2\n\n"
           "======================================================================\n"
           "ERROR: test_unrelated (other.tests.U)\n"
           "----------------------------------------------------------------------\n"
           "boom\n\n"
           "----------------------------------------------------------------------\n"
           "Ran 3 tests in 0.1s\n")
    ex = prompts.failure_excerpt(DJ, log, ["test_new (app.tests.T)"])
    assert "AssertionError: 1 != 2" in ex and "boom" not in ex and "Ran 3" not in ex
    slog = ("_______________ sympy/core/tests/test_basic.py:test_subs ________________\n"
            "Traceback\nAssertionError\n"
            "_______________ sympy/core/tests/test_basic.py:test_other _______________\nother\n"
            "======= tests finished =======\n")
    ex = prompts.failure_excerpt(SY, slog, ["test_subs"])
    assert "test_subs" in ex and "other" not in ex
    assert "last lines" in prompts.failure_excerpt(SY, "crash\n", ["test_subs"])


def test_diff_includes_new_files_and_skips_harness_files(tmp_path):
    repo = tmp_path / "r"
    repo.mkdir()
    run = lambda *a: subprocess.run(["git", *a], cwd=repo, check=True, capture_output=True)
    run("init", "-q")
    (repo / "a.py").write_text("x = 1\n")
    run("add", "a.py")
    run("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "base")
    base = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip()
    (repo / ".git" / "info" / "exclude").write_text(".venv\n.swe-task.json\n")
    (repo / "a.py").write_text("x = 2\n")
    (repo / "new.py").write_text("y = 1\n")
    (repo / ".swe-task.json").write_text("{}")
    d = workspace.diff(repo, base)
    assert "+x = 2" in d and "new.py" in d and ".swe-task.json" not in d


def test_jailed_command_shape():
    cmd = common.jailed(Path("/tmp/jail/swe/work/r/x"), ["python", "t.py"], {"A": "1"}, 60)
    assert cmd[:2] == ["sh", str(common.JAIL)]
    assert cmd[cmd.index("--") + 1:] == ["timeout", "-k", "10", "60", "python", "t.py"]
    assert "PYTHONPATH=/tmp/jail/swe/work/r/x" in cmd and "A=1" in cmd


def test_python_substitution():
    assert common.python_for("3.6") == "3.8" and common.python_for("3.11") == "3.11"


def test_merge_replaces_redone_tasks_only(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    import ladder
    monkeypatch.setattr(ladder, "RUNS", tmp_path)

    def run(name, rows):
        d = tmp_path / name
        (d / "patches").mkdir(parents=True)
        (d / "logs").mkdir()
        for iid, ok in rows.items():
            (d / "patches" / f"{iid}.diff").write_text(f"{name} {iid}")
            (d / "logs" / f"{iid}.log").write_text(f"{name} {iid}")
        (d / "results.jsonl").write_text(
            "".join(json.dumps({"instance_id": i, "resolved": ok}) + "\n" for i, ok in rows.items()))

    run("base", {"a": True, "b": False})
    run("redo", {"b": True})
    ladder.merge(SimpleNamespace(out="m", base="base", override="redo"))
    m = tmp_path / "m"
    assert ladder.results("m") == {"a": {"instance_id": "a", "resolved": True},
                                   "b": {"instance_id": "b", "resolved": True}}
    assert (m / "patches" / "a.diff").read_text() == "base a"
    assert (m / "patches" / "b.diff").read_text() == "redo b"
    assert (m / "logs" / "b.log").read_text() == "redo b"
