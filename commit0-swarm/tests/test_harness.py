"""Unit checks for the pieces that decide what an agent is given and how it is scored.

    .venv/bin/python -m pytest tests -q
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))

from run import partition, stub_weights  # noqa: E402
from workspace import parse_junit  # noqa: E402
import prompts  # noqa: E402


def test_stub_weights_counts_empty_bodies_and_owns_every_file(tmp_path):
    pkg = tmp_path / "lib"
    pkg.mkdir()
    (pkg / "a.py").write_text('def f():\n    """doc"""\n    pass\n\ndef g():\n    return 1\n\n'
                              'class C:\n    def m(self):\n        pass\n')
    (pkg / "b.py").write_text("def h():\n    return 2\n")
    w = stub_weights({"src_dir": "lib"}, tmp_path)
    assert w == {"lib/a.py": 3, "lib/b.py": 1}


def test_partition_covers_every_file_once_and_balances():
    w = {f"f{i}.py": n for i, n in enumerate([30, 20, 10, 10, 5, 5])}
    bins = partition(w, 3)
    flat = [f for b in bins for f in b]
    assert sorted(flat) == sorted(w)
    loads = sorted(sum(w[f] for f in b) for b in bins)
    assert loads[-1] - loads[0] <= 10


def test_partition_never_returns_empty_bins():
    assert len(partition({"a.py": 1}, 4)) == 1


def test_parse_junit_failure_error_skip(tmp_path):
    x = tmp_path / "j.xml"
    x.write_text('<testsuites><testsuite>'
                 '<testcase classname="t.A" name="ok"/>'
                 '<testcase classname="t.A" name="bad"><failure/></testcase>'
                 '<testcase classname="t.A" name="err"><error/></testcase>'
                 '<testcase classname="t.A" name="sk"><skipped/></testcase>'
                 '</testsuite></testsuites>')
    assert parse_junit(x) == {"t.A::ok": "pass", "t.A::bad": "fail", "t.A::err": "fail", "t.A::sk": "skip"}


def test_parse_junit_truncated_is_none_not_empty(tmp_path):
    x = tmp_path / "j.xml"
    x.write_text("<testsuites><testsuite><testcase")
    assert parse_junit(x) is None


def test_prompts_keep_out_placeholder_and_single_braces():
    task = {"name": "tinydb", "original_repo": "msiemens/tinydb", "src_dir": "tinydb",
            "test": {"test_dir": "tests/"}}
    for p in (prompts.solo(task, "/x"), prompts.swarm(task, "/x", 1, 2, ["tinydb/a.py"]),
              prompts.fixer(task, "/x", 1, 2, "3 passed", "/x/.claims"),
              prompts.solo_cont(task, "/x", "3 passed")):
        assert "{out}" in p
        assert '{"summary"' in p and "{{" not in p


def test_agent_tree_has_no_history_and_patch_round_trips(tmp_path, monkeypatch):
    """The stub commit's parent is the reference: an agent tree must not carry it."""
    import subprocess
    import run
    import workspace
    from common import load_tasks, mirror
    if not mirror("tinydb").exists():
        import pytest
        pytest.skip("mirrors not set up")
    task = load_tasks()["tinydb"]
    monkeypatch.setattr(run, "WORK", tmp_path)
    monkeypatch.setattr(run, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(workspace, "opener", lambda p: None)
    monkeypatch.setattr(workspace, "babel_data", lambda t, d: None)
    t = workspace.checkout(task, run.tree("r", "tinydb"), history=False)
    assert subprocess.run(["git", "rev-list", "--count", "HEAD"], cwd=t, capture_output=True,
                          text=True).stdout.strip() == "1"
    assert subprocess.run(["git", "cat-file", "-e", task["reference_commit"]], cwd=t,
                          stderr=subprocess.DEVNULL).returncode != 0
    (t / "tinydb" / "utils.py").write_text("# changed\n")
    subprocess.run(["git", "commit", "-qam", "agent commits"], cwd=t)   # agents may commit
    (tmp_path / "runs" / "r").mkdir(parents=True)
    p = run.save_patch("r", "tinydb", task)
    g = workspace.checkout(task, tmp_path / "g")
    subprocess.run(["git", "apply", "--binary", str(p)], cwd=g, check=True)
    assert (g / "tinydb" / "utils.py").read_text() == "# changed\n"
