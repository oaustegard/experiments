"""Live enumeration over the seed registry.

Every seed must have a built task that still satisfies the invariant its class claims.
These assert against the committed tasks/ tree, so a seed edited without a rebuild, or a
task tree edited by hand, fails here rather than silently changing what Stage 1 measures.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
HARNESS = ROOT / "harness"
sys.path.insert(0, str(HARNESS))

import build_tasks as B          # noqa: E402
import grade_agentic as G        # noqa: E402
import oracle as O               # noqa: E402

SEEDS = sorted((ROOT / "seeds").glob("*.json"))
SEED_NAMES = [s.stem for s in SEEDS]

assert SEED_NAMES, "no seeds found; the registry cannot be empty"


def meta(task):
    return json.loads((ROOT / "tasks" / task / "meta.json").read_text())


def test_every_seed_has_a_built_task():
    built = {d.name for d in (ROOT / "tasks").iterdir() if (d / "meta.json").exists()}
    assert built == set(SEED_NAMES), f"seeds and tasks disagree: {built ^ set(SEED_NAMES)}"


@pytest.mark.parametrize("task", SEED_NAMES)
def test_task_files_present(task):
    d = ROOT / "tasks" / task
    m = meta(task)
    for rel in ("issue.md", "meta.json", "tests_hidden.py",
                f"repo/{m['package']}/__init__.py", f"repo/{m['package']}/core.py",
                "repo/tests/test_public.py"):
        assert (d / rel).exists(), f"{task}: missing {rel}"
    assert (d / "repo" / m["package"] / "util.py").exists() == bool(m["moved_to_util"])


@pytest.mark.parametrize("task", SEED_NAMES)
def test_task_starts_red_under_hidden_suite(task):
    r = G.grade_one(ROOT / "tasks" / task / "repo", task)
    assert not r["passed"], f"{task}: repo already passes; nothing for an agent to fix"
    assert not r["tampered_tests"]
    assert r["n_total"] > 0


@pytest.mark.parametrize("task", SEED_NAMES)
def test_visible_tests_are_a_subset_of_hidden(task):
    m = meta(task)
    hidden = (ROOT / "tasks" / task / "tests_hidden.py").read_text()
    for name in m["visible_tests"]:
        assert f"def {name}(" in hidden, f"{task}: visible test {name} is not in the hidden suite"


@pytest.mark.parametrize("task", SEED_NAMES)
def test_visible_suite_is_red(task):
    rc, _ = B.run_pytest(ROOT / "tasks" / task / "repo")
    assert rc != 0, f"{task}: the agent would see a green suite and have no failure to chase"


@pytest.mark.parametrize("task", SEED_NAMES)
def test_issue_does_not_leak_the_bug_site(task):
    """The issue text may name the failing test; it must not name the file holding the bug.

    This is the property the whole experiment rests on: if the prompt located the bug,
    there would be no Bayes-error floor to route around.
    """
    m = meta(task)
    issue = (ROOT / "tasks" / task / "issue.md").read_text()
    assert m["bug_file"] not in issue, f"{task}: issue.md names {m['bug_file']}"
    if m["moved_to_util"]:
        assert "util" not in issue, f"{task}: issue.md points at the util module"


@pytest.mark.parametrize("task", [t for t in SEED_NAMES
                                 if json.loads((ROOT / "seeds" / f"{t}.json").read_text())
                                 .get("local_fix")])
def test_contract_tasks_have_residue(task):
    m = meta(task)
    assert m["class"] == "contract"
    assert m["contract_residue"], f"{task}: no hidden test survives the local fix"
    assert not set(m["contract_residue"]) & set(m["visible_tests"]), \
        f"{task}: residue overlaps the visible set, so the local fix does not look complete"


def test_class_spread_is_documented():
    counts = {}
    for t in SEED_NAMES:
        counts[meta(t)["class"]] = counts.get(meta(t)["class"], 0) + 1
    params = json.loads((ROOT / "params.json").read_text())
    assert set(counts) <= set(params["task_classes"]), f"undocumented class in {counts}"
    assert len(counts) >= 2, "a single difficulty class cannot show a routing gradient"
    assert params["protocol"]["n_tasks"] == len(SEED_NAMES)


def test_oracle_arithmetic():
    d = O.demo()
    # weak solves t0-t2, strong solves t1-t5: the oracle must take every task it can and
    # must never route a task to weak that weak fails.
    assert d["solve_sets"]["weak_only"] == ["t0"]
    assert d["oracle"]["solved"] == 6
    assert all(d["oracle"]["routed"][t] == "weak" for t in ("t0", "t1", "t2"))
    assert all(d["oracle"]["routed"][t] == "strong" for t in ("t3", "t4", "t5"))
    assert d["oracle"]["cost_per_completed"] < d["arms"]["weak"]["cost_per_completed"]


def test_grader_rejects_tampered_tests(tmp_path):
    task = SEED_NAMES[0]
    work = tmp_path / "w"
    import shutil
    shutil.copytree(ROOT / "tasks" / task / "repo", work)
    (work / "tests" / "test_public.py").write_text("def test_trivial():\n    assert True\n")
    r = G.grade_one(work, task)
    assert r["tampered_tests"]
    assert not r["passed"], "a run that rewrote its tests must not be scored as a pass"


def test_build_is_reproducible():
    proc = subprocess.run([sys.executable, str(HARNESS / "build_tasks.py"), "--check"],
                          capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, f"committed tasks are stale:\n{proc.stdout}\n{proc.stderr}"


@pytest.mark.parametrize("task", SEED_NAMES)
def test_visible_suite_fails_rather_than_errors(task):
    """A suite that dies on a missing name is red for the wrong reason.

    The pilot shipped two tasks whose sliced visible suite raised NameError on a helper
    the hidden suite defines for itself. Both runs "fixed" it by injecting the name via
    conftest.py instead of touching the seeded bug, so the trajectory measured something
    the experiment never intended to ask about.
    """
    import subprocess
    repo = ROOT / "tasks" / task / "repo"
    proc = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-q", "--no-header",
                           "-p", "no:cacheprovider"],
                          cwd=repo, capture_output=True, text=True, timeout=120)
    out = proc.stdout + proc.stderr
    assert "NameError" not in out, f"{task}: visible suite references an undefined name\n{out[-1500:]}"
    assert "errors during collection" not in out, f"{task}: visible suite fails to collect"


@pytest.mark.parametrize("task", SEED_NAMES)
def test_visible_suite_is_green_without_the_bug(task):
    """The slice itself must be sound: green on the reference, red only under the seed."""
    import shutil
    import tempfile
    seed = json.loads((ROOT / "seeds" / f"{task}.json").read_text())
    src = ROOT.parent / "orchestrated-coding-pareto" / "tasks" / task
    core, util = B.split_module((src / "reference.py").read_text(), seed.get("move_to_util", []))
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp) / "repo"
        B.write_repo(repo, seed, core, util, (src / "tests.py").read_text(),
                     meta(task)["visible_tests"])
        rc, out = B.run_pytest(repo)
    assert rc == 0, f"{task}: visible slice is not green on the unbugged reference\n{out[-1500:]}"


def test_grader_discards_edits_outside_the_package(tmp_path):
    """Only the package is the run's to change; anything else is reverted, not trusted."""
    import shutil
    task = "roman_strict"
    work = tmp_path / "w"
    shutil.copytree(ROOT / "tasks" / task / "repo", work)
    (work / "conftest.py").write_text("import builtins\nbuiltins.SMUGGLED = 1\n")
    r = G.grade_one(work, task)
    assert r["edits_outside_package"] == ["conftest.py"]
    assert not r["passed"], "the package still carries its bug, so the run must not pass"
