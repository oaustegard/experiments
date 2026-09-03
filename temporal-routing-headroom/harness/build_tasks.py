#!/usr/bin/env python3
"""Generate the agentic task set from orchestrated-coding-pareto's single-shot tasks.

Each source task is a spec + one reference module + a hidden pytest suite. Here each
becomes a small multi-file repository carrying a seeded bug, plus an issue report that
states the failing test and nothing about the bug's depth. That last property is the
point: SWE-Router's premise is that the task description cannot separate a one-line typo
from a cross-module fix, and this generator constructs that confound deliberately rather
than hoping a benchmark contains it.

Every task is verified at build time and refuses to ship if an invariant fails:

  pristine   -- the split repo (no bug) passes the hidden suite
  bugged     -- the bug makes the hidden suite fail
  visible    -- the exposed subset of the hidden suite also fails, so the agent sees it
  contract   -- for class "contract" only: the authored local_fix makes every visible
                test pass while at least one hidden test outside the visible set stays
                red. That is what makes a locally-plausible patch wrong.

Usage:
  python3 build_tasks.py            # build every seed
  python3 build_tasks.py --task X   # build one
  python3 build_tasks.py --check    # rebuild into a temp dir and diff against tasks/
"""
import argparse
import ast
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEEDS = ROOT / "seeds"
TASKS = ROOT / "tasks"
SOURCE_TASKS = ROOT.parent / "orchestrated-coding-pareto" / "tasks"

MAX_VISIBLE = 3
PYTEST_TIMEOUT = 60

ISSUE_TEMPLATE = """# Bug report

The test suite in this repository fails.

```
$ python -m pytest tests/ -q
{output}
```

Fix the source so that the suite passes.

Constraints:
- Do not modify anything under `tests/`.
- Do not add new dependencies; the standard library is all that is available.
- Keep the public API of the `{package}` package unchanged: `{exports}`.
"""


# ---------------------------------------------------------------- module splitting

def _named_statements(tree, src_lines):
    """Yield (name, source_text) for each top-level named statement."""
    for stmt in tree.body:
        text = "\n".join(src_lines[stmt.lineno - 1:stmt.end_lineno])
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            yield stmt.name, text
        elif isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 \
                and isinstance(stmt.targets[0], ast.Name):
            yield stmt.targets[0].id, text
        else:
            yield None, text


def split_module(source, move_to_util):
    """Partition a flat reference module into (core_src, util_src).

    Top-level imports are duplicated into both files; named statements listed in
    move_to_util go to util.py and core.py imports them back. Statement order within
    each file is preserved, so definition-before-use still holds.
    """
    tree = ast.parse(source)
    lines = source.splitlines()
    imports, core, util = [], [], []
    seen = set()
    for name, text in _named_statements(tree, lines):
        node = ast.parse(text).body[0] if text.strip() else None
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(text)
            continue
        if name is not None and name in move_to_util:
            util.append(text)
            seen.add(name)
        else:
            core.append(text)
    missing = set(move_to_util) - seen
    if missing:
        raise SystemExit(f"move_to_util names not found at top level: {sorted(missing)}")

    header = "\n".join(imports)
    util_src = (header + "\n\n\n" if header else "") + "\n\n\n".join(util) + "\n"
    back = f"from .util import {', '.join(move_to_util)}" if move_to_util else ""
    core_head = "\n".join(x for x in (header, back) if x)
    core_src = (core_head + "\n\n\n" if core_head else "") + "\n\n\n".join(core) + "\n"
    return core_src, (util_src if move_to_util else None)


def apply_patch(text, patch, what):
    find, replace = patch["find"], patch["replace"]
    n = text.count(find)
    if n != 1:
        raise SystemExit(f"{what}: pattern matched {n} times, need exactly 1: {find!r}")
    return text.replace(find, replace)


# ---------------------------------------------------------------- repo materialisation

def write_repo(dest, seed, core_src, util_src, hidden_tests, visible_tests):
    pkg = dest / seed["package"]
    shutil.rmtree(dest, ignore_errors=True)
    pkg.mkdir(parents=True)
    (dest / "tests").mkdir()
    exports = seed["exports"]
    (pkg / "__init__.py").write_text(
        f"from .core import {', '.join(exports)}\n\n"
        f"__all__ = {exports!r}\n"
    )
    (pkg / "core.py").write_text(core_src)
    if util_src is not None:
        (pkg / "util.py").write_text(util_src)
    if visible_tests is not None:
        (dest / "tests" / "test_public.py").write_text(
            slice_tests(hidden_tests, visible_tests, seed["package"]))
    (dest / "conftest.py").write_text("")


def slice_tests(hidden_src, keep, package):
    """Extract named test functions from the hidden suite, rewriting its import.

    The hidden suites import `from solution import ...`; the agentic repo exposes the
    same names from a package instead, so the import line is rewritten and nothing else.
    """
    tree = ast.parse(hidden_src)
    lines = hidden_src.splitlines()
    head, bodies = [], []
    for stmt in tree.body:
        text = "\n".join(lines[stmt.lineno - 1:stmt.end_lineno])
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            head.append(text)
        elif isinstance(stmt, ast.FunctionDef) and stmt.name.startswith("test_"):
            # a test the slice does not keep is simply dropped
            if stmt.name in keep:
                bodies.append(text)
        else:
            # everything else -- constants, classes, and non-test helper functions the
            # suite defines for its own use -- has to come along. Dropping helpers gave
            # roman_strict and toposort_lex visible suites that died on NameError, and
            # two pilot runs "fixed" that by injecting names through conftest.py.
            head.append(text)
    head = [re.sub(r"^from solution import", f"from {package} import", h) for h in head]
    return "\n".join(head) + "\n\n\n" + "\n\n\n".join(bodies) + "\n"


def hidden_suite_for(package, hidden_src):
    return re.sub(r"^from solution import", f"from {package} import",
                  hidden_src, flags=re.M)


# ---------------------------------------------------------------- running pytest

def run_pytest(repo, extra_test=None):
    """Run the repo's visible tests, or an extra suite instead; return (rc, output)."""
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / "w"
        shutil.copytree(repo, work)
        target = ["tests/"] if (work / "tests" / "test_public.py").exists() else []
        if extra_test is not None:
            (work / "tests_hidden.py").write_text(extra_test)
            target = ["tests_hidden.py"]
        if not target:
            return 0, ""
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", *target, "-q", "--no-header", "-p", "no:cacheprovider"],
            cwd=work, capture_output=True, text=True, timeout=PYTEST_TIMEOUT)
        return proc.returncode, proc.stdout + proc.stderr


def failing_tests(repo, hidden_src, package):
    _, out = run_pytest(repo, extra_test=hidden_suite_for(package, hidden_src))
    names = re.findall(r"^FAILED tests_hidden\.py::(\w+)", out, flags=re.M)
    if not names:
        names = re.findall(r"^tests_hidden\.py::(\w+) FAILED", out, flags=re.M)
    if not names:
        names = re.findall(r"^_+ (\w+) _+$", out, flags=re.M)
    return sorted(set(names)), out


def hidden_passes(repo, hidden_src, package):
    rc, out = run_pytest(repo, extra_test=hidden_suite_for(package, hidden_src))
    return rc == 0, out


# ---------------------------------------------------------------- build one task

def build(seed, dest_root, verbose=True):
    task = seed["task"]
    pkg = seed["package"]
    src_dir = SOURCE_TASKS / task
    reference = (src_dir / "reference.py").read_text()
    hidden_src = (src_dir / "tests.py").read_text()

    core0, util0 = split_module(reference, seed.get("move_to_util", []))
    dest = dest_root / task
    repo = dest / "repo"

    def materialise(core_src, util_src, visible):
        write_repo(repo, seed, core_src, util_src, hidden_src, visible)

    # 1. pristine split must pass the hidden suite -- proves the split itself is inert
    materialise(core0, util0, None)
    ok, out = hidden_passes(repo, hidden_src, pkg)
    if not ok:
        raise SystemExit(f"[{task}] split repo fails the hidden suite before any bug:\n{out[-2500:]}")

    # 2. seed the bug; the hidden suite must go red
    bug = seed["bug"]
    core_b, util_b = core0, util0
    if bug["file"] == "core.py":
        core_b = apply_patch(core0, bug, f"[{task}] bug")
    elif bug["file"] == "util.py":
        if util0 is None:
            raise SystemExit(f"[{task}] bug targets util.py but move_to_util is empty")
        util_b = apply_patch(util0, bug, f"[{task}] bug")
    else:
        raise SystemExit(f"[{task}] bug.file must be core.py or util.py")

    materialise(core_b, util_b, None)
    failed, out = failing_tests(repo, hidden_src, pkg)
    if not failed:
        raise SystemExit(f"[{task}] bug does not make the hidden suite fail:\n{out[-2500:]}")

    # 3. choose the visible subset. With a local_fix the visible tests must be exactly
    #    the ones that fix repairs, so a locally-plausible patch looks complete.
    local_fix = seed.get("local_fix")
    if local_fix is None:
        visible = failed[:MAX_VISIBLE]
    else:
        core_l, util_l = core_b, util_b
        if local_fix["file"] == "core.py":
            core_l = apply_patch(core_b, local_fix, f"[{task}] local_fix")
        else:
            util_l = apply_patch(util_b, local_fix, f"[{task}] local_fix")
        materialise(core_l, util_l, None)
        still_failing, _ = failing_tests(repo, hidden_src, pkg)
        repaired = [t for t in failed if t not in still_failing]
        if not repaired:
            raise SystemExit(f"[{task}] local_fix repairs nothing; it must fix the visible tests")
        if not still_failing:
            raise SystemExit(f"[{task}] local_fix is a complete fix; contract class needs residue")
        visible = repaired[:MAX_VISIBLE]

    # 4a. the visible slice must be GREEN on the unbugged split. A suite that dies on a
    #     missing helper is also non-zero, so "red under the bug" alone cannot tell a
    #     seeded failure from a broken test file.
    materialise(core0, util0, visible)
    rc_clean, clean_out = run_pytest(repo)
    if rc_clean != 0:
        raise SystemExit(f"[{task}] visible suite is not green without the bug -- the "
                         f"slice is broken, not the code:\n{clean_out[-2500:]}")

    # 4b. materialise the shipped repo and confirm the agent actually sees a failure
    materialise(core_b, util_b, visible)
    rc, vis_out = run_pytest(repo)
    if rc == 0:
        raise SystemExit(f"[{task}] visible suite passes on the bugged repo:\n{vis_out[-2000:]}")
    if "NameError" in vis_out or "errors during collection" in vis_out:
        raise SystemExit(f"[{task}] visible suite errors rather than fails:\n{vis_out[-2500:]}")

    # 5. contract invariant, re-checked on the shipped visible set
    contract_residue = None
    if local_fix is not None:
        core_l, util_l = core_b, util_b
        if local_fix["file"] == "core.py":
            core_l = apply_patch(core_b, local_fix, f"[{task}] local_fix")
        else:
            util_l = apply_patch(util_b, local_fix, f"[{task}] local_fix")
        materialise(core_l, util_l, visible)
        rc_v, _ = run_pytest(repo)
        residue, _ = failing_tests(repo, hidden_src, pkg)
        if rc_v != 0:
            raise SystemExit(f"[{task}] local_fix does not pass the visible suite")
        if not residue:
            raise SystemExit(f"[{task}] local_fix passes the hidden suite; not a contract task")
        contract_residue = residue
        materialise(core_b, util_b, visible)  # restore the shipped state

    (dest / "issue.md").write_text(ISSUE_TEMPLATE.format(
        output=trim_pytest(vis_out), package=pkg, exports=", ".join(seed["exports"])))
    (dest / "tests_hidden.py").write_text(hidden_suite_for(pkg, hidden_src))
    (dest / "meta.json").write_text(json.dumps({
        "task": task,
        "class": seed["class"],
        "package": pkg,
        "exports": seed["exports"],
        "bug_file": bug["file"],
        "moved_to_util": seed.get("move_to_util", []),
        "hidden_failing_under_bug": failed,
        "visible_tests": visible,
        "contract_residue": contract_residue,
    }, indent=2) + "\n")
    if verbose:
        extra = f" residue={len(contract_residue)}" if contract_residue else ""
        print(f"  {task:16s} {seed['class']:8s} bug in {bug['file']:7s} "
              f"hidden_fail={len(failed):2d} visible={len(visible)}{extra}")
    return dest


def trim_pytest(out, keep=28):
    """Normalise pytest output for embedding in issue.md.

    Two things are removed. The run duration, because it varies between builds and made
    the generated tasks non-reproducible. And any line naming a file under the package,
    because a traceback pointing at core.py or util.py hands the agent the localisation
    the probe is supposed to earn -- and hands it asymmetrically, since only the tasks
    that raise produce such a line. Test-file references stay: the agent can read those.
    """
    lines = []
    for line in out.splitlines():
        if not line.strip():
            continue
        if re.match(r"^\S*(?:core|util)\.py:\d+: ", line) or "/core.py:" in line or "/util.py:" in line:
            continue
        line = re.sub(r"\bin \d+\.\d+s\b", "in <duration>", line)
        line = re.sub(r"0x[0-9a-f]{6,}", "0x...", line)  # object reprs vary per run
        lines.append(line)
    if len(lines) <= keep:
        return "\n".join(lines)
    return "\n".join(lines[:keep - 4] + ["...", *lines[-3:]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task")
    ap.add_argument("--check", action="store_true",
                    help="build into a temp dir and diff against the committed tasks/")
    args = ap.parse_args()

    seeds = sorted(SEEDS.glob("*.json"))
    if args.task:
        seeds = [s for s in seeds if s.stem == args.task]
        if not seeds:
            raise SystemExit(f"no seed named {args.task}")
    if not seeds:
        raise SystemExit("no seeds found")

    dest_root = Path(tempfile.mkdtemp()) if args.check else TASKS
    dest_root.mkdir(parents=True, exist_ok=True)
    print(f"building {len(seeds)} task(s) -> {dest_root}")
    for path in seeds:
        build(json.loads(path.read_text()), dest_root)

    if args.check:
        rc = 0
        for path in seeds:
            a, b = TASKS / path.stem, dest_root / path.stem
            proc = subprocess.run(["diff", "-r", str(a), str(b)], capture_output=True, text=True)
            if proc.returncode != 0:
                rc = 1
                print(f"DRIFT {path.stem}\n{proc.stdout}")
        shutil.rmtree(dest_root, ignore_errors=True)
        print("committed tasks match the seeds" if rc == 0 else "committed tasks are stale")
        return rc
    return 0


if __name__ == "__main__":
    sys.exit(main())
