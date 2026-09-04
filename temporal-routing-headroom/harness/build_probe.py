#!/usr/bin/env python3
"""Build three probe fixtures, each aimed at a different reason a tier might separate.

The paired set showed Opus at effort high failing exactly where Sonnet at effort low
failed, so "harder search" is not the axis. Our calibration puts the tier gap in judgment,
ambiguity and long-horizon work instead. These three shapes test that, cheaply, by reusing
existing reference modules and hidden suites rather than authoring new graded tasks.

  three_sites   the paired trap with a third coupled site -- more places to stop early.
                Tests whether the gap is simply thoroughness after all.
  ambiguous     the flag-dependent branch removed, and visible tests chosen so they say
                nothing about what the flag should do. The fix requires inferring intent
                from the rest of the module. Judgment row.
  no_tests      no visible suite at all, only a symptom report. Every trapped run in the
                pilot stopped because the visible suite went green; with nothing to turn
                green there is nothing to stop early on.

Usage: python3 build_probe.py
"""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "harness"))
import build_tasks as B  # noqa: E402

PROBE = ROOT / "probe"
SRC = ROOT.parent / "orchestrated-coding-pareto" / "tasks"

SHAPES = {
    "three_sites": dict(
        task="text_table", package="solution", exports=["format_table"], move_to_util=[],
        patches=[
            dict(file="core.py", find="    pad = width - len(text)",
                 replace="    pad = max(0, width - len(text) - 1)"),
            dict(file="core.py", find="        while len(line) > width:",
                 replace="        while len(line) > width - 1:"),
            dict(file="core.py",
                 find='    border = "+" + "+".join("-" * (w + 2) for w in widths) + "+"',
                 replace='    border = "+" + "+".join("-" * (w + 1) for w in widths) + "+"'),
        ],
        visible_filter=lambda n: True, max_visible=2),
    "ambiguous": dict(
        task="interval_merge", package="solution", exports=["merge"], move_to_util=[],
        patches=[
            dict(file="core.py",
                 find="            if a < pb or (a, b) == (pa, pb) or (a == pb and (join_touching or a == b == pb == pa)):",
                 replace="            if a < pb:"),
        ],
        # visible tests must say nothing about touching, so the flag's behaviour has to be
        # inferred from the signature and the rest of the module rather than read off a
        # failing assertion.
        visible_filter=lambda n: "touch" not in n and "join" not in n, max_visible=2),
    "no_tests": dict(
        task="cron_next", package="solution", exports=["cron_next"],
        move_to_util=["_BOUNDS", "_parse_field"],
        patches=[
            dict(file="util.py", find="    return vals, True", replace="    return vals, False"),
            dict(file="core.py",
                 find="        if dom_r and dow_r:\n            return dom_hit or dow_hit",
                 replace="        if dom_r and dow_r:\n            return dom_hit and dow_hit"),
        ],
        visible_filter=None, max_visible=0),
}

ISSUE_WITH_TESTS = """# Bug report

The test suite in this repository fails.

```
$ python -m pytest tests/ -q
{output}
```

Fix the source so that the suite passes.

Constraints:
- Do not modify anything under `tests/`.
- Do not add new dependencies; the standard library is all that is available.
- Keep the public API of the `{package}` package unchanged: {exports}.
"""

ISSUE_NO_TESTS = """# Bug report

This repository ships no tests. A user reports that `{exports}` returns wrong results
when the day-of-month and the day-of-week fields are both constrained, and when either
one is constrained on its own.

There is no failing assertion to work from. Read the code, decide what the correct
behaviour is, and make the source implement it.

Constraints:
- Do not add new dependencies; the standard library is all that is available.
- Keep the public API of the `{package}` package unchanged: {exports}.
- A hidden suite grades this work. Convincing yourself the code is right is the task.
"""


def build(name, spec):
    task = spec["task"]
    src = SRC / task
    hidden = (src / "tests.py").read_text()
    core0, util0 = B.split_module((src / "reference.py").read_text(), spec["move_to_util"])
    dest = PROBE / name
    repo = dest / "repo"
    seed = {"package": spec["package"], "exports": spec["exports"]}

    core_b, util_b = B.apply_patches(core0, util0, spec["patches"], f"[{name}]")

    B.write_repo(repo, seed, core_b, util_b, hidden, None)
    failed, _ = B.failing_tests(repo, hidden, spec["package"])
    if not failed:
        raise SystemExit(f"[{name}] patches do not make the hidden suite fail")

    if spec["max_visible"] == 0:
        visible = None
        shutil.rmtree(repo / "tests", ignore_errors=True)
        issue = ISSUE_NO_TESTS.format(package=spec["package"], exports=", ".join(spec["exports"]))
    else:
        cand = [t for t in failed if spec["visible_filter"](t)]
        if not cand:
            raise SystemExit(f"[{name}] no failing test satisfies the visible filter")
        visible = cand[:spec["max_visible"]]
        B.write_repo(repo, seed, core0, util0, hidden, visible)
        rc_clean, out = B.run_pytest(repo)
        if rc_clean != 0:
            raise SystemExit(f"[{name}] visible slice is not green on the reference:\n{out[-1500:]}")
        B.write_repo(repo, seed, core_b, util_b, hidden, visible)
        rc, vis = B.run_pytest(repo)
        if rc == 0:
            raise SystemExit(f"[{name}] visible suite passes on the bugged repo")
        issue = ISSUE_WITH_TESTS.format(output=B.trim_pytest(vis), package=spec["package"],
                                        exports=", ".join(spec["exports"]))

    (dest / "issue.md").write_text(issue)
    (dest / "tests_hidden.py").write_text(B.hidden_suite_for(spec["package"], hidden))
    (dest / "meta.json").write_text(json.dumps({
        "probe": name, "task": task, "package": spec["package"], "exports": spec["exports"],
        "n_sites": len(spec["patches"]), "moved_to_util": spec["move_to_util"],
        "hidden_failing": failed, "visible_tests": visible,
    }, indent=2) + "\n")
    print(f"  {name:12s} from {task:14s} sites={len(spec['patches'])} "
          f"hidden_fail={len(failed):2d} visible={len(visible) if visible else 'NONE'}")


if __name__ == "__main__":
    PROBE.mkdir(exist_ok=True)
    print(f"building {len(SHAPES)} probe fixtures -> {PROBE}")
    for n, s in SHAPES.items():
        build(n, s)
