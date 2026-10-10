"""Session totals across every run: spawns, turns, tool calls, cost, lines written, stub bodies, tests.

    python harness/tally.py        -> data/tally.json, summary on stdout

Lines written are '+' lines in each saved patch (data/runs/<run>/patches/*.diff).
Stub bodies are functions whose body is only a docstring and/or `pass` in the
library's stub commit, read from the bare mirror with git show and ast.parse
(nothing from the library is executed). Tests are graded passes per run
(results.jsonl, the last row per library).
"""
from __future__ import annotations

import ast
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, git, load_tasks, mirror  # noqa: E402

RUNS = DATA / "runs"
ARMS = {"solo": ["p-solo", "f-solo", "f-cont", "c-solo", "s-solo", "s-cont1", "s-cont2", "s-cont3"],
        "swarm": ["p-swarm", "p-fix", "f-swarm", "f-fix", "f-fix2", "f-fix3", "f-fix4", "s-swarm", "s-fix", "s-fix2"]}


def stub_bodies(task: dict) -> int:
    m = mirror(task["name"])
    files = git("--git-dir", str(m), "ls-tree", "-r", "--name-only", task["base_commit"], "--",
                task["src_dir"], check=False).split()
    n = 0
    for f in files:
        if not f.endswith(".py") or (task.get("patch_exclude") and "/tests/" in f):
            continue
        try:
            mod = ast.parse(git("--git-dir", str(m), "show", f"{task['base_commit']}:{f}", check=False))
        except SyntaxError:
            continue
        for node in ast.walk(mod):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                body = [b for b in node.body if not (isinstance(b, ast.Expr) and isinstance(b.value, ast.Constant)
                                                     and isinstance(b.value.value, str))]
                n += all(isinstance(b, ast.Pass) for b in body)
    return n


def added_lines(diff: Path) -> int:
    return sum(1 for l in diff.read_text(errors="replace").splitlines()
               if l.startswith("+") and not l.startswith("+++"))


def main():
    tasks = load_tasks()
    out = {"runs": {}}
    tools, tot = Counter(), Counter()
    libs = set()
    for run in sorted(p for p in RUNS.iterdir() if p.is_dir()):
        r = Counter()
        for line in (run / "costs.jsonl").read_text().splitlines() if (run / "costs.jsonl").exists() else []:
            c = json.loads(line)
            r["spawns"] += 1
            r["turns"] += c.get("turns", 0)
            r["usd"] += c.get("usd_input_side", 0)
            r["tool_calls"] += sum(c.get("tools", {}).values())
            tools.update(c.get("tools", {}))
        for d in sorted((run / "patches").glob("*.diff")) if (run / "patches").exists() else []:
            r["lines_written"] += added_lines(d)
        last = {}
        for line in (run / "results.jsonl").read_text().splitlines() if (run / "results.jsonl").exists() else []:
            g = json.loads(line)
            last[g["lib"]] = g
        r["tests_passed"] = sum(g["passed"] for g in last.values())
        libs.update(last)
        out["runs"][run.name] = {k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()}
        tot.update(r)
    # Lines written: '+' lines of each (arm, library)'s final patch. A later run's patch
    # holds everything before it (fixers and continuations work on the same tree), so
    # summing every run's patches would count most lines twice.
    final = {}
    for arm, order in ARMS.items():
        for run in order:
            for d in sorted((RUNS / run / "patches").glob("*.diff")) if (RUNS / run / "patches").exists() else []:
                final[(arm, d.stem)] = d
    lines = {f"{arm}/{lib}": added_lines(d) for (arm, lib), d in sorted(final.items())}
    out["final_lines"] = lines
    bodies = {}
    for lib in sorted(libs):
        if lib in tasks:
            bodies[lib] = stub_bodies(tasks[lib])
    out["stub_bodies"] = bodies
    tot.pop("lines_written", None)
    tot.pop("tests_passed", None)
    out["totals"] = {**{k: round(v, 2) if isinstance(v, float) else v for k, v in tot.items()},
                     "lines_written_final": sum(lines.values()),
                     "libraries": len(libs), "stub_bodies": sum(bodies.values()), "tools": dict(tools)}
    (DATA / "tally.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out["totals"], indent=1))
    for k, v in out["runs"].items():
        print(f"{k:10s} {v}")


if __name__ == "__main__":
    main()
