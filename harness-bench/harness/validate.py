#!/usr/bin/env python3
"""Grader invariants: the reference solution must score 12/12, the stub 0/12.
A grader that cannot go red is not a grader."""
import json, shutil, subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import bench

ROOT = bench.ROOT
tasks = json.loads((ROOT / "results/tasks-pilot.json").read_text())

for arm, use_example in (("_gold", True), ("_stub", False)):
    subprocess.run([sys.executable, "harness/bench.py", "prepare",
                    "--tasks", "results/tasks-pilot.json", "--arm", arm],
                   cwd=ROOT, check=True, capture_output=True)
    if use_example:
        for t in tasks:
            lang, task = t["lang"], t["task"]
            ex = bench.ex_dir(lang, task)
            sol = bench.solution_files(lang, task)[0]
            cand = list((ex / ".meta").glob("example*")) + list((ex / ".meta").glob("*/example*"))
            cand = [c for c in cand if c.is_file()]
            assert cand, f"no example for {lang}/{task}"
            shutil.copy2(cand[0], ROOT / "work" / arm / lang / task / sol)
    subprocess.run([sys.executable, "harness/bench.py", "grade",
                    "--tasks", "results/tasks-pilot.json", "--arm", arm,
                    "--out", f"results/{arm}.json"], cwd=ROOT, check=True)
