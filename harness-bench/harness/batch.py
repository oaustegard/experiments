#!/usr/bin/env python3
"""Emit one batch brief per language: a subagent costs ~32.5k tokens before it
reads its prompt, so tasks are batched rather than dispatched one at a time
(claude-workspace docs/delegation.md)."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import bench

arm = sys.argv[1]
tasks = json.loads((bench.ROOT / "results/tasks-pilot.json").read_text())
only = set(sys.argv[2:]) or None
outdir = bench.ROOT / "briefs" / arm
outdir.mkdir(parents=True, exist_ok=True)

by_lang = {}
for t in tasks:
    key = f"{t['lang']}/{t['task']}"
    if only and key not in only:
        continue
    by_lang.setdefault(t["lang"], []).append(t["task"])

for lang, ts in by_lang.items():
    lines = [f"Solve {len(ts)} self-contained {lang} exercises. Each has its own "
             "prompt file; read it, then write the one file it names.\n"]
    for t in ts:
        wd = bench.ROOT / "work" / arm / lang / t
        lines.append(f"{len(lines)}. prompt: {bench.ROOT}/prompts/{arm}/{lang}__{t}.md\n"
                     f"   write:  {wd}/{bench.solution_files(lang, t)[0]}")
    lines.append(
        "\nRules for every exercise:\n"
        "- Read the prompt file, write the single solution file it names, move on.\n"
        "- Do NOT run tests, builds, linters or any other command. There is no test\n"
        "  file in the working directory; a hidden suite is run after you finish.\n"
        "- Do NOT create, delete or edit any other file.\n"
        "- Keep the names and signatures the stub declares.\n"
        "- Reply with one line per exercise: its name and DONE.")
    p = outdir / f"{lang}.md"
    p.write_text("\n".join(lines))
    print(p)
