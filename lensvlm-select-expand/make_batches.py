#!/usr/bin/env python3
"""Write runs/<run>/batch.md: the documents, questions and sheet paths one
subagent works through. Run name = <model>-r<ratio>-<half>."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
man = json.loads((HERE / "data" / "manifest.json").read_text())
HALVES = {"A": man[:10], "B": man[10:], "P": man[:2]}


def write(run, ratio, docs):
    d = HERE / "runs" / run
    d.mkdir(parents=True, exist_ok=True)
    lines = [f"# Batch {run}", ""]
    for m in docs:
        sheet = HERE / "data" / "sheets" / str(ratio) / f"{m['id']}.png"
        lines += [f"## {m['id']}", f"- sheet: {sheet}", f"- pages: P1..P{m['n_pages']}",
                  f"- question: {m['question']}", ""]
    (d / "batch.md").write_text("\n".join(lines))
    return d / "batch.md"


if __name__ == "__main__":
    for spec in sys.argv[1:]:          # e.g. opus-r10-A
        model, r, half = spec.split("-")
        print(write(spec, int(r[1:]), HALVES[half]))
