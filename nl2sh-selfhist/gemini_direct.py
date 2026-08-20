#!/usr/bin/env python3
"""Does a cloud generator need the retrieval tier at all?

The retrieval tier exists to feed documentation to a model too small to know the
commands. A frontier/flash model knows the head cold, so before pairing lexical
retrieval with a Gemini generator, measure whether Gemini needs the retrieval.

Same independent eval as `run_independent_eval.py` (real cyber commands, Gemini-
authored NL that does not name the utility), but the generator is Gemini itself,
given ONLY the request — no sources, no tldr, no distractors. If it routes well
here, retrieval is dead weight for the cloud path and its only remaining job is
the long tail and flag-grounding.

    python3 gemini_direct.py --model gemini-3.7-flash
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
GATE = HERE.parent / "nl2sh-retrieval"
PROMPT = """Translate this request into a single shell command. Output only the command, no explanation, no backticks.

Request: {nl}"""


def util(c: str) -> str:
    sys.path.insert(0, str(GATE))
    import pleias_gate as G
    return G.gold_utility(c)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemini-3.7-flash")
    ap.add_argument("--think", type=int, default=0)
    ap.add_argument("--nl", type=Path, default=HERE / "cyber_nl.json")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    sys.path.insert(0, str(GATE.parent / "gh-mcp-regex-fit"))
    from gemini_client import generate_many

    rows = [r for r in json.loads(a.nl.read_text()) if r.get("nl")]
    outs = generate_many([PROMPT.format(nl=r["nl"]) for r in rows],
                         model=a.model, thinking_budget=a.think, max_output_tokens=512)
    for r, o in zip(rows, outs):
        cmd = (o or "").strip().strip("`").split("\n")[0].strip()
        r["pred"] = cmd
        r["utility_ok"] = bool(cmd) and util(cmd) == r["utility"]

    clean = [r for r in rows if not r.get("names_utility")]
    n, nc = len(rows), len(clean)
    summary = {"model": a.model, "n": n, "n_leak_free": nc,
               "utility_acc_all": round(sum(r["utility_ok"] for r in rows) / n, 3),
               "utility_acc_leak_free": round(sum(r["utility_ok"] for r in clean) / nc, 3),
               "command_rate": round(sum(bool(r["pred"]) for r in rows) / n, 3)}
    dest = a.out or HERE / f"results_gemini_direct_{a.model.replace('.', '')}.json"
    dest.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1) + "\n")
    for r in rows[:10]:
        print(f"{'OK ' if r['utility_ok'] else '   '} {r['utility']:<10} {r['pred'][:52]}")
    print("\n" + json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
