#!/usr/bin/env python3
"""Independent natural language for real commands — the eval this thread lacked.

Every eval in `gh-mcp-regex-fit` and `nl2sh-retrieval` had one side authored by
the model under test, or by its sibling. The cyber-training corpus fixes the
command side (real users) but ships no natural language. This closes the gap
with the one independent model available: **Gemini writes the request for each
real command**, so neither side of the resulting NL->command eval came from
Claude, and the command distribution is real rather than templated.

The prompt asks Gemini to describe what the command accomplishes in the words a
user would type asking for it — not to transcribe the syntax — and forbids
naming the utility, so the eval does not leak its own answer (the flaw the
retrieval verification pass found in NL2Bash, where 34.7% of prompts named the
gold utility).

    python3 gen_nl.py --sample cyber_sample.json --out cyber_nl.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
GATE = HERE.parent / "gh-mcp-regex-fit"

PROMPT = """A user typed this shell command:

    {cmd}

Write the one-sentence request, in plain English, that this user would have typed \
into an AI terminal helper to get this command. Describe the goal, the way a person \
who did not know the exact command would phrase it.

Rules:
- Do NOT name the command or any flag. Describe the intent, not the syntax.
- Keep any literal argument values (filenames, hosts, numbers) that a user would \
naturally include.
- One sentence, imperative, no quotes around the whole thing.

Output only the sentence."""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=Path, required=True)
    ap.add_argument("--model", default="gemini-3.7-flash")
    ap.add_argument("--out", type=Path, default=HERE / "cyber_nl.json")
    a = ap.parse_args()

    sys.path.insert(0, str(GATE))
    from gemini_client import generate_many

    rows = json.loads(a.sample.read_text())
    prompts = [PROMPT.format(cmd=r["cmd"]) for r in rows]
    outs = generate_many(prompts, model=a.model, thinking_budget=0, max_output_tokens=512)
    n_leak = 0
    for r, nl in zip(rows, outs):
        nl = (nl or "").strip().strip('"').split("\n")[0]
        r["nl"] = nl
        # leak check: did Gemini name the utility despite instruction?
        r["names_utility"] = bool(nl) and re.search(rf"\b{re.escape(r['utility'])}\b", nl, re.I) is not None
        n_leak += r["names_utility"]
    a.out.write_text(json.dumps(rows, indent=1) + "\n")
    ok = [r for r in rows if r.get("nl")]
    print(f"generated {len(ok)}/{len(rows)} NL descriptions; {n_leak} name the utility")
    for r in ok[:8]:
        flag = " [LEAK]" if r["names_utility"] else ""
        print(f"  {r['utility']:<10} {r['nl'][:78]}{flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
