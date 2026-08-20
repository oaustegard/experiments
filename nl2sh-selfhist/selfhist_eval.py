#!/usr/bin/env python3
"""An out-of-distribution probe built from this session's own shell history.

Oskar's suggestion: stop using NL2Bash, whose 60% `find` skew distorted two
separate measurements, and use the hundreds of commands actually issued tonight
instead. The idea is sound and **this particular history does not support it**,
which is itself the finding:

    289 Bash calls issued
     66 single-line and self-contained (the rest are heredocs and pipelines)
     26 general-shell (the rest invoke this project's own scripts)
     ~15 of those 26 are variants of "print lines X to Y of file F"

An agent's shell history is dominated by *file slicing* and *project-script
invocation*, not general utility composition. Its utility distribution is far
flatter than NL2Bash's — 22 distinct leading utilities with the commonest at
24.2%, against `find` at 60.3% — but its *task* diversity is nearly nil, so it
cannot replace a benchmark. The `description` field that would have given real
paired natural language is not persisted in the transcript either; only the
commands survive.

What it can do is probe generalisation. The gate model was fine-tuned on
NL2Bash phrasing about `find`, `grep` and `chmod`; these are the shapes a coding
agent actually reaches for, rewritten against `funceq.py`'s fixture so they
execute. Small (n=14), hand-authored, and a probe rather than a benchmark.

    python3 selfhist_eval.py --model ../nl2sh-retrieval/ft
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
GATE = HERE.parent / "nl2sh-retrieval"

# (request, gold command) — task shapes taken from this session's real history,
# rewritten to run inside the funceq fixture.
PAIRS = [
    ("show lines 1 through 5 of notes.txt", "sed -n '1,5p' notes.txt"),
    ("print the first two lines of notes.txt", "head -2 notes.txt"),
    ("print the last line of notes.txt", "tail -1 notes.txt"),
    ("how many lines are in notes.txt", "wc -l notes.txt"),
    ("show me the whole of docs/readme.md", "cat docs/readme.md"),
    ("list every file under src including subdirectories", "find src -type f"),
    ("which files mention alpha", "grep -rl alpha ."),
    ("show the line numbers where alpha appears in notes.txt", "grep -n alpha notes.txt"),
    ("count the files in the current directory", "ls -1 | wc -l"),
    ("show sizes of everything here in human readable form", "ls -lh"),
    ("find the pdf files", "find . -name '*.pdf'"),
    ("which files are bigger than 100 kilobytes", "find . -size +100k"),
    ("show the headings in docs/readme.md", "grep '^#' docs/readme.md"),
    ("list the c source files under src", "find src -name '*.c'"),
]

PREFILL = ("<|language_start|>\nEnglish\n<|language_end|>\n"
           "<|query_report_start|>\nTrivial\n<|query_report_end|>\n<|answer_start|>\n")
CMDLINE = re.compile(r"`([^`\n]{2,200})`|^\s*([a-z][a-z0-9_.+-]{1,20}\s+[^\n]{2,200})$", re.M)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tldr", type=Path, required=True)
    ap.add_argument("--distractors", type=int, default=2)
    ap.add_argument("--max-new-tokens", type=int, default=140)
    ap.add_argument("--out", type=Path, default=HERE / "results_selfhist.json")
    a = ap.parse_args()

    sys.path.insert(0, str(GATE))
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import pleias_gate as G

    tldr = G.load_tldr(a.tldr)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).eval()

    import random
    rng = random.Random(20260819)
    others = [u for u in tldr if len(tldr[u]) >= 1]
    rows = []
    for nl, gold in PAIRS:
        gu = G.gold_utility(gold)
        picks = [gu] + rng.sample([u for u in others if u != gu], a.distractors)
        rng.shuffle(picks)
        srcs = [f"{u} — {d}: {c}" for u in picks if u in tldr for d, c in tldr[u][:1]]
        ids = tok(G.build_prompt(nl, srcs) + PREFILL, return_tensors="pt")
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=a.max_new_tokens,
                                 do_sample=False, pad_token_id=tok.pad_token_id)
        gen = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=False)
        ans = re.split(r"<\|answer_end\|>", gen)[0].strip()
        m = CMDLINE.search(ans)
        cand = (m.group(1) or m.group(2) or "").strip() if m else ""
        rows.append({"nl": nl, "gold_cmd": gold, "gold_utility": gu, "command": cand,
                     "utility_ok": bool(cand) and G.gold_utility(cand) == gu,
                     "answer": ans[:300], "seconds": round(time.perf_counter() - t0, 1)})
        print(f"{'OK ' if rows[-1]['utility_ok'] else '   '} {nl[:44]:<46} -> {cand[:52]!r}")

    n = len(rows)
    summary = {"model": a.model, "n": n,
               "utility_acc": round(sum(r["utility_ok"] for r in rows) / n, 3),
               "command_rate": round(sum(bool(r["command"]) for r in rows) / n, 3)}
    a.out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1) + "\n")
    print("\n" + json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
