#!/usr/bin/env python3
"""Is the clean-room gap the model, or the number of calls it got?

`compiled-claude-cleanroom` scores 0.540 on wild against `compiled-gemini-
cleanroom`'s 0.176, and it is tempting to read that as compiler capability. The
two arms did not run the same procedure, though. Claude ran an agent loop —
read the spec, wrote a generator script, validated, smoke-tested — and emitted
154 rules, about two per target. Gemini got one `generateContent` call and
emitted 78, about one per target, with all 79 targets competing for a single
output budget.

This arm removes that confound and nothing else: the same clean-room prompt,
the same model, the same executor, but the catalogue is split into chunks of
`--chunk` targets and each chunk gets its own call. Per-chunk output pressure
drops by ~10x. Rules are concatenated in catalogue order, then the model is
given one ordering pass over its own concatenated list, because chunking
destroys the global specific-before-general ordering the executor depends on.

If chunked Gemini closes the gap, the difference was budget and procedure. If
it does not, it is the model.

    python3 chunked_arm.py --chunk 10
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from arms import labels
from compile_variants import _ask, _write
from cues import CUE_NAMES
from gemini_arms import COMPILE_PROMPT, render_catalogue

HERE = Path(__file__).resolve().parent

CHUNK_NOTE = """

## Scope of THIS call

You are writing rules for only these {n} targets, which are a subset of the \
catalogue above. Ignore every other target — another call covers those. Do not \
emit a rule whose label is outside this list:

{subset}

Because you are covering few targets, spend the budget on them: enumerate the \
surface forms generously, and write more than one rule per target where a \
target is reached by genuinely different phrasings."""

ORDER_PROMPT = """These regex routing rules were written in independent batches, so their \
relative order is arbitrary. Rules execute in order and the first match wins, so a \
general rule sitting above a specific one silently blocks it.

Reorder the list. Do not add, delete, reword or re-label anything — return exactly \
the same rule objects, permuted so that specific patterns precede general ones and \
a rule that could swallow another's requests sits below it.

{rules}

Output ONLY the reordered JSON array."""


def chunk_targets(n: int) -> list[list[str]]:
    labs = labels()
    return [labs[i:i + n] for i in range(0, len(labs), n)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--chunk", type=int, default=10)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--no-order-pass", action="store_true")
    a = ap.parse_args()
    from gemini_client import DEFAULT_MODEL
    model = a.model or DEFAULT_MODEL

    base = COMPILE_PROMPT.format(n_targets=len(labels()), catalogue=render_catalogue(),
                                 cue_list="\n".join(f"- {c}" for c in CUE_NAMES))
    groups = chunk_targets(a.chunk)
    allowed = set(labels())
    out: list[dict] = []
    for i, g in enumerate(groups, 1):
        prompt = base + CHUNK_NOTE.format(n=len(g), subset="\n".join(f"- {x}" for x in g))
        got = _ask(prompt, model)
        keep = [r for r in got if isinstance(r, dict) and r.get("label") in set(g)]
        out += keep
        print(f"  chunk {i}/{len(groups)} ({len(g)} targets): {len(got)} returned, {len(keep)} in scope")

    if not a.no_order_pass:
        ordered = _ask(ORDER_PROMPT.format(rules=json.dumps(out, indent=0)), model)
        # An ordering pass must permute, not rewrite. Fall back if it did not.
        same = (len(ordered) == len(out)
                and sorted(json.dumps(r, sort_keys=True) for r in ordered if isinstance(r, dict))
                == sorted(json.dumps(r, sort_keys=True) for r in out))
        print(f"  ordering pass: {len(ordered)} rules, permutation-only = {same}"
              + ("" if same else " -> keeping the unordered list"))
        if same:
            out = ordered

    _write(out, model, a.tag or "gemini-chunked",
           {"supervision": f"none (clean room, {len(groups)} chunks of {a.chunk} targets)",
            "n_chunks": len(groups), "chunk_size": a.chunk,
            "order_pass": not a.no_order_pass,
            "prompt_sha": hashlib.sha256(base.encode()).hexdigest()[:16]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
