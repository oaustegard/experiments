#!/usr/bin/env python3
"""Is the clean-room rule set narrow because the model cannot do better, or
because nobody asked it to be broad?

Every compiled arm so far fails the same way: high precision, low coverage —
65-71% of its errors are abstentions, not wrong labels. That has two very
different explanations. Either an independent compiler genuinely cannot
anticipate the phrasings it has never seen, or the clean-room prompt quietly
optimises for precision ("omitting a target is better than a rule you do not
believe in") and the model is obeying.

This arm changes exactly one thing: the instruction. Same catalogue, same cues,
same executor, same splits, no supervision. If coverage moves, the deficit was
the prompt.

    python3 breadth_arm.py --model gemini-3.7-flash
"""

from __future__ import annotations

import argparse
import hashlib

from compile_variants import _ask, _write
from gemini_arms import COMPILE_PROMPT, render_catalogue
from arms import labels
from cues import CUE_NAMES

BREADTH = COMPILE_PROMPT.replace(
    """- Cover the targets you believe requests will actually hit. Omitting a target is \
better than a rule you do not believe in, because a wrong rule earlier in the \
order blocks every rule after it.""",
    """- Cover EVERY target. A missing rule means the router silently declines a \
request it should have served, which is the most expensive failure here.
- Write BROAD patterns. Assume the request is phrased conversationally by \
someone who has never read the tool names: "cut a branch called next on X", \
"show every team I'm in", "read me the chatter on #1980". A pattern that only \
matches the schema's own vocabulary will match almost nothing real. Enumerate \
five to ten surface forms per target, including verb-less and elliptical ones.
- Order still matters, so put the specific rules first — but do not buy \
precision by leaving targets uncovered.""")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    from gemini_client import DEFAULT_MODEL
    model = a.model or DEFAULT_MODEL
    prompt = BREADTH.format(n_targets=len(labels()), catalogue=render_catalogue(),
                            cue_list="\n".join(f"- {c}" for c in CUE_NAMES))
    assert prompt != COMPILE_PROMPT, "breadth substitution did not apply"
    _write(_ask(prompt, model), model, a.tag or "gemini-breadth",
           {"supervision": "none (clean room, breadth-instructed)",
            "prompt_sha": hashlib.sha256(prompt.encode()).hexdigest()[:16],
            "prompt_chars": len(prompt)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
