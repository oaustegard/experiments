#!/usr/bin/env python3
"""Dump the family-A errors a Claude reviser is allowed to see, round by round.

`gh-mcp-regex-fit/compile_variants.py compile_iterated` runs the iterated arm
with Gemini as the reviser: score the current rules on family A, sample at most
120 errors, hand them back, take the rewritten list. That arm peaked at two
rounds and then lost 0.123 *in sample* at round three.

The Claude equivalent cannot be a single API call, because the reviser is the
session model. So this script reproduces exactly the half of the loop that is
code — scoring, error sampling, and the RNG draw order — and leaves the revision
itself to the model. Reproducing the RNG draw order matters: `compile_iterated`
creates one `random.Random(seed)` and samples from it once per round, so the
round-2 sample depends on how many numbers round 1 consumed. `--round N` here
replays rounds 1..N against the rules files that actually exist so the N-th
sample is the one that arm would have shown.

Only family A is ever read. Family B and wild are not opened by this file.

    python3 claude_iter_errors.py --round 1 --rules rules_claude-cleanroom.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--harness", type=Path, default=Path("../gh-mcp-regex-fit"),
                    help="the gh-mcp-regex-fit directory (relative to this script)")
    ap.add_argument("--rules", default="rules_claude-cleanroom.json",
                    help="rules file, relative to --harness, to score and dump errors for")
    ap.add_argument("--round", type=int, default=1,
                    help="which revision round this sample is for (replays the RNG)")
    ap.add_argument("--prior", nargs="*", default=[],
                    help="rules files used in rounds 1..round-1, in order, for RNG replay")
    ap.add_argument("--seed", type=int, default=20260819)
    ap.add_argument("--max-errors", type=int, default=120)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    here = Path(__file__).resolve().parent
    harness = (here / a.harness).resolve()
    sys.path.insert(0, str(harness))
    from eval import load_split, score            # noqa: E402
    from gemini_arms import CompiledRouter        # noqa: E402

    rows = [r for r in load_split(harness / "data" / "family_a.jsonl") if r.get("label")]
    rng = random.Random(a.seed)

    chain = list(a.prior) + [a.rules]
    if len(chain) != a.round:
        raise SystemExit(f"--round {a.round} needs {a.round - 1} --prior files, got {len(a.prior)}")

    shown = None
    summary = []
    for i, rf in enumerate(chain, start=1):
        s = score(CompiledRouter(harness / rf), rows)
        errs = s["errors"]
        shown = rng.sample(errs, min(a.max_errors, len(errs)))
        summary.append({"round": i, "rules": rf, "family_a_acc": s["label_acc"],
                        "n_rules": len(CompiledRouter(harness / rf).rules),
                        "n_errors": len(errs), "n_shown": len(shown)})

    for row in summary:
        print(f"round {row['round']:>2}  {row['rules']:<34} "
              f"acc {row['family_a_acc']:.4f}  {row['n_rules']:>3} rules  "
              f"{row['n_errors']:>3} errors  {row['n_shown']:>3} shown", file=sys.stderr)

    out = {"round": a.round, "rules": a.rules, "family_a_acc": summary[-1]["family_a_acc"],
           "n_errors": summary[-1]["n_errors"], "errors": shown}
    text = json.dumps(out, indent=1) + "\n"
    if a.out:
        (here / a.out if not a.out.is_absolute() else a.out).write_text(text)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
