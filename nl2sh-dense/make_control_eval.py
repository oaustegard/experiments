#!/usr/bin/env python3
"""A human-authored query set, in the eval format, as the control for §7.

The cyber eval's natural language was written by `gemini-3.7-flash`. `enrich.py`
has `gemini-3.5-flash-lite` write the corpus's query vocabulary. If recall rises
on that eval, two readings fit: the corpus got better, or two members of one
model family converged on phrasing. This repo has already paid 0.3 accuracy for
not separating those.

NL2Bash's English was written by human annotators, so it separates them: a lift
that survives here is not family alignment. Its own defects are known and are
why it is a control rather than a headline — 50.3% of this sample's gold
utilities are `find`, and the annotators wrote the English while looking at the
command, so 34.7% of prompts name the utility outright. Both are handled: rows
naming their gold utility are dropped, and the `find` share is reported beside
every number taken from this file.

Sampling is capped per utility, because NL2Bash's raw order is `find` all the way
down and a control whose constant prior is 0.5 measures the prior. With
`--exclude-adapter-rows`, rows the adapter trained on are dropped too — needed
only when an adapter is in the pipeline, and it costs most of the non-`find`
pool, so the default is off.

    python3 make_control_eval.py --nl2bash <nl2bash>/data/bash --n 300
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import dense_index as D  # noqa: E402
import queries as Q  # noqa: E402
import retrieve as R  # noqa: E402
import adapter as A  # noqa: E402
from eval_dense import tldr_from_chunks  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nl2bash", type=Path, required=True)
    ap.add_argument("--chunks", type=Path, default=D.DEFAULT_CHUNKS)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--per-utility", type=int, default=3,
                    help="max rows per gold utility, to defuse the find skew")
    ap.add_argument("--exclude-adapter-rows", action="store_true")
    ap.add_argument("--adapter-cap", type=int, default=200,
                    help="the adapter's per-utility cap, to reproduce its train set")
    ap.add_argument("--seed", type=int, default=20260820)
    ap.add_argument("--out", type=Path, default=HERE / "nl2bash_control.json")
    a = ap.parse_args()

    chunks = R.load_chunks(a.chunks)
    tldr = tldr_from_chunks(chunks)
    utilities = {c.utility for c in D.page_chunks(chunks)}
    trained_on = set()
    if a.exclude_adapter_rows:
        trained_on = {nl for nl, _ in A.training_pairs(a.nl2bash, utilities,
                                                       a.adapter_cap, a.seed, 100000)}

    pool = [r for r in Q.nl2bash(a.nl2bash, n=10 ** 6, seed=a.seed, tldr=tldr)
            if not r["names_utility"] and r["nl"] not in trained_on]
    seen: Counter = Counter()
    rows = []
    for r in pool:
        if seen[r["utility"]] >= a.per_utility:
            continue
        seen[r["utility"]] += 1
        rows.append(r)
        if len(rows) >= a.n:
            break
    counts = Counter(r["utility"] for r in rows)
    top, k = counts.most_common(1)[0]
    a.out.write_text(json.dumps(rows, indent=1) + "\n")
    print(f"wrote {len(rows)} human-authored rows over {len(counts)} utilities; "
          f"constant prior {top} {k / len(rows):.3f}; "
          f"{len(trained_on)} adapter-training rows excluded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
