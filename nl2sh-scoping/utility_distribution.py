#!/usr/bin/env python3
"""What shape is the natural-language-to-shell problem, before building anything?

Two architectures were on the table for a terminal helper — a cascade (rules
answer the head, model handles the tail) and retrieval-augmented generation
(deterministic shortlister narrows utilities and extracts parameters, docs go to
the model). They want opposite things from the distribution: a cascade needs a
*thick head*, RAG needs a *long tail the model does not already know*.

The intended measurement was the head/tail split of a real shell history. That
was unavailable — a week-old laptop — so this runs on NL2Bash instead
(12,607 pairs, scraped from forums and tutorials), which turns out to be the
more appropriate corpus anyway: a helper is asked about what you would have to
look up, not about what you type most, and `ls`/`cd`/`git status` dominate a
history while never reaching a helper.

One caveat the numbers cannot be read without: **60.3% of NL2Bash leads with
`find`**, an artifact of how the corpus was scraped. That makes it usable for
correctness evaluation and useless for frequency weighting, so everything here
is reported for the non-find subset as well, and the non-find numbers are the
ones to quote.

Parsing is deliberately crude — quoted spans are masked, then split on
`| || && ;` — so treat the stage counts as approximate. Flag counts are
distinct `-x`/`--xyz` tokens outside quotes.

    git clone --depth 1 https://github.com/TellinaTool/nl2bash.git
    python3 utility_distribution.py --data nl2bash/data/bash/all.cm
"""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPLIT = re.compile(r"\|\||&&|\||;")
FLAG = re.compile(r"(?<![\w-])(--?[A-Za-z][\w-]*)")
WRAPPERS = {"sudo", "time", "nohup", "command", "!"}


def _mask(cmd: str) -> str:
    """Blank quoted spans so separators and flags inside strings do not count."""
    return re.sub(r'"[^"]*"|\'[^\']*\'', '""', cmd)


def stages(cmd: str) -> list[str]:
    return [s.strip() for s in SPLIT.split(_mask(cmd)) if s.strip()]


def utility(segment: str) -> str:
    for tok in segment.split():
        if tok in WRAPPERS:
            continue
        if "=" in tok and not tok.startswith("-"):
            continue  # VAR=value prefix
        return tok.strip("()`$")
    return ""


def flags(cmd: str) -> set[str]:
    return set(FLAG.findall(_mask(cmd)))


def profile(cmds: list[str]) -> dict:
    n = len(cmds)
    lead = collections.Counter(utility(stages(c)[0]) for c in cmds if stages(c))
    stage_hist = collections.Counter(min(len(stages(c)), 5) for c in cmds)
    flag_hist = collections.Counter(min(len(flags(c)), 6) for c in cmds)
    return {
        "n": n,
        "distinct_lead_utilities": len(lead),
        "singleton_utilities": sum(1 for _, c in lead.items() if c == 1),
        "coverage_by_top_n": {str(k): round(sum(c for _, c in lead.most_common(k)) / n, 4)
                              for k in (5, 10, 20, 30, 50, 100, 200)},
        "stages": {("5+" if k == 5 else str(k)): round(v / n, 4)
                   for k, v in sorted(stage_hist.items())},
        "flags": {("6+" if k == 6 else str(k)): round(v / n, 4)
                  for k, v in sorted(flag_hist.items())},
        "top_utilities": [[u, c, round(c / n, 4)] for u, c in lead.most_common(15)],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True, help="NL2Bash all.cm")
    ap.add_argument("--out", type=Path, default=HERE / "results.json")
    a = ap.parse_args()

    cmds = [l.rstrip("\n") for l in a.data.read_text(encoding="utf-8",
                                                     errors="replace").splitlines() if l.strip()]
    out = {"source": "NL2Bash all.cm",
           "all": profile(cmds),
           "non_find": profile([c for c in cmds if not c.strip().startswith("find")])}
    a.out.write_text(json.dumps(out, indent=1) + "\n")

    for name in ("all", "non_find"):
        p = out[name]
        print(f"\n=== {name}  (n={p['n']}) ===")
        print(f"distinct leading utilities: {p['distinct_lead_utilities']}"
              f"  (appearing once: {p['singleton_utilities']})")
        print("coverage by top-N utilities: "
              + "  ".join(f"{k}:{v:.1%}" for k, v in p["coverage_by_top_n"].items()))
        print("flags per command:          "
              + "  ".join(f"{k}:{v:.1%}" for k, v in p["flags"].items()))
        print("pipeline stages:            "
              + "  ".join(f"{k}:{v:.1%}" for k, v in p["stages"].items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
