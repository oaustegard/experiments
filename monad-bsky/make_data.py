#!/usr/bin/env python3
"""Build Monad training rows from the sibling experiment's generator.

`needle-bsky/gen_data.py` already writes (query, tools, answer, reasoning) rows
from templates whose entity pools are disjoint from the eval set. This reuses it
verbatim and re-renders each row into Monad's prompt format, with one change:
`k=18`, so every row declares the whole catalogue. Needle needed `k=5` because
its retrieval head renders five tools per turn; Monad has no retrieval stage and
the prose rendering of all 18 costs 574 tokens, so it sees everything.

    python3 make_data.py -n 800 --out data/train.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from _lib.paths import experiment

from monad_bsky.prompt import build_prompt, build_target


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=800)
    ap.add_argument("--seed", type=int, default=20260818)
    ap.add_argument("--arm", default="tuned-min")
    ap.add_argument("--out", default=str(HERE / "data" / "train.jsonl"))
    a = ap.parse_args()

    needle = experiment("needle-bsky")
    sys.path.insert(0, str(needle))
    from gen_data import build  # the same templates, same seed
    from needle_bsky.router import load_schemas

    schemas = load_schemas(a.arm)
    rows = build(a.n, a.seed, schemas, k=len(schemas))  # k=18: declare everything

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    with out.open("w") as fh:
        for r in rows:
            ans = r["answers"][0] if r["answers"] else None
            name = ans["name"] if ans else None
            args = ans["arguments"] if ans else {}
            fh.write(
                json.dumps(
                    {
                        "query": r["query"],
                        "prompt": build_prompt(r["tools"], r["query"]),
                        "target": build_target(r.get("reasoning"), name, args),
                        "name": name,
                    }
                )
                + "\n"
            )
            key = name or "(refuse)"
            counts[key] = counts.get(key, 0) + 1

    print(f"{len(rows)} rows -> {out}")
    for k in sorted(counts, key=lambda x: -counts[x]):
        print(f"  {k:22} {counts[k]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
