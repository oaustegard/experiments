#!/usr/bin/env python3
"""Split the error budget: is the retrieval head or the selector losing queries?

Needle renders at most five tools per turn. With 18 declared, a query can fail
two different ways — the right tool never entered the context (retrieval), or it
did and the model picked a neighbour (selection). The Python surface does not
expose which five were rendered, so this measures selection under *perfect*
retrieval instead: each query gets its own five-tool catalogue containing the
correct tool plus four deterministic distractors, which is exactly the regime
where retrieval cannot be the cause.

    oracle accuracy - full-catalogue accuracy = what retrieval costs

Off-topic items get five distractors and no correct tool, which is the same
refusal test at catalogue size five.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from eval import load_items, score_one, summarize
from needle_bsky.router import Router, load_schemas

K = 5  # Needle renders five tools per turn


class _OneShotRouter(Router):
    """A Router whose agent is rebuilt per query over a five-tool subset."""

    def __init__(self, schemas, threshold=0.0):
        import needle

        self.arm = "oracle"
        self.threshold = threshold
        self.schemas = schemas
        self.agent = needle.Needle(tools=schemas)


def subset_for(item: dict, all_schemas: list[dict], rng: random.Random) -> list[dict]:
    by_name = {s["name"]: s for s in all_schemas}
    want = item["tool"][0] if item["tool"] else None
    pool = [n for n in by_name if n != want]
    picks = rng.sample(pool, K - (1 if want else 0))
    names = ([want] if want else []) + picks
    rng.shuffle(names)
    return [by_name[n] for n in names]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="tuned-min", help="which schema wording to draw the five from")
    ap.add_argument("--seed", type=int, default=20260818)
    ap.add_argument("--evalset", default=str(HERE / "evalset.jsonl"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    items = load_items(Path(a.evalset))
    schemas = load_schemas(a.arm)
    rng = random.Random(a.seed)

    rows = []
    for it in items:
        sub = subset_for(it, schemas, rng)
        r = _OneShotRouter(sub)
        rows.append(score_one(it, r.route(it["query"])))

    res = {
        "arm": f"oracle-{a.arm}",
        "k": K,
        "seed": a.seed,
        "summary": summarize(rows),
        "rows": rows,
    }
    out = Path(a.out or HERE / f"results_oracle-{a.arm}.json")
    out.write_text(json.dumps(res, indent=1))
    s = res["summary"]
    print(
        f"oracle-{a.arm}: tool {s['tool_acc']:.3f}  routable {s['tool_acc_routable']:.3f}  "
        f"refuse {s['refusal_acc']:.3f}  args {s['args_acc_routable']:.3f}  "
        f"invented {s['invented_rate']:.3f}  median {s['median_latency_ms']:.0f}ms"
    )
    print(f"  mean routing latency at k=5: {statistics.mean(r['latency_ms'] for r in rows):.0f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
