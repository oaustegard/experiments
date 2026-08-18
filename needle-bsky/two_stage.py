#!/usr/bin/env python3
"""Two-stage routing: a five-way group choice, then a five-tool agent.

Two measurements from the flat arms motivate this. Declaring a sixth tool costs
a fixed ~750 ms per turn, and giving the model a five-tool catalogue that
contains the right answer is worth +11 to +17 points of routable accuracy. Both
say the same thing: keep every agent at five tools.

So route in two turns. Stage 1 declares five *group* tools, each of which just
names a family of reads and takes the query through unchanged. Stage 2 declares
the ≤5 real tools in the chosen group. Neither agent is ever above the
retrieval threshold, so neither pays the retrieval cost, and stage 2 sees a
catalogue small enough to be in the oracle regime.

    python3 two_stage.py --stage1 heuristic   # writes results_two_stage_heuristic.json
    python3 two_stage.py --stage1 needle      # the model-based stage 1, for contrast
    python3 two_stage.py --arm tuned          # which wording the leaf tools use

The groups, the heuristic and the router itself live in `needle_bsky/grouped.py`
so the CLI can use the same code path; this script is the eval harness over it.

Cost of a wrong group is total: the right tool is not in stage 2's catalogue at
all, so a group error is unrecoverable by construction. The group table below
therefore reports stage-1 accuracy separately.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from eval import load_items, score_one, summarize
from needle_bsky.grouped import GROUP_OF, GROUPS, GroupedRouter

K_MAX = 5  # no group may exceed the retrieval threshold
assert all(len(v) <= K_MAX for v in GROUPS.values())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="tuned-min")
    ap.add_argument("--stage1", default="needle", choices=["needle", "heuristic"])
    ap.add_argument("--evalset", default=str(HERE / "evalset.jsonl"))
    a = ap.parse_args()

    items = load_items(Path(a.evalset))
    r = GroupedRouter(a.arm, threshold=0.0, stage1=a.stage1)

    rows, group_rows = [], []
    for it in items:
        d = r.route(it["query"])
        group = (d.raw or {}).get("group")
        row = score_one(it, d)
        want = GROUP_OF.get(it["tool"][0]) if it["tool"] else None
        row["group"] = group
        row["group_expected"] = want
        row["group_ok"] = (group == want) if want else (group is None)
        rows.append(row)
        group_rows.append(row)

    routable = [x for x in rows if x["expected"]]
    res = {
        "arm": f"two-stage-{a.stage1}-{a.arm}",
        "stage1": a.stage1,
        "groups": GROUPS,
        "summary": summarize(rows),
        "stage1_group_acc_routable": round(sum(x["group_ok"] for x in routable) / len(routable), 4),
        "stage1_refused_off_topic": round(
            sum(1 for x in rows if not x["expected"] and x["group"] is None) / max(1, len(rows) - len(routable)), 4
        ),
        "rows": rows,
    }
    (HERE / f"results_two_stage_{a.stage1}.json").write_text(json.dumps(res, indent=1))
    s = res["summary"]
    print(
        f"two-stage[{a.stage1}]-{a.arm}: tool {s['tool_acc']:.3f}  routable {s['tool_acc_routable']:.3f}  "
        f"refuse {s['refusal_acc']:.3f}  args {s['args_acc_routable']:.3f}  "
        f"invented {s['invented_rate']:.3f}  median {s['median_latency_ms']:.0f}ms"
    )
    print(f"  stage-1 group accuracy on routable queries: {res['stage1_group_acc_routable']:.3f}")
    print(f"  mean two-turn latency: {statistics.mean(x['latency_ms'] for x in rows):.0f} ms")
    bad = [(x["id"], x["group_expected"], x["group"]) for x in routable if not x["group_ok"]]
    print(f"  group errors ({len(bad)}): {bad[:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
