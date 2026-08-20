#!/usr/bin/env python3
"""Score one exported `.cact` on `needle-bsky/evalset.jsonl`.

    python3 eval_pruned.py --weights /tmp/ctl.cact --label control

One process per weights file, because the engine cannot unload a loaded `.cact`
(`needle-bsky/RESULTS.md`). Scoring and schemas are `needle-bsky`'s, imported
unchanged, at the `tuned-min` arm — the same configuration `needle-tool-naming`
used for its `canon` control, so the numbers line up across all three
experiments.

Confidence is `None` on any `weights=` path, so the gate is out of scope here
and the metric is routing accuracy.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from _lib.paths import experiment  # noqa: E402

NEEDLE_BSKY = experiment("needle-bsky")
sys.path.insert(0, str(NEEDLE_BSKY))
from eval import load_items, score_one, summarize  # noqa: E402
from needle_bsky.router import Router  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--arm", default="tuned-min")
    ap.add_argument("--evalset", default=str(NEEDLE_BSKY / "evalset.jsonl"))
    ap.add_argument("--cut", default="none", help="recorded verbatim in the result file")
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--engram", default=None)
    ap.add_argument("--out-dir", default=str(HERE))
    a = ap.parse_args()

    items = load_items(Path(a.evalset))
    r = Router(arm=a.arm, threshold=0.0, weights=a.weights)
    rows = [score_one(it, r.route(it["query"])) for it in items]

    res = {
        "label": a.label,
        "cut": a.cut,
        "layers": a.layers,
        "engram_layers": a.engram,
        "arm": a.arm,
        "weights": a.weights,
        "summary": summarize(rows),
        "rows": rows,
    }
    Path(a.out_dir, f"results_{a.label}.json").write_text(json.dumps(res, indent=1))
    s = res["summary"]
    def num(key):
        # refusal_acc is None on an evalset with no off-topic items.
        return f"{s[key]:.3f}" if s[key] is not None else "  n/a"

    print(f"{a.label:14} layers {str(a.layers):>3}  engram {str(a.engram):<8} "
          f"routable {num('tool_acc_routable')}  tool {num('tool_acc')}  "
          f"refuse {num('refusal_acc')}  args {num('args_acc_routable')}  "
          f"median {statistics.median(x['latency_ms'] for x in rows):.0f}ms", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
