#!/usr/bin/env python3
"""Time the same two depths at a five-tool catalogue.

`needle-bsky` measured that declaring a sixth tool costs 3.6x the per-turn
latency and then nothing more out to 18, because the contrastive retrieval head
is a fixed per-turn cost above five. If that fixed cost dominates an 18-tool
turn, a depth saving would be invisible there and should reappear at k=5, where
a turn is ~180 ms instead of ~1200 ms.

Same five-tool subsets, same seed and rule as `needle-bsky/oracle.py`, so the
only thing differing between the two arms is the number of layers.
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
sys.path.insert(0, str(HERE.parent))
from _lib.paths import experiment  # noqa: E402

NEEDLE_BSKY = experiment("needle-bsky")
sys.path.insert(0, str(NEEDLE_BSKY))
from eval import load_items  # noqa: E402
from oracle import subset_for  # noqa: E402
from needle_bsky.router import Router, load_schemas  # noqa: E402

SEED = 20260818


class _R(Router):
    def __init__(self, schemas, weights):
        import needle
        self.arm, self.threshold, self.schemas = "k5", 0.0, schemas
        self.agent = needle.Needle(tools=schemas, weights=weights)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--label", required=True)
    a = ap.parse_args()

    items = load_items(NEEDLE_BSKY / "evalset.jsonl")
    schemas = load_schemas("tuned-min")
    rng = random.Random(SEED)
    lat = []
    for it in items:
        sub = subset_for(it, schemas, rng)
        # Time route() only. Constructing the agent re-runs needle_init (the
        # engine holds one global session), which is not part of a turn.
        lat.append(_R(sub, a.weights).route(it["query"]).latency_ms)
    med = statistics.median(lat)
    print(f"{a.label:12} k=5  median {med:7.1f} ms  mean {statistics.mean(lat):7.1f} ms  n={len(lat)}",
          flush=True)
    (HERE / f"timing_k5_{a.label}.json").write_text(json.dumps(
        {"label": a.label, "weights": a.weights, "k": 5,
         "median_ms": round(med, 1), "mean_ms": round(statistics.mean(lat), 1)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
