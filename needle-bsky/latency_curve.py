#!/usr/bin/env python3
"""Per-turn latency against the number of declared tools.

Needle renders at most five tools per turn; above five a retrieval head runs.
This measures what that costs by padding one agent's catalogue while holding the
five tools the probe queries actually need.

    python3 latency_curve.py       # writes results_latency_vs_catalogue.json

Run it on an otherwise idle box. On four cores a concurrent trainer moves every
number here by an order of magnitude (ERRORS.md #6).
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from needle_bsky.router import load_schemas

SIZES = (5, 6, 8, 12, 18)
CORE = ["get_user_posts", "search_posts", "get_followers", "get_trending_topics", "get_likes"]
QUERIES = [
    "grab danabra.mov's timeline",
    "dig up posts on rust compiler",
    "who follows mackuba.eu",
    "name the current topics, nothing else",
    "pull the likers of https://bsky.app/profile/hailey.at/post/3kzzrmfw2ok2b",
]
REPEATS = 4


def main() -> int:
    import needle

    schemas = load_schemas("tuned-min")
    by = {s["name"]: s for s in schemas}
    rest = [n for n in by if n not in CORE]

    out = {}
    detail = {}
    for k in SIZES:
        names = CORE + rest[: k - len(CORE)]
        agent = needle.Needle(tools=[by[n] for n in names])
        lat = []
        for q in QUERIES * REPEATS:
            agent.reset()
            t = time.perf_counter()
            agent.complete(q)
            lat.append((time.perf_counter() - t) * 1000)
        lat = lat[1:]  # the first call pays the lazy engine bind
        out[k] = round(statistics.median(lat), 1)
        detail[k] = {
            "median_ms": out[k],
            "p90_ms": round(sorted(lat)[int(0.9 * len(lat))], 1),
            "n": len(lat),
        }
        print(f"n_tools={k:2d}  median {out[k]:7.1f} ms  p90 {detail[k]['p90_ms']:7.1f} ms")

    (HERE / "results_latency_vs_catalogue.json").write_text(json.dumps(out, indent=1))
    (HERE / "results_latency_detail.json").write_text(json.dumps(detail, indent=1))
    print(f"sixth-tool ratio: {out[6] / out[5]:.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
