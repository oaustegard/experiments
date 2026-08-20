#!/usr/bin/env python3
"""Ask the engine what it actually costs, rather than deriving it.

`needle`'s completion result carries `peak_ram_mb`, which `needle_bsky.router`
already threads onto every Decision and `eval.py` then drops. For a memory
budget that field is the whole question, so this runs a handful of real routing
turns and reports it alongside the process RSS.

    python3 rss_probe.py --weights /tmp/x.cact --label all2
"""

from __future__ import annotations

import argparse
import json
import resource
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from _lib.paths import experiment  # noqa: E402

NEEDLE_BSKY = experiment("needle-bsky")
sys.path.insert(0, str(NEEDLE_BSKY))
from eval import load_items  # noqa: E402
from needle_bsky.router import Router  # noqa: E402


def rss_mb() -> float:
    # ru_maxrss is KiB on Linux.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=None, help="omit for the stock engine weights")
    ap.add_argument("--label", required=True)
    ap.add_argument("--n", type=int, default=12)
    a = ap.parse_args()

    items = load_items(NEEDLE_BSKY / "evalset.jsonl")[: a.n]
    before = rss_mb()
    r = Router(arm="tuned-min", threshold=0.0, weights=a.weights)
    peaks, after_first = [], None
    for it in items:
        d = r.route(it["query"])
        if d.peak_ram_mb is not None:
            peaks.append(d.peak_ram_mb)
        if after_first is None:
            after_first = rss_mb()

    out = {
        "label": a.label, "weights": a.weights, "turns": len(items),
        "engine_peak_ram_mb": round(statistics.median(peaks), 2) if peaks else None,
        "engine_peak_ram_max": round(max(peaks), 2) if peaks else None,
        "rss_before_mb": round(before, 1),
        "rss_after_first_turn_mb": round(after_first, 1),
        "rss_final_mb": round(rss_mb(), 1),
    }
    print(json.dumps(out), flush=True)
    (HERE / f"rss_{a.label}.json").write_text(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
