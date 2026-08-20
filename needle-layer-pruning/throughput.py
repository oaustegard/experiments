#!/usr/bin/env python3
"""Engine throughput at two depths, normalised by tokens rather than wall clock.

Wall-clock per turn is confounded here: a pruned model degrades, and a degraded
model emits a different number of tokens, so a turn can get slower for reasons
that have nothing to do with depth. The engine reports `prefill_tps` and
`decode_tps`, which are per-token rates and do not carry that confound.

    python3 throughput.py --weights /tmp/t_ctl.cact --label control
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
from eval import load_items  # noqa: E402
from needle_bsky.router import Router  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--label", required=True)
    a = ap.parse_args()

    items = load_items(NEEDLE_BSKY / "evalset.jsonl")
    r = Router(arm="tuned-min", threshold=0.0, weights=a.weights)
    pre, dec, lat = [], [], []
    for it in items:
        d = r.route(it["query"])
        lat.append(d.latency_ms)
        if d.prefill_tps:
            pre.append(d.prefill_tps)
        if d.decode_tps:
            dec.append(d.decode_tps)

    def med(xs):
        return round(statistics.median(xs), 1) if xs else None

    out = {"label": a.label, "n": len(items), "median_latency_ms": med(lat),
           "median_prefill_tps": med(pre), "median_decode_tps": med(dec),
           "n_prefill": len(pre), "n_decode": len(dec)}
    print(f"{a.label:12} latency {out['median_latency_ms']} ms   "
          f"prefill {out['median_prefill_tps']} tok/s   decode {out['median_decode_tps']} tok/s",
          flush=True)
    (HERE / f"throughput_{a.label}.json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
