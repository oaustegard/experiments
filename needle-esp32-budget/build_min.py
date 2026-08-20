#!/usr/bin/env python3
"""Export the smallest configuration the three sibling experiments justify.

`needle-layer-pruning` found exactly one survivable cut, `[9,13)`;
`needle-quantization` found `default=2` free. This composes them — 23 layers at
2 bits — and is the floor of everything measured in this series.

    python3 build_min.py --out /tmp/min.cact
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from _lib.paths import experiment  # noqa: E402

sys.path.insert(0, str(experiment("needle-layer-pruning")))
from prune import prune  # noqa: E402

CHECKPOINT = "checkpoints/needle2.pkl"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", type=int, default=9)
    ap.add_argument("--count", type=int, default=4)
    ap.add_argument("--spec", default="default=2")
    ap.add_argument("--checkpoint", default=CHECKPOINT)
    a = ap.parse_args()

    from needle.model.run import load_checkpoint
    from needle.model.export import write_export
    from needle.model.architecture import effective_kv_window
    from needle.model.tokenizer import get_tokenizer

    params, cfg = load_checkpoint(a.checkpoint)
    if a.count:
        params, cfg = prune(params, cfg, a.start, a.count)

    t0 = time.perf_counter()
    info = write_export(params, cfg, a.out, bits_map=a.spec,
                        tokenizer=get_tokenizer(cfg.vocab_size),
                        kv_window=effective_kv_window(cfg))
    print(json.dumps({"layers": cfg.num_layers, "engram": list(cfg.engram_layers),
                      "spec": a.spec, "bytes": info["bytes"],
                      "mb": round(info["bytes"] / 1e6, 2),
                      "export_seconds": round(time.perf_counter() - t0, 1)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
