#!/usr/bin/env python3
"""Export a `.cact` for one pruned depth, or for the unpruned control.

    python3 build_pruned.py --out /tmp/x.cact                  # control, no cut
    python3 build_pruned.py --start 12 --count 4 --out /tmp/y.cact

Exports at the checkpoint's own mixed-precision spec
(`embedding=4,mhc=4,default=2`, i.e. CQ2) so the control is as close to the
shipped blob as this path can get. The control is what every pruned arm is
compared against — never the standard engine weights, since the `weights=`
path also gives up the confidence head.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from prune import prune  # noqa: E402

CHECKPOINT = "checkpoints/needle2.pkl"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=None)
    ap.add_argument("--count", type=int, default=0)
    ap.add_argument("--out", required=True)
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
    info = write_export(params, cfg, a.out,
                        bits_map=cfg.weight_bits or None,
                        tokenizer=get_tokenizer(cfg.vocab_size),
                        kv_window=effective_kv_window(cfg))
    print(f"layers {cfg.num_layers}  engram {cfg.engram_layers}  "
          f"{info['bytes'] / 1e6:.2f} MB  {info['tensors']} tensors  "
          f"{time.perf_counter() - t0:.0f}s  -> {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
