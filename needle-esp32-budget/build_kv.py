#!/usr/bin/env python3
"""Export one `.cact` at a given KV window (and optionally a weight-bit spec).

    python3 build_kv.py --kv-window 160 --out /tmp/x.cact

`kv_window` is baked into the blob header at export and the engine sizes its
cache from it, so this is a real runtime-memory knob rather than a hint. The
checkpoint pins 256; `KV_WINDOW_MIN` is 160 and `kv_budget_window()` would allow
704 under Cactus's own 11.5 MiB KV allowance.
"""

from __future__ import annotations

import argparse
import json
import time

CHECKPOINT = "checkpoints/needle2.pkl"
SHIPPED_BITS = "embedding=4,mhc=4,default=2"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kv-window", type=int, required=True)
    ap.add_argument("--spec", default=SHIPPED_BITS)
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoint", default=CHECKPOINT)
    a = ap.parse_args()

    from needle.model.run import load_checkpoint
    from needle.model.export import write_export
    from needle.model.tokenizer import get_tokenizer

    params, cfg = load_checkpoint(a.checkpoint)
    t0 = time.perf_counter()
    info = write_export(params, cfg, a.out, bits_map=a.spec,
                        tokenizer=get_tokenizer(cfg.vocab_size),
                        kv_window=a.kv_window)
    print(json.dumps({"kv_window": a.kv_window, "spec": a.spec,
                      "bytes": info["bytes"], "mb": round(info["bytes"] / 1e6, 2),
                      "export_seconds": round(time.perf_counter() - t0, 1)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
