#!/usr/bin/env python3
"""Export one `.cact` at a given weight-bit spec.

    python3 build_quant.py --spec 'embedding=4,mhc=4,default=2' --out /tmp/x.cact

The exporter accepts 2, 3 and 4 bits (`CQ_BITS`) plus ternary, spelled `1.58`
(`TERNARY_BITS`, codebook {-1.224, 0, +1.224}). Keys are canonical tensor names
(`embedding`, `mhc*`, `attn.*`, `engram0.tables`, …) and a spec must carry a
`default=`. Everything else about the checkpoint is untouched, so an arm differs
from the control only in how its weights are packed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

CHECKPOINT = "checkpoints/needle2.pkl"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoint", default=CHECKPOINT)
    a = ap.parse_args()

    from needle.model.run import load_checkpoint
    from needle.model.export import write_export
    from needle.model.architecture import effective_kv_window
    from needle.model.tokenizer import get_tokenizer
    from needle.model.quantize import parse_bits_map

    parse_bits_map(a.spec)  # fail loudly here, not inside the packer
    params, cfg = load_checkpoint(a.checkpoint)

    t0 = time.perf_counter()
    info = write_export(params, cfg, a.out, bits_map=a.spec,
                        tokenizer=get_tokenizer(cfg.vocab_size),
                        kv_window=effective_kv_window(cfg))
    print(json.dumps({"spec": a.spec, "bytes": info["bytes"],
                      "mb": round(info["bytes"] / 1e6, 2), "tensors": info["tensors"],
                      "export_seconds": round(time.perf_counter() - t0, 1)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
