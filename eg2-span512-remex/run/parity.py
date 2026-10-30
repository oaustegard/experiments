"""Are the re-embedded vectors the ones the live index was built from?

Usage: python3 -I run/parity.py

Encodes out/docs.npy with the live index's own quantizer (remex 2.1.0, Quantizer(768, bits=2,
seed=42, mean=index.json mu)) and compares the codes with
../embeddinggemma2-webgpu-tail/web/bsky/codes.u8.txt, per code and per post, over the live
posts. A post edited or re-tokenized differently shows up as a row with many flips; float noise
from different batch padding shows up as scattered single flips. Writes results/parity.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import remex
from remex.packing import unpack

HERE = Path(__file__).resolve().parent.parent
IDX = HERE.parent / "embeddinggemma2-webgpu-tail" / "web" / "bsky"


def main() -> int:
    meta = json.loads((IDX / "index.json").read_text())
    n, d = meta["n"], meta["d"]
    mu = np.array(meta["mu"], np.float32)
    live_idx = np.load(HERE / "out" / "docs.npy")
    live = np.linalg.norm(live_idx, axis=1) > 0
    qz = remex.Quantizer(d, bits=2, seed=42, mean=mu)
    new = qz.encode(live_idx[live]).indices
    packed = np.frombuffer((IDX / "codes.u8.txt").read_bytes(), np.uint8)
    old = unpack(packed, 2, n * d).reshape(n, d)[live]
    flips = (new != old).sum(1)
    res = {"remex_version": remex.__version__, "n_live": int(live.sum()),
           "codes_flipped": int(flips.sum()), "codes_total": int(new.size),
           "code_agreement": float(1 - flips.sum() / new.size),
           "posts_identical": float((flips == 0).mean()),
           "posts_over_5pct_flipped": int((flips > 0.05 * d).sum()),
           "flips_per_post_p99": float(np.quantile(flips, 0.99))}
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "parity.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
