"""Step 4: does a head with remex's rotation folded in produce remex's own codes?

Usage: python3 -I run/parity.py

Three encodings of the 61,686 posts at 4 bits, all from the same pooled states and the fp32
checkpoint head W0 = A S B^T:
  a  unit(p A S), remex rotating the rows itself                  (rotation after the model)
  b  unit(p A S R^T), encoded with rotation="none"                (rotation inside an fp32 head)
  c  unit(p Hq), Hq = int8 +-127 rounding of A S R^T, rotation="none"   (the head to deploy)
Reports the share of the 61,686 x 256 codes on which b equals a and c equals a, and how far
the unequal codes are apart. Writes results/parity.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import remex
from safetensors.numpy import load_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, RES, int8_cols, svd_head, unit

BITS = 4


def main():
    P = np.load(OUT / "vec" / "docs.p.npy").astype(np.float64)
    W0 = load_file(str(OUT / "rel" / "st" / "epoch1" / "2_Dense" / "model.safetensors"))["linear.weight"].T.astype(np.float64)
    H0, _, _ = svd_head(W0)
    Ya = unit(P @ H0).astype(np.float32)
    mu = Ya.mean(0).astype(np.float32)
    qa = remex.Quantizer(256, bits=BITS, seed=42, mean=mu)
    a = qa.encode(Ya).indices
    R = qa.R.astype(np.float64)
    res = {"bits": BITS, "n_codes": int(a.size), "remex": remex.__version__}
    q, s = int8_cols(H0 @ R.T, 127)
    for name, head in (("fp32_head", H0 @ R.T), ("int8_head", q.astype(np.float64) * s)):
        Yr = unit(P @ head).astype(np.float32)
        codes = remex.Quantizer(256, bits=BITS, seed=42, rotation="none", mean=Yr.mean(0).astype(np.float32)).encode(Yr).indices
        diff = np.abs(codes.astype(np.int16) - a.astype(np.int16))
        res[name] = {"codes_equal_share": float((diff == 0).mean()),
                     "unequal_codes_one_level_apart_share": float((diff[diff > 0] == 1).mean())}
    (RES / "parity.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
