"""Round 3: the full 262,144-row table (trained cut rows + rows learned by run/tok_rows.py),
compressed with frequency-tiered product quantization, against the trimmed shipped layout.
Usage: python3 -I run/r3_full_codecs.py [--table ridge_e3]  -> results/r3_full_codecs.json (resumes)
Rows are ordered by training-side token frequency (.work/freq.npy, ties by id); tier k takes the
next n rows. Bytes include each tier's 256 x 512 fp16 codebook."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vc import HERE, LS, WORK, Benches, cut_rows, full_student  # noqa: E402
from tcodecs import rpq, tiered  # noqa: E402

torch.set_num_threads(4)
K = 41741
ARMS = {
    "kept_int3+rare_pq16": [(K, ("int", 3)), (None, ("pq", 16))],
    "pq128/pq16": [(K, ("pq", 128)), (None, ("pq", 16))],
    "pq256/pq64/pq16": [(16384, ("pq", 256)), (40000, ("pq", 64)), (None, ("pq", 16))],
    "pq128/pq8": [(K, ("pq", 128)), (None, ("pq", 8))],
    "pq64/pq16": [(K, ("pq", 64)), (None, ("pq", 16))],
    "rpq16+128": ("rpq", [(16, None), (128, K)]),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="ridge_e3")
    ap.add_argument("--arms", nargs="*")
    a = ap.parse_args()
    sd = torch.load(LS / ".work/mac/out/l1/mac_k4_v32k/final.pt", map_location="cpu")
    X = np.load(WORK / f"table_{a.table}.npy")
    rows = cut_rows()
    X[rows] = sd["tm.embed_tokens.weight"].numpy()  # holdout rows back to their trained values
    w = np.load(WORK / "freq.npy").astype(np.float64) + 1.0
    w[rows] += 0.5  # kept rows first among equals
    B = Benches()
    p = HERE / "results" / "r3_full_codecs.json"
    out = json.loads(p.read_text()) if p.exists() else {}
    for name in a.arms or ARMS:
        if name in out:
            continue
        spec = ARMS[name]
        if spec is None:
            R, nb = X, X.size * 2
        elif spec[0] == "rpq":
            R, nb = rpq(X, spec[1], w)
        else:
            R, nb = tiered(X, w, spec)
        c = (R * X).sum(1) / np.maximum(np.linalg.norm(R, axis=1) * np.linalg.norm(X, axis=1), 1e-12)
        r = B.run(full_student(sd, table=R.astype(np.float32)), "full")
        r.update({"table_mb": round(nb / 1e6, 3), "row_cos_kept": round(float(c[rows].mean()), 4),
                  "row_cos_all": round(float(c.mean()), 4)})
        out[name] = r
        print(name, r, flush=True)
        p.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
