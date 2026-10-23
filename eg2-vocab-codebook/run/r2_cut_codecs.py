"""Round 2: codecs on the cut's own 41,741-row trained table at matched bytes (no vocab change).
Usage: python3 -I run/r2_cut_codecs.py   -> results/r2_cut_codecs.json"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vc import HERE, LS, WORK, Benches, cut_rows, cut_student  # noqa: E402
from tcodecs import int_g32, pq_apply, pq_bytes, pq_fit, rpq  # noqa: E402

torch.set_num_threads(4)


def main():
    sd = torch.load(LS / ".work/mac/out/l1/mac_k4_v32k/final.pt", map_location="cpu")
    X = sd["tm.embed_tokens.weight"].numpy()
    w = np.load(WORK / "freq.npy")[cut_rows()] + 1.0
    B = Benches()
    arms = {"float": lambda: (X, X.size * 2)}
    for b in (4, 3, 2, 1):
        arms[f"int{b}_g32"] = (lambda b=b: int_g32(X, b))
    for M in (64, 128, 256):
        for wn, ww in (("unif", None), ("freq", w)):
            def f(M=M, ww=ww):
                pq = pq_fit(X, M, w=ww)
                return pq_apply(pq, X)[0], pq_bytes(len(X), M)
            arms[f"pq{M}_{wn}"] = f
    arms["rpq128+64"] = lambda: rpq(X, [(128, None), (64, None)], w)[:2]
    out = {}
    p = HERE / "results" / "r2_cut_codecs.json"
    if p.exists():
        out = json.loads(p.read_text())
    for name, f in arms.items():
        if name in out:
            continue
        R, nb = f()
        c = float(((R * X).sum(1) / np.linalg.norm(R, axis=1) / np.linalg.norm(X, axis=1)).mean())
        r = B.run(cut_student(sd, R), "cut")
        r.update({"table_mb": round(nb / 1e6, 3), "row_cos": round(c, 4)})
        out[name] = r
        print(name, r, flush=True)
        p.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
