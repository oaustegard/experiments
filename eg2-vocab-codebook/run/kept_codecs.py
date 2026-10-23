"""Reconstruction cosine of the cut's 41,741 trained rows under each codec, with bytes per row.
The rows are close to isotropic (coding gain of a PCA rotation 0.6 dB), so the codec's
quantizer shape and side-information cost are what differ."""
import sys
from pathlib import Path

import faiss
import numpy as np
import remex
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vc import LS  # noqa: E402
from tcodecs import int_g32  # noqa: E402

X = torch.load(LS / ".work/mac/out/l1/mac_k4_v32k/final.pt", map_location="cpu")["tm.embed_tokens.weight"].numpy()
n = len(X)


def cos(R):
    return float(((R * X).sum(1) / np.linalg.norm(R, axis=1) / np.linalg.norm(X, axis=1)).mean())


for b in (4, 3, 2):
    R, nb = int_g32(X, b)
    print(f"int{b}_g32", round(cos(R), 4), nb / n, "B/row")
for b in (4, 3, 2):
    for rot in ("haar", "rht"):
        q = remex.Quantizer(512, bits=b, seed=0, rotation=rot)
        R = q.decode(q.encode(X))
        print(f"remex{b}_{rot}", round(cos(R), 4), 512 * b / 8 + 2, "B/row")
for M, nbits in ((128, 8), (128, 10), (256, 8), (64, 12), (170, 8)):
    if 512 % M:
        continue
    pq = faiss.ProductQuantizer(512, M, nbits)
    pq.train(np.ascontiguousarray(X))
    R = pq.decode(pq.compute_codes(np.ascontiguousarray(X)))
    cb = (2 ** nbits) * 512 * 2
    print(f"pq{M}x{nbits}", round(cos(R), 4), M * nbits / 8, "B/row +", cb / 1e6, "MB codebook")
