#!/usr/bin/env python3
"""Does the 'One Bit Beats Two' reversal reproduce on SPECTER2 with THIS harness?

Same code path as reconcile.py (Matryoshka bit-shave vs independent codebook,
R@10 vs fp32-kNN), but on the blog's own broad-NLP SPECTER2 cache instead of
Jina. Self-retrieval: 200 sampled vectors as queries against the 10k corpus,
ground truth = exact fp32 cosine top-10 (self excluded).

If SPECTER2 reverses (1>2) and Jina does not, under identical code, the driver
is the embedder, not the codebook construction.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

ROOT = Path("/home/user/claude-workspace"); A = Path(__file__).resolve().parent / "assets"
sys.path.insert(0, str(ROOT / ".spokes/remax/src"))
from remex import Quantizer

BITS = [1, 2, 3, 4, 8]
K = 10
NQ = 200
SEED = 0


def unit(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)


def recall10(D, qidx, order_fn):
    """order_fn(j) -> doc indices best->worst for query qidx[j]; self excluded."""
    tot = 0.0
    for j, qi in enumerate(qidx):
        fp = D @ D[qi]; fp[qi] = -np.inf
        gt = set(np.argsort(-fp)[:K].tolist())
        cand = [i for i in order_fn(j) if i != qi][:K]
        tot += len(gt & set(cand)) / K
    return tot / len(qidx)


def main():
    raw = np.load(A / "specter2_nlp_broad.npy").astype(np.float32)
    D = unit(raw)
    rng = np.random.default_rng(SEED)
    qidx = rng.choice(D.shape[0], NQ, replace=False)
    Q = D[qidx]
    print(f"SPECTER2 broad: {D.shape[0]} docs, {NQ} queries (self-retrieval) — R@10 vs fp32-kNN", flush=True)

    qz8 = Quantizer(d=768, bits=8, seed=SEED)
    comp8 = qz8.encode(np.ascontiguousarray(D))
    matry = {b: recall10(D, qidx, lambda j, b=b: qz8.search(comp8, Q[j], k=D.shape[0], precision=b)[0])
             for b in BITS}
    indep = {}
    for b in BITS:
        qz = Quantizer(d=768, bits=b, seed=SEED)
        comp = qz.encode(np.ascontiguousarray(D))
        indep[b] = recall10(D, qidx, lambda j, qz=qz, comp=comp: qz.search(comp, Q[j], k=D.shape[0])[0])

    print(f"  {'bits':<6}" + "".join(f"{b:>8}" for b in BITS))
    print(f"  {'matry':<6}" + "".join(f"{matry[b]:>8.3f}" for b in BITS) + "   <- blog setup")
    print(f"  {'indep':<6}" + "".join(f"{indep[b]:>8.3f}" for b in BITS))
    print(f"  matryoshka: {'REVERSAL (1>2)' if matry[1] > matry[2] else 'monotone'}   |   "
          f"independent: {'REVERSAL (1>2)' if indep[1] > indep[2] else 'monotone'}")
    print(f"\n  blog reported (broad, n=10k): 1=0.635 2=0.501 3=0.595 4=0.731 8=0.971")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
