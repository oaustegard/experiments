#!/usr/bin/env python3
"""Reconcile this experiment with the 'One Bit Beats Two' blog.

The blog (SPECTER2) shows 1-bit > 2-bit > 3-bit at R@10-vs-fp32 — a reversal.
This experiment (Jina) shows monotone 8>4>2>1. Claim: the difference is
codebook CONSTRUCTION, not the embedder.

  * Matryoshka extraction  — build ONE 8-bit Lloyd-Max code, bit-shave to k
    bits (search at precision=k). This is the blog's Stage-1 setup.
  * Independent codebook    — build Quantizer(bits=k) fresh per level, each
    Lloyd-Max-optimized for its own width. This is what score_fidelity.py used.

Metric: recall@10 vs fp32-kNN (the blog's exact metric), on cached Jina vectors.
Prediction: Matryoshka reproduces the 1>2 reversal; independent is monotone.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

ROOT = Path("/home/user/claude-workspace"); HERE = Path(__file__).resolve().parent; A = HERE / "assets"
sys.path.insert(0, str(ROOT / ".spokes/remax/src"))
from remex import Quantizer

BITS = [1, 2, 3, 4, 8]
K = 10


def recall10(D, Q, order_fn):
    gt = np.argsort(-(Q @ D.T), axis=1)[:, :K]
    tot = 0.0
    for j in range(Q.shape[0]):
        cand = order_fn(j)[:K]
        tot += len(set(gt[j].tolist()) & set(cand.tolist())) / K
    return tot / Q.shape[0]


def main():
    for corpus, dn, qn in [("muninn", ".vec_doc.npz", ".vec_qry.npz"),
                           ("nfcorpus", ".nf_doc.npz", ".nf_qry.npz")]:
        if not (A / dn).exists():
            continue
        D = np.load(A / dn)["m"].astype(np.float32); Q = np.load(A / qn)["m"].astype(np.float32)
        print(f"\n### {corpus}: {D.shape[0]} docs, {Q.shape[0]} queries — R@10 vs fp32-kNN")

        # Matryoshka: one 8-bit code, bit-shave via precision=
        qz8 = Quantizer(d=768, bits=8, seed=0)
        comp8 = qz8.encode(np.ascontiguousarray(D))
        matry = {b: recall10(D, Q, lambda j, b=b: qz8.search(comp8, Q[j], k=K, precision=b)[0]) for b in BITS}

        # Independent: fresh codebook per bit-width
        indep = {}
        for b in BITS:
            qz = Quantizer(d=768, bits=b, seed=0)
            comp = qz.encode(np.ascontiguousarray(D))
            indep[b] = recall10(D, Q, lambda j, qz=qz, comp=comp: qz.search(comp, Q[j], k=K)[0])

        print(f"  {'bits':<6}" + "".join(f"{b:>8}" for b in BITS))
        print(f"  {'matry':<6}" + "".join(f"{matry[b]:>8.3f}" for b in BITS) + "   <- blog setup")
        print(f"  {'indep':<6}" + "".join(f"{indep[b]:>8.3f}" for b in BITS) + "   <- score_fidelity.py")
        m2 = "REVERSAL (1>2)" if matry[1] > matry[2] else "monotone"
        i2 = "monotone" if indep[2] >= indep[1] else "REVERSAL (1>2)"
        print(f"  matryoshka: {m2}   |   independent: {i2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
