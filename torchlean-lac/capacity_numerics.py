#!/usr/bin/env python3
"""Where LAC's float32 read fails, under three rounding models.

The Lean theorems in LAC/Capacity.lean are about the model TorchLean's FP32
arithmetic implements: every stored key entry, the product, and the sum are
each rounded to binary32 (round-to-nearest-even). This script runs that
pipeline in numpy float32 next to two simpler models so the theorem's scope
is visible against the numbers:

  A  exact-then-round   fl(i^2) == fl(i^2 - 1)            one rounding of the exact scores
  B  fp32 pipeline      argmax_j fl(fl(fl(2j) * fl(i)) + fl(-j^2)) != i   (what Capacity.lean models)
  C  float32 matmul     argmax_j (K @ q) != i, K, q float32     (BLAS dot product)

Usage: python3 capacity_numerics.py [N]      (default N = 12000)
"""
import sys

import numpy as np

f32 = np.float32
N = int(sys.argv[1]) if len(sys.argv) > 1 else 12000


def model_a_ties(i):
    return f32(i * i) == f32(i * i - 1)


def model_b_fails(i):
    js = np.arange(0, i + 1, dtype=np.int64)
    k0 = (2 * js).astype(f32)
    k1 = (-(js * js)).astype(f32)
    s = ((k0 * f32(i)).astype(f32) + k1).astype(f32)
    return int(np.argmax(s)) != i


def model_c_fails(i):
    js = np.arange(0, i + 1, dtype=np.int64)
    K = np.stack([(2 * js).astype(f32), (-(js * js)).astype(f32)], 1)
    q = np.array([i, 1], dtype=f32)
    return int(np.argmax(K @ q)) != i


def first_true(pred, lo, hi):
    for i in range(lo, hi):
        if pred(i):
            return i
    return None


def runs(pred, lo, hi):
    """Maximal runs of consecutive i in [lo, hi) with pred(i) true."""
    out, start = [], None
    for i in range(lo, hi + 1):
        hit = i < hi and pred(i)
        if hit and start is None:
            start = i
        if not hit and start is not None:
            out.append((start, i - 1))
            start = None
    return out


if __name__ == "__main__":
    print(f"N = {N}")
    for name, pred in (("A exact-then-round tie", model_a_ties),
                       ("B fp32 pipeline read fails", model_b_fails),
                       ("C float32 matmul read fails", model_c_fails)):
        first = first_true(pred, 1, N)
        r = runs(pred, 1, N)
        head = r[0] if r else None
        n_fail = sum(1 for i in range(4097, N) if pred(i))
        print(f"{name:30s} first={first} first_run={head} "
              f"failures_in_[4097,{N})={n_fail}/{N - 4097}")
        if head:
            after = [i for i in range(head[1] + 1, min(head[1] + 400, N)) if not pred(i)][:6]
            print(f"{'':30s} first successes after the run: {after}")
