#!/usr/bin/env python3
"""Pre-train every Gaussian grid the sweep will need, into assets/grids/.

Split out from the runner because grid training is the single expensive
one-off in this experiment (tens of minutes for K=2^16) and it depends only
on (m, K) — not on the corpus, the rotation seed, or the arm.  Training it
inside the sweep would re-pay that cost on every resume.
"""
from __future__ import annotations

import time

import grids

DIMS = (100, 768, 1024)
BITS = (1, 2, 3, 4, 6, 8)


def main():
    want = sorted({(grids.pick_m(b, d), 1 << (b * grids.pick_m(b, d)))
                   for d in DIMS for b in BITS if grids.pick_m(b, d) > 1})
    print(f"{len(want)} grids: {want}")
    for m, K in want:
        t = time.time()
        _, mse = grids.train_gaussian_grid(m, K, log=lambda s: print(s, flush=True))
        print(f"  grid m={m} K={K}: held-out mse/dim={mse:.6f} "
              f"({time.time() - t:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
