#!/usr/bin/env python3
"""Salvage a v1-format grid into the v2 cache, but only if it earns it.

Context: bumping `GRID_VERSION` correctly invalidated every cached grid, and
the expensive one (m=8, K=65536 — the 2 bits/coordinate QuIP# regime) costs
~40 minutes to retrain.  For that single configuration the v1 and v2 training
procedures differ only by the addition of a `product-raw` candidate, which at
2 bits/coordinate is the scalar quantizer (mse/dim 0.1175) and cannot possibly
win against a grid in the 0.08 range.

That is an argument, and an argument is exactly what the adversarial review
punished last time — the "provably no worse than scalar" claim was also sound
reasoning that did not survive contact with the code.  So this script does not
migrate on the strength of the argument.  It re-measures the candidate grid on
the v2 reporting stream, re-measures the `product-raw` candidate the v2
procedure would have added, and refuses the migration unless the file actually
wins by the v2 selection rule and lands where the training log said it should.

Anything that fails a check is left alone, to be retrained from scratch.

    python3 migrate_grid.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import grids

HERE = Path(__file__).resolve().parent
CACHE = HERE / "assets" / "grids"

#: (m, K) grids worth salvaging, and the **held-out** mse/dim the v1 run
#: reported for them.  Anything not listed here is simply retrained.
#:
#: Read this number off the v1 log's final "grid m=.. K=..: held-out mse/dim="
#: line, NOT off an intermediate "lloyd it=.." line -- those are training-set
#: MSE and are optimistic by ~11% at the 61 samples/codepoint the largest
#: grids get.  Seeding this constant with a training number is what made the
#: first run of this script refuse a perfectly good grid.
SALVAGE = {(8, 1 << 16): 0.088695}
TOL = 0.05


def main():
    src = CACHE / "grid_m8_K65536.npz"
    dst = CACHE / f"grid_v{grids.GRID_VERSION}_m8_K65536.npz"
    if dst.exists():
        print(f"{dst.name} already present — nothing to do")
        return 0
    if not src.exists():
        print(f"{src.name} absent — nothing to salvage, will be retrained")
        return 0

    z = np.load(src)
    C = z["C"]
    m, K = C.shape[1], C.shape[0]
    expected = SALVAGE.get((m, K))
    if expected is None:
        print(f"({m}, {K}) is not on the salvage list — retrain")
        return 0
    bits = round(np.log2(K) / m)

    # 1. re-measure on the v2 reporting stream (a different seed from the one
    #    the v1 run selected on, so this is not just re-reading its own answer)
    mse = grids.held_out_mse(C, seed=grids.REPORT_SEED)
    # 2. the candidate v2 would have added, and which v1 never considered
    raw = grids.product_init(bits, m)
    mse_raw = grids.held_out_mse(raw, seed=grids.REPORT_SEED)
    # 3. the scalar arm it has to beat for axis C to mean anything
    _, mse_scalar = grids.lloyd_max_1d(bits)

    print(f"grid m={m} K={K} ({bits}b/coord)")
    print(f"  v1 training log said     : {expected:.6f}")
    print(f"  re-measured (v2 stream)  : {mse:.6f}")
    print(f"  product-raw candidate    : {mse_raw:.6f}")
    print(f"  scalar Lloyd-Max         : {mse_scalar:.6f}")
    print(f"  Shannon bound 2^-2b      : {2.0 ** (-2 * bits):.6f}")

    checks = [
        (abs(mse - expected) / expected < TOL,
         f"re-measured MSE within {TOL:.0%} of the v1 log"),
        (mse < mse_raw,
         "beats the product-raw candidate v2 would have added"),
        (mse < mse_scalar,
         "beats the scalar quantizer at the same rate"),
        (mse > 2.0 ** (-2 * bits),
         "does not beat the Shannon bound"),
    ]
    ok = True
    for passed, label in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}")
        ok &= passed
    if not ok:
        print("\nREFUSED — leaving the v1 file in place; retrain instead.")
        return 1

    tmp = dst.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, C=C, mse=mse, init="product-init (migrated v1->v2)",
                        version=grids.GRID_VERSION)
    tmp.replace(dst)
    src.unlink()
    print(f"\nMIGRATED -> {dst.name} (mse/dim {mse:.6f}); v1 file removed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
