#!/usr/bin/env python3
"""Two-sided calibration gate.  Non-zero exit on any failure.

The issue this experiment answers says, of axis C:

    If axis C shows **no** difference, that is evidence the VQ arm is
    under-implemented, **not** evidence that scalar is optimal.  Check the VQ
    arm against a published number before concluding anything.

So the gate runs *before* the sweep and its verdict is a precondition on
reading any axis-C result.  Following METHODS.md principle 2, it is two-sided:
it must certify the good implementations clean **and** catch a deliberately
broken one.  A one-sided "nothing looked wrong" pass is what let `svgview`
ship seven green tests over an input path that was never connected.

Checks
------
G0  the empirical MSE instrument agrees with the closed-form scalar answer,
    via two code paths that share nothing                          [instrument]
G1  scalar Lloyd-Max MSE == Max (1960) table 1                     [published]
G2  E8 normalised second moment == 0.0716821 (Conway & Sloane)     [published]
G3  trained m-dim Gaussian grid beats scalar at the same rate, and never
    beats the Shannon bound 2^-2b                                  [bracket]
G4  the trained 8-dim grid at 2 bits/coord is at least as good as a tuned
    ball-shaped E8 lattice codebook of the same size — i.e. our grid is not
    worse than QuIP#'s actual codebook family                      [published]
G5  grid quality improves monotonically with sub-vector dimension m [ordering]
G6  RHT incoherence matches Haar's — otherwise axis A is confounded by a weak
    rotation rather than measuring rotation cost                    [bracket]
G7  KNOWN-BAD: a deliberately under-trained grid must be caught by G3/G4's
    own criteria.  If it passes, the gate cannot discriminate and every other
    check above is decoration.                                     [two-sided]
G8  payload bytes are identical for scalar and vector arms at matched
    (d, bits); side-channel bytes are itemised, not folded away    [budget]

G3 is not hypothetical: it fired on the first real build of this experiment,
where Lloyd seeded from random samples converged to grids *worse* than the
scalar quantizer at 6 and 8 bits.  See RESULTS.md — that would have been
reported as "scalar wins axis C at high rate", which is exactly the wrong
conclusion the issue warns about.
"""
from __future__ import annotations

import math
import sys

import numpy as np
from scipy.spatial import cKDTree

import grids
import quantizers as qz

FAILURES: list[str] = []
NOTES: list[str] = []


def check(ok: bool, label: str, detail: str):
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {label}: {detail}")
    if not ok:
        FAILURES.append(f"{label}: {detail}")
    return ok


# --------------------------------------------------------------------------
# E8 ball codebook — the published-family anchor for G4


def enumerate_e8(max_norm: float) -> np.ndarray:
    """All E8 lattice points with ||p|| <= max_norm.

    E8 = D8 u (D8 + (1/2,...,1/2)), D8 = integer vectors of even coordinate
    sum.  Built dimension by dimension with norm pruning, so the intermediate
    sets stay small instead of materialising a 9^8 box.
    """
    t = max_norm ** 2
    lim = math.floor(max_norm) + 1
    vals = np.arange(-lim, lim + 1)
    pts = np.zeros((1, 0), dtype=np.int32)
    sq = np.zeros(1)
    for _ in range(8):
        pts = np.repeat(pts, vals.size, axis=0)
        col = np.tile(vals, pts.shape[0] // vals.size)[:, None]
        sq = np.repeat(sq, vals.size) + col[:, 0] ** 2
        pts = np.hstack([pts, col])
        keep = sq <= t
        pts, sq = pts[keep], sq[keep]
    d8 = pts[pts.sum(axis=1) % 2 == 0].astype(np.float64)
    shifted = d8 + 0.5
    shifted = shifted[np.sum(shifted ** 2, axis=1) <= t]
    return np.vstack([d8, shifted])


def e8_ball_codebook(k: int, max_norm: float = 4.2) -> np.ndarray:
    """The k lowest-norm E8 points — a ball-shaped subset, which is the
    shaping QuIP#'s E8P codebook uses.  Returned unscaled."""
    pts = enumerate_e8(max_norm)
    order = np.argsort(np.sum(pts ** 2, axis=1), kind="stable")
    if order.size < k:
        raise SystemExit(f"only {order.size} E8 points within {max_norm}; raise max_norm")
    return pts[order[:k]]


def codebook_mse(C: np.ndarray, n: int = 300_000, seed: int = 11) -> float:
    """Per-dimension MSE of a codebook against N(0, I_m)."""
    m = C.shape[1]
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, m)).astype(np.float32)
    dist, _ = cKDTree(np.ascontiguousarray(C, dtype=np.float32)).query(X, k=1, workers=-1)
    return float(np.mean(dist ** 2)) / m


def best_scaled_mse(C: np.ndarray, n: int = 200_000) -> tuple[float, float]:
    """Minimise MSE over a scalar scaling of the codebook.

    Without this the E8 anchor would be handicapped by an arbitrary scale and
    G4 would be a strawman in our own favour — the opposite of the failure the
    gate exists to catch.
    """
    best = (np.inf, 1.0)
    for s in np.linspace(0.25, 1.6, 28):
        m = codebook_mse(C * s, n=n)
        if m < best[0]:
            best = (m, float(s))
    # refine around the winner
    lo, hi = best[1] * 0.85, best[1] * 1.15
    for s in np.linspace(lo, hi, 21):
        m = codebook_mse(C * s, n=n)
        if m < best[0]:
            best = (m, float(s))
    return best


# --------------------------------------------------------------------------
# gates


def g0_measurement_path():
    """Validate the empirical MSE path against the closed-form scalar answer.

    `codebook_mse` (sampling + KD-tree nearest neighbour) and `lloyd_max_1d`
    (exact integration against the normal density) share no code.  Lifting the
    scalar levels into an m-dimensional product grid makes them measure the
    same quantity, because nearest-neighbour assignment on a product grid
    decomposes into independent per-coordinate assignment.  If they agree, a
    later disagreement between a trained grid and the scalar arm is a fact
    about the grid rather than about the measurement.

    This is METHODS.md principle 1 applied to the instrument, and it is what
    localised the random-init Lloyd defect described in RESULTS.md: the
    measurement was exonerated first, so the grid was the only suspect left.
    """
    print("\nG0  empirical MSE path vs closed-form scalar (disjoint code paths)")
    ok = True
    for b, m in ((2, 2), (4, 2), (6, 2), (2, 4)):
        _, closed = grids.lloyd_max_1d(b)
        measured = codebook_mse(grids.product_init(b, m), n=1_000_000)
        rel = abs(measured - closed) / closed
        ok &= check(rel < 1e-2, f"b={b} m={m} product grid",
                    f"measured={measured:.7f} closed-form={closed:.7f} "
                    f"rel={rel:.2e}")
    return ok


def g1_scalar_table():
    print("\nG1  scalar Lloyd-Max vs Max (1960) table 1")
    ok = True
    for b, pub in grids.MAX_1960_MSE.items():
        _, mse = grids.lloyd_max_1d(b)
        rel = abs(mse - pub) / pub
        ok &= check(rel < 2e-3, f"b={b}",
                    f"mse={mse:.6f} published={pub} rel={rel:.2e}")
    for b in (6, 8):
        _, mse = grids.lloyd_max_1d(b)
        NOTES.append(f"scalar LM b={b}: mse={mse:.3e}, "
                     f"ratio to Shannon 2^-2b = {mse / 2.0 ** (-2 * b):.3f} "
                     f"(Panter-Dite asymptote 2.721)")
    return ok


def g2_e8_nsm():
    print("\nG2  E8 normalised second moment vs Conway & Sloane")
    nsm = grids.e8_nsm(n=600_000)
    return check(abs(nsm - 0.0716821) / 0.0716821 < 5e-3, "E8 NSM",
                 f"{nsm:.7f} vs published 0.0716821")


def g3_bracket(rows):
    print("\nG3  trained grid: beats scalar, never beats Shannon")
    ok = True
    for b, m, K, mse_vq, mse_sc in rows:
        shannon = 2.0 ** (-2 * b)
        ok &= check(mse_vq < mse_sc, f"b={b} m={m} < scalar",
                    f"vq={mse_vq:.5f} scalar={mse_sc:.5f} "
                    f"gain={10 * math.log10(mse_sc / mse_vq):.2f} dB")
        ok &= check(mse_vq > shannon, f"b={b} m={m} > Shannon",
                    f"vq={mse_vq:.5f} bound={shannon:.5f} "
                    f"ratio={mse_vq / shannon:.3f}")
    return ok


def g4_e8_anchor():
    print("\nG4  trained 8-dim grid at 2 bits/coord vs a tuned E8 ball codebook")
    C8 = e8_ball_codebook(1 << 16)
    mse_e8, s = best_scaled_mse(C8)
    Cg, _ = grids.train_gaussian_grid(8, 1 << 16, log=lambda m: print("   ", m))
    mse_grid = codebook_mse(Cg)
    _, mse_sc = grids.lloyd_max_1d(2)
    NOTES.append(f"E8 ball codebook (2^16 pts, scale {s:.3f}): mse/dim={mse_e8:.5f}; "
                 f"trained grid m=8: {mse_grid:.5f}; scalar LM: {mse_sc:.5f}")
    ok = check(mse_grid <= mse_e8 * 1.02, "grid <= E8 (2% slack)",
               f"grid={mse_grid:.5f} E8={mse_e8:.5f} ratio={mse_grid / mse_e8:.3f}")
    ok &= check(mse_e8 < mse_sc, "E8 itself beats scalar",
                f"E8={mse_e8:.5f} scalar={mse_sc:.5f}")
    return ok, mse_e8, mse_grid


def g5_monotone_m():
    print("\nG5  grid quality improves with sub-vector dimension m (b=2)")
    prev, ok = None, True
    _, sc = grids.lloyd_max_1d(2)
    seq = [(1, sc)]
    for m in (2, 4, 8):
        C, _ = grids.train_gaussian_grid(m, 1 << (2 * m), log=None)
        seq.append((m, codebook_mse(C)))
    for m, v in seq:
        if prev is not None:
            ok &= check(v < prev[1] * 1.001, f"m={m} <= m={prev[0]}",
                        f"{v:.5f} vs {prev[1]:.5f}")
        prev = (m, v)
    return ok


def g6_incoherence():
    print("\nG6  RHT incoherence matches Haar (else axis A is confounded)")
    ok = True
    rng = np.random.default_rng(3)
    for d in (100, 768, 1024):
        X = rng.standard_normal((512, d)).astype(np.float32)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        # worst case for a rotation is a coordinate-aligned spike, not a
        # random vector — that is exactly what incoherence has to kill
        E = np.zeros((64, d), np.float32)
        E[np.arange(64), rng.choice(d, 64, replace=False)] = 1.0
        stats = {}
        for kind in ("haar", "rht"):
            R = qz.ROTATIONS[kind](d, 5)
            mu_rand = float(np.mean(np.max(np.abs(R.apply(X)), axis=1)))
            mu_spike = float(np.mean(np.max(np.abs(R.apply(E)), axis=1)))
            stats[kind] = (mu_rand, mu_spike)
        hr, hs = stats["haar"]
        rr, rs = stats["rht"]
        ok &= check(rs < hs * 1.25, f"d={d} spike incoherence",
                    f"rht={rs:.4f} haar={hs:.4f} (1/sqrt(d)={1 / math.sqrt(d):.4f})")
        NOTES.append(f"d={d}: mean max|coord| random unit vec — "
                     f"haar {hr:.4f}, rht {rr:.4f}")
    return ok


def g7_known_bad(mse_e8: float):
    """The two-sided half: G3/G4's criteria must REJECT an under-trained grid.

    An 8-dim grid stopped after 2 Lloyd iterations from a random-sample init
    is exactly the plausible-looking-but-wrong VQ arm the issue warns about.
    If the gate calls it fine, the gate proves nothing about the real arm.
    """
    print("\nG7  KNOWN-BAD: under-trained grid must be rejected")
    rng = np.random.default_rng(4242)
    X = rng.standard_normal((400_000, 8)).astype(np.float32)
    C_bad, _ = grids._lloyd(X, 1 << 16, 2, rng)
    mse_bad = codebook_mse(C_bad)
    _, mse_sc = grids.lloyd_max_1d(2)
    caught_vs_e8 = not (mse_bad <= mse_e8 * 1.02)
    NOTES.append(f"under-trained m=8 grid (2 Lloyd iters): mse/dim={mse_bad:.5f} "
                 f"vs converged E8 anchor {mse_e8:.5f}, scalar {mse_sc:.5f}")
    return check(caught_vs_e8, "G4 criterion rejects the bad grid",
                 f"bad={mse_bad:.5f} > E8*1.02={mse_e8 * 1.02:.5f}")


def g8_budget():
    print("\nG8  payload bytes matched across codebooks; side channels itemised")
    ok = True
    for d in (100, 768, 1024):
        for b in (2, 4):
            a_s = qz.Arm(rotation="haar", norm="exactnorm", codebook="scalar",
                         bits=b, d=d, seed=0)
            a_v = qz.Arm(rotation="rht", norm="blockscale", codebook="vector",
                         bits=b, d=d, seed=0)
            bs, bv = a_s.bytes_per_vector(), a_v.bytes_per_vector()
            ok &= check(bs["payload"] == bv["payload"], f"d={d} b={b} payload",
                        f"scalar={bs['payload']} vector={bv['payload']}")
            ok &= check(bs["side"] == 4.0 and bv["side"] == 2.0 * a_v.nblocks,
                        f"d={d} b={b} side channels",
                        f"exactnorm={bs['side']}B blockscale={bv['side']}B "
                        f"({a_v.nblocks} blocks of {a_v.block})")
    return ok


def main():
    print("=" * 78)
    print("CALIBRATION GATE — remex-vs-higgs-ablation")
    print("=" * 78)
    g0_measurement_path()
    g1_scalar_table()
    g2_e8_nsm()

    rows = []
    for b in (1, 2, 3, 4, 6, 8):
        m = grids.pick_m(b, 768)
        if m == 1:
            continue
        C, _ = grids.train_gaussian_grid(m, 1 << (b * m),
                                         log=lambda s: print("   ", s))
        _, sc = grids.lloyd_max_1d(b)
        rows.append((b, m, 1 << (b * m), codebook_mse(C), sc))
    g3_bracket(rows)
    _, mse_e8, _ = g4_e8_anchor()
    g5_monotone_m()
    g6_incoherence()
    g7_known_bad(mse_e8)
    g8_budget()

    print("\n" + "-" * 78)
    print("NOTES")
    for n in NOTES:
        print(f"  - {n}")
    print("-" * 78)
    if FAILURES:
        print(f"\nGATE FAILED — {len(FAILURES)} check(s):")
        for f in FAILURES:
            print(f"  * {f}")
        return 1
    print("\nGATE PASSED — axis-C results are readable.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
