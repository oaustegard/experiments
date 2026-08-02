#!/usr/bin/env python3
"""Audit of `calibrate.py` — can each of its checks actually fail?

Written against the `gating` skill's `references/auditing.md`, which asks a
different question than "does the suite pass".  For each check: name a concrete
input that makes it go red, then *observe* it going red.  The four verdicts are
the skill's:

    CONFIRMED   a named input makes it fail, and it was observed failing
    PLAUSIBLE   a named input should make it fail, not yet observed
    CANNOT FAIL no input makes it fail — decoration
    BLIND       it can fail, but not on the regime that matters

Everything here is a probe against the *first-run* gate as committed, so the
probes must keep working against `calibrate.py` unchanged.  `gate.py` is the
rebuilt gate that closes what this finds; the two live side by side on purpose,
because the audit is evidence and deleting it would leave only the conclusion.

Run:  python3 audit.py           (~2 min; no corpora needed)
"""
from __future__ import annotations

import math
import sys

import numpy as np

import calibrate
import grids
import quantizers as qz

VERDICTS: list[tuple[str, str, str]] = []


def verdict(check: str, v: str, detail: str):
    VERDICTS.append((check, v, detail))
    print(f"\n  ==> {check}: {v}\n      {detail}")


# --------------------------------------------------------------------------
# Pass 1 — an assertion whose truth does not depend on the subject
# --------------------------------------------------------------------------


def probe_g6_identity_rotation():
    """G6 asserts `rht_spike < haar_spike * 1.25` and nothing else.

    That is a *relative* comparison between the two things under test, with no
    anchor outside them.  If both rotations were broken in the same way — the
    single most likely way for a rotation layer to break, since they share the
    apply/inverse plumbing and the same call site — the ratio stays 1.0 and the
    check reports PASS.

    The named input: swap both entries of `ROTATIONS` for the identity.  A
    rotation that does nothing has the worst possible incoherence (a
    coordinate spike stays a coordinate spike, max|coord| = 1.0), which is the
    exact property G6 exists to certify.  1/sqrt(d) is printed in G6's detail
    string but is never asserted against.
    """
    print("\n" + "=" * 74)
    print("PROBE 1 — G6 incoherence, both rotations replaced by the identity")
    print("=" * 74)
    saved = dict(qz.ROTATIONS)
    calibrate.FAILURES.clear()
    try:
        qz.ROTATIONS["haar"] = qz.IdentityRotation
        qz.ROTATIONS["rht"] = qz.IdentityRotation
        calibrate.g6_incoherence()
    finally:
        qz.ROTATIONS.clear()
        qz.ROTATIONS.update(saved)
    failed = len(calibrate.FAILURES)
    calibrate.FAILURES.clear()
    if failed == 0:
        verdict(
            "G6 incoherence", "CANNOT FAIL",
            "Both rotations replaced by the identity — the strongest possible "
            "violation of what G6 certifies — and G6 reported PASS at every d. "
            "The check compares RHT to Haar only; with no absolute anchor, a "
            "shared failure is invisible. Axis A would be read off two "
            "no-op rotations.")
    else:
        verdict("G6 incoherence", "CONFIRMED",
                f"identity rotations produced {failed} failure(s)")


def probe_g3_degenerate_trainer():
    """G3 asserts `grid_mse < scalar_mse`, with no floor on the margin.

    Since the trainer now carries `product-raw` (the scalar quantizer lifted to
    m dimensions) as a *candidate*, a totally broken Lloyd stage does not make
    the vector arm bad — it makes it exactly equal to the scalar arm.  And
    "exactly equal" is compared against a closed-form scalar value using a
    *sampled* measurement, so which side of the inequality it lands on is
    sampling noise.

    The named input: a trainer whose Lloyd refinement is a no-op, i.e. the
    vector arm degrades to the scalar arm.  This is the realistic shape of a
    broken vector arm after the product-init fix — the fix that was introduced
    precisely so the arm could never come out worse.  It bought that guarantee
    by making G3's assertion nearly true by construction.
    """
    print("\n" + "=" * 74)
    print("PROBE 2 — G3 bracket, vector trainer degraded to the scalar arm")
    print("=" * 74)
    rows = []
    for b, m in ((2, 8), (3, 4), (4, 4)):
        K = 1 << (b * m)
        P = grids.product_init(b, m)          # Lloyd contributed nothing
        mse_vq = calibrate.codebook_mse(P, n=1_000_000)
        _, mse_sc = grids.lloyd_max_1d(b)
        rows.append((b, m, K, mse_vq, mse_sc))
        print(f"    b={b} m={m}: product-only grid={mse_vq:.7f} "
              f"scalar={mse_sc:.7f}  gain={10 * math.log10(mse_sc / mse_vq):+.3f} dB")
    calibrate.FAILURES.clear()
    calibrate.g3_bracket(rows)
    failed = len(calibrate.FAILURES)
    calibrate.FAILURES.clear()
    gains = [10 * math.log10(sc / vq) for _, _, _, vq, sc in rows]
    if failed == 0:
        verdict(
            "G3 beats-scalar", "CANNOT FAIL",
            f"A vector arm contributing zero quantization gain "
            f"({min(gains):+.3f}..{max(gains):+.3f} dB) passed G3 at every "
            f"rate. G3 has no floor on the margin, so it certifies "
            f"'vector quantization happened' from an arm that did none. "
            f"Axis C's entire premise rests on this check.")
    else:
        verdict("G3 beats-scalar", "BLIND",
                f"{failed}/{2 * len(rows)} assertions fired on a zero-gain arm "
                f"— it lands on the right side of the inequality only by "
                f"sampling noise, not by a stated margin")


# --------------------------------------------------------------------------
# Pass 2 — the oracle's range
# --------------------------------------------------------------------------


def probe_g1_range():
    """G1 is anchored on Max (1960) table 1, which stops at 5 bits.

    The sweep runs at 6 and 8 bits.  The 16%-high scalar MSE that the first
    run's adversarial review caught by hand lived at exactly 8 bits, and the
    gate as committed still has no assertion there — only a printed NOTE.

    Two separate things are checked here, and only the first is proven by
    construction:

      1. the RANGE GAP — b=6 and b=8 lie outside the anchor, so no assertion
         in the suite constrains them.  This is structural and certain.
      2. whether the specific historical value is caught.  An attempt to
         reconstruct it from the fixed-point identity is included and it
         DOES NOT reproduce the defect — it converges to the correct answer.
         So the reconstruction is reported as a failed reproduction rather
         than dressed up as a demonstration; the historical value is carried
         as the literal RESULTS.md records.
    """
    print("\n" + "=" * 74)
    print("PROBE 3 — G1 anchor range vs the rates the sweep actually uses")
    print("=" * 74)
    covered = sorted(grids.MAX_1960_MSE)
    used = [1, 2, 3, 4, 6, 8]
    gap = [b for b in used if b not in covered]
    print(f"    anchor covers b={covered}; sweep uses b={used}; unanchored: {gap}")

    # Reconstruct the historical defect: the fixed-point identity, evaluated on
    # levels that have not fully converged.  This is what `lloyd_max_1d`
    # returned before the fix, and it is 16% high at 8 bits.
    def identity_mse(bits, iters=20_000):
        from scipy.stats import norm
        k = 1 << bits
        levels = norm.ppf((np.arange(k) + 0.5) / k)
        prev = np.inf
        for _ in range(iters):
            bnd = 0.5 * (levels[:-1] + levels[1:])
            edges = np.concatenate([[-np.inf], bnd, [np.inf]])
            p = np.diff(norm.cdf(edges))
            pdf = norm.pdf(edges)
            with np.errstate(invalid="ignore", divide="ignore"):
                levels = np.where(p > 1e-300, (pdf[:-1] - pdf[1:]) / np.maximum(p, 1e-300), levels)
            mse = 1.0 - float(np.sum(p * levels ** 2))
            if abs(prev - mse) < 1e-15:
                break
            prev = mse
        return mse

    repro = identity_mse(8)
    _, true8 = grids.lloyd_max_1d(8)
    panter = 2.7207 * 2.0 ** (-2 * 8)
    #: The value RESULTS.md records as having shipped in the first build,
    #: before the adversarial review replaced the fixed-point identity with
    #: direct integration.  Carried as a literal: the reconstruction below
    #: does not reproduce it.
    historical = 4.791e-5
    print(f"    b=8 attempted reconstruction of the pre-fix path = {repro:.7e}")
    print(f"    b=8 corrected (direct integration)               = {true8:.7e}")
    print(f"    -> reconstruction error {100 * (repro - true8) / true8:+.3f}% "
          f"— DOES NOT reproduce the documented +16%; the defect depended on "
          f"more than the identity itself, so this probe proves nothing about "
          f"that specific value")
    print(f"    b=8 historical value per RESULTS.md              = "
          f"{historical:.7e} ({100 * (historical - true8) / true8:+.1f}%)")
    print(f"    b=8 Panter-Dite asymptote 2.7207*2^-16           = {panter:.7e}")
    print(f"    ratio to Shannon: true {true8 / 2.0 ** -16:.3f}, "
          f"historical {historical / 2.0 ** -16:.3f}, bound 2.7207 "
          f"-> Panter-Dite would reject the historical value, "
          f"and the true value clears it by only "
          f"{100 * (2.7207 - true8 / 2.0 ** -16) / 2.7207:.1f}%")

    verdict(
        "G1 published-table anchor", "BLIND",
        f"The anchor covers b<=5; the sweep runs to 8, so b=6 and b=8 carry no "
        f"assertion at all — that part is structural and certain. The "
        f"consequence is not merely a missing check: the scalar MSE is G3's "
        f"threshold, so an inflated value makes the one check guarding axis C "
        f"strictly MORE permissive. Panter-Dite (2.7207*2^-2b) bounds the "
        f"unanchored rates and is currently printed as a NOTE rather than "
        f"asserted; it would reject the historical value "
        f"(ratio {historical / 2.0 ** -16:.3f} > 2.7207). Caveat on this "
        f"probe: the attempt above to reconstruct the historical defect "
        f"failed ({100 * (repro - true8) / true8:+.3f}%), so the range gap is "
        f"demonstrated and the specific value is taken from the record.")


def probe_g8_shared_bytes():
    """G8 certifies that *payload* bytes match and that side channels are
    itemised.  It says nothing about shared bytes.

    RESULTS.md's single most decision-relevant finding is that counting the
    shared codebook reverses the recall-per-byte ordering below ~350k vectors.
    That number is computed in the writeup and gated by nothing.

    The named input: a vector arm whose codebook grows without bound.  G8
    passes unchanged, because the quantity it checks does not move.
    """
    print("\n" + "=" * 74)
    print("PROBE 4 — G8 byte budget, vector codebook inflated 16x")
    print("=" * 74)
    a_v = qz.Arm(rotation="rht", norm="blockscale", codebook="scalar",
                 bits=4, d=768, seed=0)
    base_shared = a_v.shared_bytes()

    class Fat:
        m = 1
        C = np.zeros((1 << 20, 8), np.float32)   # 32 MiB codebook
        levels = np.zeros(16, np.float32)
    a_v.cb = Fat()
    fat_shared = a_v.shared_bytes()
    print(f"    shared bytes: {base_shared:,} -> {fat_shared:,} "
          f"({fat_shared / max(base_shared, 1):.0f}x)")
    calibrate.FAILURES.clear()
    calibrate.g8_budget()
    failed = len(calibrate.FAILURES)
    calibrate.FAILURES.clear()
    verdict(
        "G8 byte budget", "BLIND",
        f"G8 checks payload and per-vector side channels only ({failed} "
        f"failures with a 32 MiB codebook in play). The shared cost — the "
        f"quantity that inverts the recall-per-byte conclusion in RESULTS.md "
        f"— is computed by `shared_bytes()` and asserted nowhere. The "
        f"adversarial review's NOT-MATCHED verdict is carried in prose while "
        f"the gate reports the budget green.")


# --------------------------------------------------------------------------
# Pass 6 — how much of the suite does the known-bad actually reach?
# --------------------------------------------------------------------------


def probe_known_bad_reach():
    """G7 is the gate's only known-bad, and it exercises exactly one criterion.

    Its own docstring says "G3/G4's own criteria", but the code evaluates only
    `not (mse_bad <= mse_e8 * 0.99)` — G4's. Check what G3 would have said
    about the same under-trained grid.
    """
    print("\n" + "=" * 74)
    print("PROBE 5 — reach of the single known-bad (G7)")
    print("=" * 74)
    # The under-trained grid from G7, measured in the first run's gate.log.
    mse_bad, mse_e8 = 0.09142, 0.09120
    _, mse_sc = grids.lloyd_max_1d(2)
    g4_rejects = not (mse_bad <= mse_e8 * 0.99)
    g3_rejects = not (mse_bad < mse_sc)
    print(f"    under-trained m=8 grid: {mse_bad:.5f}")
    print(f"    G4 criterion (<= E8*0.99 = {mse_e8 * 0.99:.5f}): "
          f"{'REJECTS' if g4_rejects else 'ACCEPTS'}")
    print(f"    G3 criterion (< scalar  = {mse_sc:.5f}): "
          f"{'REJECTS' if g3_rejects else 'ACCEPTS'}")
    covered = ["G4"]
    uncovered = ["G0", "G1", "G2", "G3", "G5", "G6", "G8"]
    verdict(
        "known-bad coverage", "BLIND",
        f"One known-bad, reaching {covered}. G3 — the check axis C actually "
        f"rests on — ACCEPTS the same under-trained grid ({mse_bad:.5f} < "
        f"{mse_sc:.5f} scalar), so it is not covered despite the docstring's "
        f"claim. Unexercised: {uncovered}. G3 did fire historically on the "
        f"random-init grid, but that was a transient experiment; nothing in "
        f"the suite re-runs it, so a future change that re-loosens it is "
        f"silent.")


def main():
    print("=" * 74)
    print("AUDIT — calibrate.py, per gating/references/auditing.md")
    print("=" * 74)
    probe_g6_identity_rotation()
    probe_g3_degenerate_trainer()
    probe_g1_range()
    probe_g8_shared_bytes()
    probe_known_bad_reach()

    print("\n" + "=" * 74)
    print("SUMMARY")
    print("=" * 74)
    for c, v, _ in VERDICTS:
        print(f"  {v:<12} {c}")
    bad = [c for c, v, _ in VERDICTS if v in ("CANNOT FAIL", "BLIND")]
    print(f"\n  {len(bad)} of {len(VERDICTS)} probed checks are CANNOT FAIL or BLIND.")
    print("  Checks not probed here (G0, G2, G5) are PLAUSIBLE: each has a")
    print("  named breaking input and an external anchor, none observed red.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
