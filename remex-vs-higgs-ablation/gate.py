#!/usr/bin/env python3
"""Calibration gate, rebuilt on the `gating` skill's harness.

This replaces `calibrate.py` (kept alongside, unmodified, because `audit.py`'s
probes are evidence about *that* file and would be meaningless rewritten).

What changed, and why
---------------------
`calibrate.py` passed all 8 of its checks and caught a real defect in the first
run, so it was not a bad gate.  The audit found it was a gate with holes it
could not see from inside a green run:

  G6  CANNOT FAIL   compared RHT's incoherence to Haar's and to nothing else,
                    so replacing *both* rotations with the identity — the
                    worst possible incoherence — still passed.
  G3  CANNOT FAIL   asserted only `grid < scalar` with no margin.  Since the
                    trainer carries the scalar product grid as a fallback
                    candidate, a vector arm that does no vector quantization
                    at all lands within sampling noise of the threshold.
  G1  BLIND         anchored on Max (1960), which stops at 5 bits; the sweep
                    runs to 8.  The +16% defect the first run's adversarial
                    review found by hand at b=8 is still not rejected by any
                    assertion, and because the scalar MSE is G3's threshold,
                    inflating it *loosens* the check guarding axis C.
  G8  BLIND         checked payload bytes only.  Shared bytes — the quantity
                    that reverses the recall-per-byte conclusion — were
                    asserted nowhere.
  G7  one known-bad reaching one check (G4).

The three obligations from the skill, made mechanical: every check below
carries an anchor outside this code, the harness refuses to report PASS
without a known-bad it demonstrably rejected, and it refuses again without a
written statement of what it cannot catch.

Run:
    python3 gate.py            # full — precondition on reading axis C
    python3 gate.py --fast     # ~15 s subset, for mutate.py
"""
from __future__ import annotations

import argparse
import itertools
import math
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

import grids
import quantizers as qz
from _gate_harness import Gate

# Reuse the E8-ball machinery rather than reimplementing it; it is an anchor
# and a second copy would be a second thing to get wrong.
from calibrate import best_scaled_mse, codebook_mse, e8_ball_codebook

#: Panter-Dite: the high-rate asymptotic distortion of the optimal scalar
#: quantizer for a unit-variance Gaussian is (sqrt(3)*pi/2) * 2^-2b
#: = 2.7207 * 2^-2b.  Panter & Dite (1951); see also Gersho & Gray ch. 6.
#: The optimal fixed-rate quantizer approaches this FROM BELOW, so it is a
#: hard upper bound at every finite rate and the anchor that covers the rates
#: Max (1960)'s table does not reach.
PANTER_DITE = 2.7207

#: Seeds used to check the analytic standard error against an observed spread.
NOISE_SEEDS = (11, 101, 1009, 10007, 100003)


def _sq_err(C: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Per-sample squared quantization error of codebook C on samples X."""
    dist, _ = cKDTree(np.ascontiguousarray(C, np.float32)).query(
        X, k=1, workers=-1)
    return dist.astype(np.float64) ** 2


def paired_gain(C_vq: np.ndarray, C_raw: np.ndarray, n: int, seed: int = 11):
    """Compare two codebooks on the SAME samples, with the noise of the
    comparison measured from the samples themselves.

    The skill's rule is to derive a tolerance from measured noise rather than
    pick one for comfort.  The textbook way to do that here is not to re-run
    the measurement five times — it is to note that the two codebooks can be
    scored on one shared sample, which makes the comparison *paired*: the
    per-sample difference d_i = e_raw_i - e_vq_i has its own standard error
    std(d)/sqrt(n), and the common sampling fluctuation cancels instead of
    adding.  That is both cheaper (two KD-tree passes, not ten) and much
    tighter than differencing two independent estimates.

    Returns (mse_vq, mse_raw, gain_db, se_of_the_difference).
    """
    m = C_vq.shape[1]
    X = np.random.default_rng(seed).standard_normal((n, m)).astype(np.float32)
    e_vq, e_raw = _sq_err(C_vq, X), _sq_err(C_raw, X)
    d = e_raw - e_vq
    mse_vq, mse_raw = float(e_vq.mean()) / m, float(e_raw.mean()) / m
    se = float(d.std(ddof=1) / math.sqrt(n)) / m
    gain_db = 10 * math.log10(mse_raw / mse_vq) if mse_vq > 0 else float("inf")
    return mse_vq, mse_raw, gain_db, se


# --------------------------------------------------------------------------


def build(fast: bool) -> Gate:
    g = Gate("remex-vs-higgs axis-C calibration" + (" [FAST]" if fast else ""))
    # Fast mode trades statistical power for turnaround: it exists so
    # `mutate.py` can run the gate a hundred times, not to certify anything.
    N = 40_000 if fast else 1_000_000
    NE8 = 25_000 if fast else 600_000
    DIMS = (100,) if fast else (100, 768, 1024)

    # ---- the measurement instrument, before anything it measures ---------
    # Lifting the scalar levels into an m-dim product grid makes the sampled
    # KD-tree path and the closed-form integral measure the same quantity, by
    # two code paths that share nothing.  Exonerate the instrument first, or a
    # later disagreement has two suspects instead of one.
    pairs = ((2, 2),) if fast else ((2, 2), (4, 2), (6, 2), (2, 4))
    for b, m in pairs:
        _, closed = grids.lloyd_max_1d(b)
        g.anchor(f"MSE instrument, b={b} m={m} product grid",
                 measured=codebook_mse(grids.product_init(b, m), n=N),
                 published=closed, rel_tol=2e-2,
                 source="closed-form integration against the normal density")

    # The tolerance every grid check below leans on is an analytic standard
    # error, so exonerate that too.  The honest test is not "is the paired se
    # smaller than an unpaired one" -- those estimate different quantities, and
    # comparing them was an error in the first draft of this gate, caught by
    # this check going red.  What has to hold is that the analytic se of the
    # DIFFERENCE, computed from a single stream, predicts the spread of that
    # same difference actually observed across independent streams.
    Pa, Pb = grids.product_init(2, 2), grids.product_init(1, 2)
    seeds = NOISE_SEEDS[:3] if fast else NOISE_SEEDS
    runs = [paired_gain(Pa, Pb, n=N, seed=s) for s in seeds]
    diffs = [r[1] - r[0] for r in runs]
    se_obs = float(np.std(diffs, ddof=1))          # spread of a single estimate
    se_pred = runs[0][3]
    g.note(f"noise model: analytic paired se {se_pred:.3e} vs observed spread "
           f"of the same difference over {len(seeds)} independent streams "
           f"{se_obs:.3e} (ratio {se_pred / se_obs:.2f}; with {len(seeds)} "
           f"draws the observed spread is itself only good to ~"
           f"{100 / math.sqrt(2 * (len(seeds) - 1)):.0f}%)")
    g.check(0.33 < se_pred / se_obs < 3.0,
            "analytic paired se predicts the observed spread of the difference "
            "[anchor: repeated measurement]",
            f"analytic={se_pred:.3e} observed={se_obs:.3e} "
            f"ratio={se_pred / se_obs:.2f}", kind="anchor")

    # ---- scalar arm: published table, within its range -------------------
    for b, pub in sorted(grids.MAX_1960_MSE.items()):
        _, mse = grids.lloyd_max_1d(b)
        # 3e-3 rather than 2e-3 because of b=5 alone: ours is 0.24% above the
        # table and it is the table that is imprecise (converged to residual
        # 6.5e-8, bit-stable over 2e3..2e5 iterations, and the Gaussian's
        # log-concavity makes the Lloyd-Max fixed point unique -- Fleischer
        # 1964).  Any real defect here is percent-scale.
        g.anchor(f"scalar Lloyd-Max b={b}", measured=mse, published=pub,
                 rel_tol=3e-3, source="Max (1960) table 1")

    # ---- scalar arm ABOVE the table's range ------------------------------
    # This is the hole the audit found, and it is where the first run's
    # 16%-high value lived.  Panter-Dite bounds the rates Max does not reach.
    hi_rates = (6,) if fast else (6, 8)
    ratios = []
    for b in hi_rates:
        _, mse = grids.lloyd_max_1d(b)
        r = mse / 2.0 ** (-2 * b)
        ratios.append((b, r))
        g.bracket(f"scalar Lloyd-Max b={b} vs Panter-Dite [anchor: "
                  f"Panter & Dite (1951)]",
                  value=r, lo=2.0, hi=PANTER_DITE,
                  why="optimal fixed-rate scalar quantizer approaches "
                      "2.7207*2^-2b from below; above it means the quantizer "
                      "is not optimal or the distortion is misevaluated")
    if len(ratios) > 1:
        g.check(all(a[1] < b_[1] for a, b_ in itertools.pairwise(ratios)),
                "scalar ratio-to-Shannon increases with rate",
                " < ".join(f"{r:.4f}(b={b})" for b, r in ratios),
                kind="bracket")

    # ---- vector machinery: a published lattice constant ------------------
    g.anchor("E8 normalised second moment", measured=grids.e8_nsm(n=NE8),
             published=0.0716821, rel_tol=8e-3 if fast else 5e-3,
             source="Conway & Sloane, SPLAG table 2.3")

    # ---- the grids the sweep actually uses -------------------------------
    if fast:
        want = [(2, 2)]
    else:
        want, seen = [], set()
        for d in DIMS:
            for b in (1, 2, 3, 4, 6, 8):
                m = grids.pick_m(b, d)
                if m == 1 or (m, 1 << (b * m)) in seen:
                    continue
                seen.add((m, 1 << (b * m)))
                want.append((b, m))

    def grid_verdict(C, b, m, n):
        """The axis-C criterion, as ONE function.

        Both the real check below and the known-bad at the bottom call this,
        so the known-bad exercises the criterion that actually runs rather
        than a second statement of it that could drift from it.  Returns
        (accepted, detail, parts).
        """
        mse_vq = codebook_mse(C, n=n)
        shannon = 2.0 ** (-2 * b)
        # The degenerate baseline: the scalar quantizer lifted to m dimensions.
        # This is what the trainer falls back to, so it is exactly what a
        # broken vector arm looks like -- which makes it the right thing to
        # require a margin over.  `grid < scalar` alone cannot tell the two
        # apart; `grid < product_raw - 3 se` can.
        P = grids.product_init(b, m)
        _, mse_raw, gain_db, se = paired_gain(C, P, n=n)
        floor = mse_raw - 3.0 * se
        ok = shannon < mse_vq < floor
        return ok, (f"{shannon:.6g} < {mse_vq:.6g} < {floor:.6g} — gain over "
                    f"the lifted scalar grid {gain_db:+.2f} dB must clear 3 se "
                    f"of the paired estimator (se={se:.2e}); cannot beat the "
                    f"rate-distortion bound 2^-2b"), (mse_vq, gain_db, se)

    for b, m in want:
        K = 1 << (b * m)
        C, _ = grids.train_gaussian_grid(m, K, log=None)
        _, mse_sc = grids.lloyd_max_1d(b)
        ok, detail, (mse_vq, _, _) = grid_verdict(C, b, m, N)
        g.check(ok, f"grid b={b} m={m} K={K}: real VQ gain, below Shannon",
                detail, kind="bracket")
        g.check(mse_vq < mse_sc, f"grid b={b} m={m} beats scalar (closed form)",
                f"vq={mse_vq:.6g} scalar={mse_sc:.6g}", kind="check")
        g.note(f"b={b} m={m} K={K}: grid={mse_vq:.6g} scalar={mse_sc:.6g} "
               f"gain={10 * math.log10(mse_sc / mse_vq):+.2f} dB "
               f"ratio-to-Shannon={mse_vq / 2.0 ** (-2 * b):.3f}")

    # ---- the grid family vs a published codebook family ------------------
    if not fast:
        C8 = e8_ball_codebook(1 << 16)
        mse_e8, s = best_scaled_mse(C8)
        Cg, _ = grids.train_gaussian_grid(8, 1 << 16, log=None)
        mse_grid = codebook_mse(Cg, n=N)
        _, mse_sc2 = grids.lloyd_max_1d(2)
        g.note(f"E8 ball codebook (2^16 pts, best scale {s:.3f}): "
               f"mse/dim={mse_e8:.5f}")
        # 1%, not 2%: the trained grid earns the granular gain the lattice has
        # PLUS the density shaping a uniform-density lattice ball cannot, so a
        # real one clears E8 by a margin.  The old 2% slack was generosity and
        # it cost the check its power -- the under-trained known-bad passed it.
        g.check(mse_grid <= mse_e8 * 0.99, "trained m=8 grid beats a tuned E8 "
                "ball by >=1% [anchor: QuIP# E8P codebook family]",
                f"grid={mse_grid:.5f} E8={mse_e8:.5f} "
                f"ratio={mse_grid / mse_e8:.3f}", kind="anchor")
        g.check(mse_e8 < mse_sc2, "the E8 anchor itself beats scalar",
                f"E8={mse_e8:.5f} scalar={mse_sc2:.5f}", kind="anchor")

        # monotone in sub-vector dimension -- an ordering theory requires
        _, sc = grids.lloyd_max_1d(2)
        seq, prev = [(1, sc)], None
        for m in (2, 4, 8):
            C, _ = grids.train_gaussian_grid(m, 1 << (2 * m), log=None)
            seq.append((m, codebook_mse(C, n=N)))
        for m, v in seq:
            if prev is not None:
                g.check(v < prev[1] * 1.001, f"quality monotone in m: "
                        f"m={m} <= m={prev[0]}", f"{v:.5f} vs {prev[1]:.5f}",
                        kind="bracket")
            prev = (m, v)
    else:
        mse_e8 = None

    # ---- rotations: absolute anchors, not just RHT vs Haar ---------------
    rng = np.random.default_rng(3)
    for d in DIMS:
        E = np.zeros((64, d), np.float32)
        E[np.arange(64), rng.choice(d, 64, replace=False)] = 1.0
        X = rng.standard_normal((256, d)).astype(np.float32)
        X /= np.linalg.norm(X, axis=1, keepdims=True)

        # The reference an absolute incoherence check needs: what max|coord|
        # SHOULD be after a good rotation.  Computed by drawing uniformly
        # random unit vectors directly -- Gaussian, then normalise -- which
        # touches none of the rotation code under test.  A Haar rotation maps
        # any fixed unit vector to exactly this distribution, so it is the
        # ideal both arms are measured against, and it is manufactured ground
        # truth rather than a value read off the subject.
        U = rng.standard_normal((4096, d))
        U /= np.linalg.norm(U, axis=1, keepdims=True)
        ideal = float(np.mean(np.max(np.abs(U), axis=1)))
        g.note(f"d={d}: ideal E[max|coord|] of a uniform unit vector = "
               f"{ideal:.4f} (independent draw, no rotation code involved); "
               f"1/sqrt(d)={1 / math.sqrt(d):.4f}")

        spike = {}
        for kind in ("haar", "rht"):
            R = qz.ROTATIONS[kind](d, 5)
            spike[kind] = float(np.mean(np.max(np.abs(R.apply(E)), axis=1)))

            # conservation law: an orthogonal map preserves the norm, and
            # inverse(apply(x)) == x.  Cheap, and it rules out a large family
            # of indexing and sign errors in one line.
            Y = R.apply(X)
            g.check(float(np.max(np.abs(np.linalg.norm(Y, axis=1) - 1.0))) < 2e-3,
                    f"{kind} d={d} preserves the norm [anchor: orthogonality]",
                    f"max |‖Rx‖-1| = "
                    f"{float(np.max(np.abs(np.linalg.norm(Y, axis=1) - 1.0))):.2e}",
                    kind="anchor")
            g.check(float(np.max(np.abs(R.inverse(Y) - X))) < 2e-3,
                    f"{kind} d={d} round-trips [anchor: R^-1 R = I]",
                    f"max |R^-1Rx - x| = "
                    f"{float(np.max(np.abs(R.inverse(Y) - X))):.2e}",
                    kind="anchor")

            # ABSOLUTE incoherence, two probes.
            #
            # (i) a coordinate spike.  Upper edge is 2x the independently
            # computed ideal: a factor of 2 in mu is a factor of 4 in the
            # incoherence-driven error term, which is where a rotation stops
            # doing the job axis A assumes it does.  The identity sits at 1.0,
            # 3.6-9x the ideal, so it is rejected with room to spare.  The
            # lower edge 1/sqrt(d) is a hard floor -- coordinates square to 1
            # -- and it is INCLUSIVE, because a Hadamard transform attains it
            # exactly: at a power-of-two d the block spans the whole vector,
            # one round maps a delta to a perfectly flat +-1/sqrt(d), and a
            # strict bound turned the optimum into a failure.  That is what
            # this check did on its first full run at d=1024.
            floor_mu = 1.0 / math.sqrt(d)
            g.check(floor_mu * (1 - 1e-9) <= spike[kind] < 2.0 * ideal,
                    f"{kind} d={d} spike incoherence [anchor: uniform unit "
                    f"vector, ideal {ideal:.4f}]",
                    f"{floor_mu:.6g} <= {spike[kind]:.6g} < {2 * ideal:.6g} — "
                    f"a rotation that fails to spread a coordinate spike "
                    f"leaves max|coord| near 1, and axis A then compares two "
                    f"no-ops", kind="bracket")

            # (ii) a random unit vector.  For a power-of-two d the spike probe
            # above is degenerate for the RHT -- a Hadamard transform flattens
            # a delta by construction, so it passes at the floor no matter how
            # well the rest of the rotation mixes.  A random input is the
            # discriminating case, and it is bracketed on BOTH sides of the
            # ideal rather than only from above.
            mu_rand = float(np.mean(np.max(np.abs(R.apply(X)), axis=1)))
            g.bracket(f"{kind} d={d} random-vector incoherence [anchor: "
                      f"uniform unit vector, ideal {ideal:.4f}]",
                      value=mu_rand, lo=0.5 * ideal, hi=2.0 * ideal,
                      why="the probe the spike test cannot supply at "
                          "power-of-two d, where FWHT flattens a delta by "
                          "construction")
        g.check(spike["rht"] < spike["haar"] * 1.25,
                f"d={d} RHT incoherence within 25% of Haar's",
                f"rht={spike['rht']:.4f} haar={spike['haar']:.4f}",
                kind="check")

    # ---- the codec itself, not just the codebook -------------------------
    # Everything above scores point sets.  Mutation testing found that the
    # gate never ran the ENCODER: mutating ScalarCodebook.encode_decode's
    # decision boundaries (grids.py:166), its size==1 branch (:170),
    # VectorCodebook's query k (:194) and Arm.encode_decode's norm-mode
    # branch (quantizers.py:232) all left the gate green.  Those are the
    # functions the sweep runs on every vector of every corpus.
    rq = np.random.default_rng(17)
    codec_cases = (("scalar", 2, 64), ("vector", 2, 100)) if fast else \
                  (("scalar", 2, 64), ("scalar", 4, 64), ("vector", 2, 100))
    for kind, bl, dl in codec_cases:
        cb = grids.build_codebook(kind, bl, dl, log=None)
        Z = rq.standard_normal((4000, dl)).astype(np.float32)
        R1 = cb.encode_decode(Z)

        # (a) idempotence.  The reconstruction is already a codepoint, so
        # re-encoding cannot move it.  A conservation law, and it fails the
        # moment the decision boundaries disagree with the levels they were
        # derived from.
        drift = float(np.max(np.abs(cb.encode_decode(R1) - R1)))
        g.check(drift == 0.0, f"{kind} b={bl} d={dl} codec is idempotent "
                f"[anchor: Q(Q(x)) = Q(x)]", f"max drift {drift:.2e}",
                kind="anchor")

        # (b) membership: every reconstructed value must BE a codepoint.
        if kind == "scalar":
            member, n_cp = bool(np.isin(R1, cb.levels).all()), cb.levels.size
        else:
            dd, _ = cKDTree(cb.C).query(R1.reshape(-1, cb.m), k=1, workers=-1)
            member, n_cp = float(np.max(dd)) < 1e-6, cb.C.shape[0]
        g.check(member, f"{kind} b={bl} d={dl} output is drawn from the "
                f"codebook [anchor: definition of a quantizer]",
                f"{n_cp} codepoints", kind="anchor")

        # (c) the encoder attains what the point set promises.  Measured
        # distortion through encode_decode vs the KD-tree MSE of the
        # codebook's own points -- two disjoint paths.  A mis-derived
        # boundary picks a sub-optimal codepoint, which shows up here and
        # nowhere else.
        got = float(np.mean((Z - R1) ** 2))
        want = grids.lloyd_max_1d(bl)[1] if kind == "scalar" \
            else codebook_mse(cb.C, n=N)
        g.bracket(f"{kind} b={bl} d={dl} encoder attains the codebook's own "
                  f"distortion", value=got, lo=want * 0.9, hi=want * 1.1,
                  why="the point-set MSE cannot see a wrong decision boundary")

    # ---- the full arm round-trip -----------------------------------------
    for nk in ("exactnorm", "blockscale"):
        arm = qz.Arm(rotation="haar", norm=nk, codebook="scalar", bits=3,
                     d=64, seed=0)
        Xa = rq.standard_normal((2000, 64)).astype(np.float32) * 3.0
        Xh = arm.encode_decode(Xa)
        rel = float(np.mean(np.sum((Xh - Xa) ** 2, axis=1)
                            / np.sum(Xa ** 2, axis=1)))
        _, mse3 = grids.lloyd_max_1d(3)
        # The rotation is orthogonal and a rotated unit vector has coordinate
        # variance exactly 1/d, so relative squared error must land on the
        # codebook's own per-coordinate MSE.  Ties the arm to the published
        # scalar anchor instead of to itself.
        g.bracket(f"arm({nk}) round-trip error == codebook MSE [anchor: "
                  f"orthogonality + Lloyd-Max]",
                  value=rel, lo=mse3 * 0.75, hi=mse3 * 1.35,
                  why="rotate/scale/quantize/unrotate must lose nothing "
                      "beyond the quantizer itself")

    # remex's 1-bit MIPS result rests on the exact-norm arm reproducing
    # document norms with ZERO spread.  That is a property of encode_decode
    # and was asserted only in prose.
    a1 = qz.Arm(rotation="haar", norm="exactnorm", codebook="scalar", bits=1,
                d=64, seed=0)
    X1 = rq.standard_normal((2000, 64)).astype(np.float32) * 3.0
    ratio = (np.linalg.norm(a1.encode_decode(X1), axis=1)
             / np.linalg.norm(X1, axis=1))
    g.check(float(np.std(ratio)) < 1e-5,
            "1-bit exact-norm arm reproduces relative norms exactly "
            "[anchor: constant-modulus code]",
            f"mean={float(np.mean(ratio)):.5f} std={float(np.std(ratio)):.2e} "
            f"(sqrt(2/pi)=0.79788)", kind="bracket")

    # ---- the floor control -----------------------------------------------
    # `uniform_levels` is the naive-uniform FLOOR the result leans on ("the
    # floor sits below remex everywhere, so the rotation and the Lloyd-Max
    # levels are both doing work").  Every mutation to it survived, because
    # the gate never called it.
    for bl in (2, 4):
        u = grids.uniform_levels(bl)
        mu = grids._scalar_mse_gaussian(u)
        _, ml = grids.lloyd_max_1d(bl)
        bad = grids._scalar_mse_gaussian(grids.uniform_levels(bl, clip=0.5))
        g.bracket(f"uniform floor control b={bl} sits between Lloyd-Max and a "
                  f"mis-clipped uniform", value=mu, lo=ml, hi=bad,
                  why="the floor must be a TUNED uniform quantizer: worse than "
                      "optimal scalar, better than an arbitrary clip, or "
                      "'remex beats the floor' is not evidence")

    # ---- the replication control must stay dominated ---------------------
    # LM+QJL is carried to replicate the settled 2026-04-02 result: `prod` is
    # strictly dominated at every bit width.  A harness that made it look
    # competitive would be broken, so gate it rather than only observing it.
    aq = qz.QJLArm(bits=4, d=64, seed=0)
    asc = qz.Arm(rotation="haar", norm="exactnorm", codebook="scalar",
                 bits=4, d=64, seed=0)
    Xq = rq.standard_normal((1500, 64)).astype(np.float32) * 3.0
    eq = float(np.mean(np.sum((aq.encode_decode(Xq) - Xq) ** 2, axis=1)))
    es = float(np.mean(np.sum((asc.encode_decode(Xq) - Xq) ** 2, axis=1)))
    g.check(eq > es, "LM+QJL control stays dominated by plain scalar at a "
            "matched budget [anchor: settled result, 2026-04-02]",
            f"qjl={eq:.5f} scalar={es:.5f} (both 4 bits/coord total)",
            kind="anchor")

    # ---- the anchor table cannot quietly shrink --------------------------
    # Mutating a KEY of MAX_1960_MSE (grids.py:39) dropped a rate from the
    # published-table check and the gate reported PASS over fewer rows.
    g.check(sorted(grids.MAX_1960_MSE) == [1, 2, 3, 4, 5],
            "the Max (1960) anchor table still covers b=1..5",
            f"keys={sorted(grids.MAX_1960_MSE)}", kind="anchor")

    # ---- byte budget, including the shared bytes -------------------------
    # b=4 at d=100 picks m=4, K=65536, and fast mode bypasses the grid cache
    # (see main()), so including it would retrain a 65k-codepoint grid on every
    # mutation run.  Fast mode keeps b=2 (m=5, K=1024) and says so.
    for d in DIMS:
        for b in ((2,) if fast else (2, 4)):
            a_s = qz.Arm(rotation="haar", norm="exactnorm", codebook="scalar",
                         bits=b, d=d, seed=0)
            a_v = qz.Arm(rotation="rht", norm="blockscale", codebook="vector",
                         bits=b, d=d, seed=0)
            bs, bv = a_s.bytes_per_vector(), a_v.bytes_per_vector()
            g.check(bs["payload"] == bv["payload"], f"d={d} b={b} payload matched",
                    f"scalar={bs['payload']} vector={bv['payload']}")
            # `d * bits / 8` -> `d / bits / 8` (quantizers.py:206) SURVIVED,
            # because the check above compares the two arms and both come from
            # the same expression, so both moved together.  That is the
            # "assertion restates the implementation" anti-pattern, and it was
            # in the rebuilt gate, not the old one.  Compare to arithmetic.
            g.check(bs["payload"] == d * b / 8.0 and bv["payload"] == d * b / 8.0,
                    f"d={d} b={b} payload == d*bits/8 [anchor: arithmetic]",
                    f"scalar={bs['payload']} vector={bv['payload']} "
                    f"expected={d * b / 8.0}", kind="anchor")
            # :208 `payload + side` -> `payload - side` SURVIVED; nothing read
            # `total` at all.
            g.check(bs["total"] == bs["payload"] + bs["side"]
                    and bv["total"] == bv["payload"] + bv["side"],
                    f"d={d} b={b} total == payload + side",
                    f"scalar={bs['total']} vector={bv['total']}", kind="bracket")
            # :177 `cap = min(BLOCK, d // 2)` -> `max` SURVIVED: at d=100 that
            # collapses the arm to ONE block, so the per-block-scale arm
            # becomes a global scale and axis B stops testing block
            # granularity.  The byte check cannot see it -- side == 2*nblocks
            # holds just as well for nblocks == 1.
            g.check(a_v.nblocks >= 2, f"d={d} b={b} blockscale arm has >1 block",
                    f"nblocks={a_v.nblocks} block={a_v.block}", kind="bracket")
            g.check(bs["side"] == 4.0 and bv["side"] == 2.0 * a_v.nblocks,
                    f"d={d} b={b} side channels itemised",
                    f"exactnorm={bs['side']}B blockscale={bv['side']}B "
                    f"({a_v.nblocks}x{a_v.block})")
            g.check(a_v.block % a_v.cb.m == 0,
                    f"d={d} b={b} block tiles the sub-vector dim",
                    f"block={a_v.block} m={a_v.cb.m}")

            # The audit's fourth finding.  RESULTS.md's most decision-relevant
            # claim -- that counting shared bytes reverses the recall-per-byte
            # ordering -- rests on shared_bytes(), which nothing asserted.
            # Check it against the analytic cost, term by term.
            want_s = d * d * 4 + (1 << b) * 4
            want_v = len(a_v.R.perms) * d * 5 + a_v.cb.C.size * 4
            g.check(a_s.shared_bytes() == want_s,
                    f"d={d} b={b} scalar shared bytes == d²·4 + 2^b·4",
                    f"{a_s.shared_bytes():,} vs {want_s:,}", kind="bracket")
            g.check(a_v.shared_bytes() == want_v,
                    f"d={d} b={b} vector shared bytes == perms·d·5 + K·m·4",
                    f"{a_v.shared_bytes():,} vs {want_v:,}", kind="bracket")
            # The claim RESULTS.md actually leans on is about the CODEBOOK
            # term, not total shared bytes.  Asserting the total was an
            # overreach and this check caught it: at d=100, b=2 the vector
            # arm's grid is 20 KiB while remex's Haar matrix is 40 KiB, so the
            # total runs the other way.  The codebook comparison is the one
            # that holds at every configuration (K = 2^(b*m) >= 2^b).
            cb_v, cb_s = a_v.cb.C.size * 4, a_s.cb.levels.size * 4
            g.check(cb_v > cb_s, f"d={d} b={b} vector codebook exceeds the "
                    f"scalar level table", f"{cb_v:,}B vs {cb_s:,}B",
                    kind="bracket")
            g.note(f"d={d} b={b} shared bytes: remex(haar+scalar)="
                   f"{a_s.shared_bytes():,}B "
                   f"HIGGS-like(rht+vector)={a_v.shared_bytes():,}B "
                   f"— the rotation term dominates below b=3 at d=100, which "
                   f"is why the amortization table must be read per bit width")

    # ======================================================================
    # KNOWN-BADS — break it the way it would plausibly break, confirm red
    # ======================================================================

    # KB1: the original G7.  An 8-dim grid stopped after 2 Lloyd iterations
    # from a random init is the plausible under-implemented VQ arm the issue
    # warns about.  Retained because it is the one the first run relied on.
    if not fast:
        rngb = np.random.default_rng(4242)
        Xb = rngb.standard_normal((400_000, 8)).astype(np.float32)
        C_bad, _ = grids._lloyd(Xb, 1 << 16, 2, rngb)
        mse_bad = codebook_mse(C_bad, n=N)
        g.known_bad("an under-trained m=8 grid is rejected by the E8 anchor",
                    rejected=not (mse_bad <= mse_e8 * 0.99),
                    detail=f"bad={mse_bad:.5f} > E8*0.99={mse_e8 * 0.99:.5f}")

    # KB2: the CANNOT-FAIL the audit found in G6.  Both rotations replaced by
    # the identity is the shared failure a relative check is blind to.
    d0 = DIMS[0]
    rk = np.random.default_rng(1)
    Ei = np.zeros((64, d0), np.float32)
    Ei[np.arange(64), rk.choice(d0, 64, replace=False)] = 1.0
    Ui = rk.standard_normal((4096, d0))
    Ui /= np.linalg.norm(Ui, axis=1, keepdims=True)
    ideal0 = float(np.mean(np.max(np.abs(Ui), axis=1)))
    ident = float(np.mean(np.max(np.abs(qz.IdentityRotation(d0, 0).apply(Ei)), axis=1)))
    g.known_bad("an identity 'rotation' is rejected by the incoherence bracket",
                rejected=not (1.0 / math.sqrt(d0) < ident < 2.0 * ideal0),
                detail=f"identity spike incoherence {ident:.4f} outside "
                       f"({1.0 / math.sqrt(d0):.4f}, {2 * ideal0:.4f}) — the "
                       f"old relative check passed this, because both arms "
                       f"were equally broken")

    # KB3: the CANNOT-FAIL the audit found in G3.  A vector arm whose Lloyd
    # stage does nothing degrades to the scalar product grid, not to something
    # obviously bad -- so it must be rejected by the margin, not the sign.
    b0, m0 = (2, 2) if fast else (2, 8)
    P0 = grids.product_init(b0, m0)
    # A trainer whose Lloyd stage contributes nothing: the arm degrades to the
    # scalar quantizer lifted to m dimensions, which is the trainer's own
    # fallback candidate and therefore the realistic shape of a broken vector
    # arm *after* the product-init fix -- not visibly bad, just the scalar arm
    # wearing an m-dimensional hat.  Run through the same criterion the live
    # grids face, so the known-bad cannot drift from the check it certifies.
    #
    # An earlier version used one Lloyd iteration on a small sample instead,
    # to avoid the degenerate zero standard error below.  That was wrong and
    # the gate caught it: at m=8 with K=65536 and 2,000 samples, one iteration
    # relocates ~63k empty-cell codepoints toward the mode and earns a real
    # +0.10 dB, so it is not a zero-gain arm at all.
    accepted, detail0, (_, gain0, _) = grid_verdict(P0, b0, m0, N)
    _, sc0 = grids.lloyd_max_1d(b0)
    mse_flat = codebook_mse(P0, n=N)
    g.known_bad("a vector arm with zero quantization gain is rejected by the "
                "SAME criterion the real grids face",
                rejected=not accepted,
                detail=f"unrefined product grid gains {gain0:+.3f} dB -> "
                       f"{'REJECTED' if not accepted else 'ACCEPTED'}. It "
                       f"DOES pass the old `< scalar` test "
                       f"({mse_flat:.6g} < {sc0:.6g}), which is why that test "
                       f"certified nothing. [{detail0}]")

    # KB4: the historical 8-bit defect.  4.791e-5 is the value RESULTS.md
    # records as having shipped in the first build, taken as a literal from
    # the record -- `audit.py` tried to reconstruct it from the fixed-point
    # identity and could not, so it is used here as a documented wrong value
    # rather than a regenerated one.  Max (1960) stops at 5 bits, so before
    # the Panter-Dite bracket nothing in the gate could see it at all.
    bad_b, bad_mse = 8, 4.791e-5
    g.known_bad("the b=8 scalar MSE that shipped in the first run is rejected "
                "by Panter-Dite",
                rejected=not (bad_mse / 2.0 ** (-2 * bad_b) < PANTER_DITE),
                detail=f"recorded pre-fix ratio "
                       f"{bad_mse / 2.0 ** (-2 * bad_b):.3f} >= {PANTER_DITE}; "
                       f"the published table's range ends at b=5 so G1 could "
                       f"never see this")

    # KB5: an arm that forgets to charge for its codebook.
    class _Free:
        m, C, levels = 1, None, np.zeros(4, np.float32)
    a_free = qz.Arm(rotation="rht", norm="blockscale", codebook="vector",
                    bits=2, d=d0, seed=0)
    real = a_free.shared_bytes()
    a_free.cb = _Free()
    g.known_bad("an arm that drops its codebook from shared bytes is rejected",
                rejected=a_free.shared_bytes() != real,
                detail=f"{a_free.shared_bytes():,}B without the grid vs "
                       f"{real:,}B with it")

    # ======================================================================
    # WHAT THIS GATE CANNOT CATCH
    # ======================================================================
    g.coverage(
        "Max (1960) covers b<=5 only. Above that the anchor is Panter-Dite, "
        "which is asymptotic: it bounds the scalar MSE from above but does not "
        "pin it, so an error smaller than the gap to the asymptote "
        "(~2.6% at b=6, ~0.6% at b=8) is invisible.")
    g.coverage(
        "Every codebook check scores against N(0, I). If the rotated corpus "
        "coordinates are not Gaussian — heavy tails, residual anisotropy the "
        "rotation failed to kill — every grid here is calibrated for the wrong "
        "source and all of these checks still pass. Nothing in the gate looks "
        "at a real corpus.")
    g.coverage(
        "The incoherence bracket is evaluated on coordinate spikes and random "
        "unit vectors, not on the corpora. A rotation that spreads those but "
        "interacts badly with real embedding geometry passes.")
    g.coverage(
        "The incoherence upper edge is 2x the ideal. That catches a rotation "
        "that has stopped working (the identity is 3.6-9x out) but not one "
        "that is merely mediocre: a rotation leaving max|coord| up to twice "
        "the Haar value passes, and axis A would read it as a fair comparison.")
    g.coverage(
        "The spike probe is degenerate for the RHT at power-of-two d: the "
        "Hadamard block spans the whole vector, so one round maps a delta to "
        "exactly +-1/sqrt(d) and the check passes at the floor however badly "
        "the permutation and sign stages behave. The random-vector probe "
        "covers that case; the spike probe alone would not.")
    g.coverage(
        "Byte accounting is checked against an analytic formula, not against "
        "bytes actually written to disk. Nothing here serialises an index, so "
        "a codec whose real encoding is larger than its accounting says would "
        "pass.")
    g.coverage(
        "The recall pipeline itself — top-k, recall@k, Spearman — is outside "
        "this gate; it is anchored by the fp32 control (recall 1.000 by "
        "construction) and the LM+QJL replication control in run_ablation.py, "
        "neither of which runs here.")
    g.coverage(
        "Seed variance of the trained grids is not gated. Grids are trained "
        "once at a fixed seed; a seed-sensitive trainer would show up as a "
        "quiet shift in axis C, not as a failure.")
    g.coverage(
        "STATISTICAL, NOT PRACTICAL. The axis-C margin is 3 standard errors of "
        "a PAIRED estimator, and that se shrinks as the two codebooks become "
        "similar — measured, a product grid perturbed by N(0, 1e-3) gains "
        "+0.0001 dB against a 3-se margin of 1.2e-06 and is ACCEPTED. So the "
        "check certifies 'the gain is real', not 'the gain is worth having'. "
        "Real grids here gain 0.35-1.41 dB; an arm gaining 0.01 dB would pass. "
        "The only practical floor in this gate is the E8-ball anchor, and it "
        "constrains one configuration (m=8 at 2 bits/coord), not all nine.")
    # ---- survivors from mutate.py that are NOT fixed, stated instead -----
    g.coverage(
        "MUTATION SURVIVOR — tuning constants. GRID_VERSION, K_MAX, "
        "M_CANDIDATES, the Lloyd iteration count and the training-sample "
        "budget can all be changed without any check firing. They select "
        "WHICH grid gets built rather than whether it is correct, and every "
        "grid actually built is then bracketed against scalar and Shannon. "
        "A change here silently alters the configuration the sweep reports, "
        "so RESULTS.md must state m and K per rate, which it does.")
    g.coverage(
        "MUTATION SURVIVOR — `_lloyd`'s in-loop training MSE and its logging "
        "and convergence probes (grids.py:232-254). The training distortion "
        "is discarded: selection and reporting both use held_out_mse. So "
        "those lines are genuinely dead for correctness, and mutations to "
        "them are equivalent mutants rather than gaps.")
    g.coverage(
        "MUTATION SURVIVOR — block-size selection (quantizers.py:186-187) "
        "beyond the >1-block floor now gated. Shrinking the candidate range "
        "changes d=100's block from 50 to 25: still a legal tiling, still "
        ">1 block, but a different axis-B granularity than the one reported. "
        "Gated for validity, not for identity.")
    g.coverage(
        "MUTATION SURVIVOR — the QJL replication control's internals "
        "(quantizers.py:260-297: sketch scale, residual norm, the "
        "sqrt(pi/2) debiasing). The gate now only checks that the control "
        "stays DOMINATED by plain scalar, which is the property the sweep "
        "reads it for. A QJL arm that is wrong in a way that keeps it "
        "dominated would pass, and the sweep would report a control that is "
        "correctly ordered for the wrong reason.")
    g.coverage(
        "Equivalent mutants confirmed by inspection, not gaps: the sign in "
        "_trunc_second_moment (grids.py:152-154) is unobservable because "
        "Sum(fb - fa) telescopes to zero for any symmetric level set; "
        "`* sign` -> `/ sign` (quantizers.py:120) is identity for sign = ±1; "
        "and numpy treats reshape(-2, B) exactly like reshape(-1, B).")
    if fast:
        g.coverage(
            "FAST mode: d=768/1024, b=4 and b=8, the E8 ball anchor, the "
            "m-monotonicity ordering and the under-trained-grid known-bad are "
            "all skipped, and every sample count is cut ~25x. Fast mode exists "
            "so mutate.py can run the gate a hundred times; it is not a "
            "release gate and nothing should be published on its verdict.")
    return g


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true",
                    help="~15s subset for mutate.py; NOT a release gate")
    args = ap.parse_args()

    tmp = None
    if args.fast:
        # Bypass the on-disk grid cache.  A cache keyed on (m, K) cannot notice
        # that the trainer changed, so mutation-testing the trainer against a
        # warm cache would score every mutation as a survivor for the wrong
        # reason.  This is the skill's "cache keyed on the problem rather than
        # the method" anti-pattern, and it applies to the gate's own tooling.
        tmp = Path(tempfile.mkdtemp(prefix="gridcache-"))
        grids.GRID_CACHE = tmp
    try:
        return build(args.fast).report()
    finally:
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
