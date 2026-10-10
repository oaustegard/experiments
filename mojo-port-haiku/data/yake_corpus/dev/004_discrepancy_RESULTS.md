# RESULTS — discrepancy lower-bound records (issue #166)

Run date: 2026-07-24. All record claims below are certified in exact
arithmetic (integer / Fraction / ℤ[√2]) and independently re-verified by
`verify_certificates.py` through a separate code path. Calibration gates
G1–G9 green before any search ran (`tests/test_calibration.py`).

## Literature gate (logged first, per issue mandate)

Full findings in [issue #166 comment](https://github.com/oaustegard/claude-workspace/issues/166#issuecomment-5065826877). Load-bearing facts:

- **Komlós:** Kunisky (arXiv:2111.02974) holds K ≥ 1+√2 ≈ 2.414
  asymptotically via explicit tree matrices at n = 2^k; previous best was an
  unpublished Spielman example (n=15, disc ∈ [2.005, 2.006], no matrix
  published). SDP relaxation lower bounds provably never exceed 1 (Nikolov).
  ⇒ the honest computational deliverable is **per-size explicit certified
  K(n) lower bounds**, not "beat 2.414".
- **Beck–Fiala** (verified from Bukh's paper, primary source): BF 1981 gives
  disc ≤ 2t−2; **Bednarchak–Helm 1997: disc ≤ 2t−3 for t ≥ 3,
  unconditional**; Helm's claimed 2t−4 is unverified (Bukh: "unable to
  understand Helm's proof"); Bukh 2015: 2t−log*t for large t. 2025
  breakthroughs (Bansal–Jiang arXiv:2508.01937 + follow-up) resolve the
  conjecture for t ≥ log²n — asymptotic only; **small-t exact values were
  unpublished** as far as our searches found.

## Target B — Beck–Fiala exact small-t values (all exact)

D(t,n) = max discrepancy over set systems on n elements with element-degree
≤ t. CEGAR verdicts use m = t·n rows, which is WLOG-complete (≤ t·n nonempty
sets exist in any such system; duplicates/empties never change disc), so
every table entry is an **exact value**, not a bound. Every witness is
triple-checked: exhaustive enumeration, independent SAT UNSAT certificate,
degree audit.

| n | D(2,n) | D(3,n) | D(4,n) |
|---|--------|--------|--------|
| 3 | 2 | 2 | 2 |
| 4 | 2 | 2 | 2 |
| 5 | 2 | 2 | 2 |
| 6 | 2 | 2 | 2 |
| 7 | 2 | **3** | 3 |
| 8 | 2 | 3 | 3 |
| 9 | 2 | 3 | 3 |

**Headlines:**

1. **D(3) = 3 exactly.** The CEGAR search, starting from nothing, *rediscovered
   the Fano plane* as the witness at n=7 — and proved no degree-3 system on
   ≤ 6 elements reaches discrepancy 3, so **the Fano plane is the
   minimum-ground-set witness**. Combined with Bednarchak–Helm's
   unconditional 2t−3 = 3 upper bound, D(3) = 3 is closed. Small-t truth
   sits at **t**, not √t (√3 ≈ 1.73) and not near 2t−2 = 4.
2. **D(2) = 2** (triangle = minimal witness, n=3; exact through n=9; the
   folklore Eulerian-orientation argument gives disc ≤ 2 for all t=2
   systems, so this is D(2) closed).
3. **D(4,n≤9) = 3, so D(4) ∈ {3,4,5}** — the live open cell. The natural
   classical candidate PG(2,3) (13 points, degree 4) is weak: disc = 2
   (computed exactly). A CEGAR probe for a disc-4 degree-4 system at n = 10
   (m = 40, 5M-conflict budget per round) produced no verdict in ~40 min of
   wall time and was cut; see "Deferred" below.
4. Growth picture (`growth.png`): D(2)=2, D(3)=3 lie exactly on the D(t)=t
   line, far below 2t−3. Consistent with (but of course not evidence
   against) the O(√t) conjecture; the interesting frontier is whether
   D(4) = 4 continues the D(t) = t pattern.

Witnesses (from `bf_results.json`):
- D(3,7)=3: `{0,1,5} {0,2,6} {0,3,4} {1,2,4} {1,3,6} {2,3,5} {4,5,6}` (Fano).
- D(4,7)=3: 9-set witness incl. a *duplicated* set — duplicates are legal
  degree budget spends: `{0,1,4} {0,2,3} {0,2,3} {0,5,6} {1,2,6} {1,3,5}
  {1,4,5} {2,4,5} {3,4,6}`.

## Target A — per-size Komlós lower bounds K(n)

**A negative result first (caught before compute was wasted):** the hoped-for
"compute exact Δ of Kunisky's finite instances, maybe Δ > δ" gap is
**provably empty**: every row of Â^{T_k} has ℓ₁-norm exactly δ, so the
all-ones coloring achieves ‖Â·1‖∞ = δ while unsatisfiability forces ≥ δ —
hence Δ = δ in one line. Verified computationally k = 1..4 (Stage 1 output;
e.g. Δ(n=16) = (16+12√2)/16 = 2.06066 exactly). Kunisky's published finite
lower-bound values were already exact.

**Certified per-size records** (rational matrices, exact column norms ≤ 1,
exhaustive Fraction-arithmetic certification; envelope = best certified
value at ≤ that size):

| n | Kunisky-family baseline | certified record (this work) | exact certified value | K(n) ≥ (envelope) |
|---|---|---|---|---|
| 2 | 1.414214 | — | √2 = (0+2√2)/2 (Kunisky, exact) | 1.414214 |
| 3 | 1.414214 | **1.571296** | 14186314619/9028416320 | 1.571296 |
| 4 | 1.707107 | **1.730949** | 33345/19264 | 1.730949 |
| 5 | 1.707107 | **1.785250** | 11913415/6673248 | 1.785250 |
| 6 | 1.707107 | **1.788760** | 24373373002045/13625848532736 | 1.788760 |
| 7 | 1.707107 | **1.830151** | 3190582127717707/1743343727431680 | 1.830151 |
| 8–15 | 1.914214 | saturates at baseline | (4+8√2)/8 (Kunisky, exact) | 1.914214 |
| 16 | 2.060660 | saturates at baseline | (16+12√2)/16 (Kunisky, exact) | 2.060660 |

Full witness matrices (numerators/denominators) are in
`komlos_results.json`; `verify_certificates.py` re-derives every value by
pure-Fraction exhaustive enumeration. Search coverage: heavy-restart pass
(24 restarts × 3000 iters, β→150) ran for n = 3..12; n = 13..16 had the
lighter stage-2 pass only (6 restarts × 400 iters) — every n ≥ 8 saturated
at the padded-Kunisky value under both schedules.

We also computed δ*(n), the best value ANY Kunisky-style tree (not just
complete ones, with optimal column scaling — a small DP, `tree_dp.py`)
can certify at each size: complete trees win exactly at powers of two and
padding dominates everywhere else (e.g. δ*(7) = 1.463 < padded 1.707).
So the n = 3..7 search records exceed everything the tree construction can
produce at those sizes, not merely the specific published instances.

Notes:
- **n=4 beats Kunisky's own n=4 instance** (1.7309 > 1.7071 = Â^{T_2}):
  complete-tree matrices are not per-size optimal even at powers of two.
- n=8..16: the smoothed max-min ascent saturates exactly at the padded
  Kunisky value from every seed tried (Kunisky matrices are strong local
  optima of the piecewise-linear landscape under this ascent). Certified
  values there are the Kunisky exact values.
- At n=15 Spielman's unpublished numerical 2.005–2.006 exceeds our explicit
  certified 1.914; ours is (to our knowledge) the best *explicit, certified*
  value at n = 15, his the best claimed.
- Cross-feed check (issue's Target-B→A pipe): Fano/√3 gives disc = 3/√3
  = √3 ≈ 1.7321 at n=7 — a clean closed form, but below the n=7 search
  record; degree-t witnesses cap at D(t)/√t, so they only feed A if D(t)
  grows ≳ t.
- Search fragility: deep restarts moved n=7 from 1.7257 to 1.8302; the n=4
  record was found by one seed and not re-found by 24 later ones. Reported
  values are running maxima over all certified attempts, which is why
  fine-grained restart schedules matter more than iteration count here.

## Deferred (explicitly, not silently)

- **D(4) resolution (k=4 at n ≥ 10)**: CEGAR UNSAT/witness beyond n=9 needs
  either more solver budget, column symmetry breaking valid under CEGAR
  (nontrivial: counterexample colorings break column symmetry), or an
  orderly-generation pass (nauty genbg) instead of SAT.
- **n = 17..32 Komlós sizes**: enumeration certification is 2^{n-1}; beyond
  n ≈ 24 needs the branch-and-bound certifier sketched in the issue.
- **ℤ[√2]-valued search records**: the rationalizer only certifies over ℚ;
  a ℚ(√2) rationalizer would capture √2-structured optima losslessly
  (the n≥8 saturation suggests optima may live there).
- **Liu–Reis prefix/chairman stretch target**: untouched (primary targets
  did not saturate).

## Repro

```bash
python3 tests/test_calibration.py      # gates G1-G9
python3 run_komlos.py 16 400 6         # stage 1 + baseline per-size pass
python3 deep_search.py 3 16 24 3000    # heavy restart pass (updates JSON)
python3 run_beck_fiala.py 2,3,4 9      # exact D(t,n) table
python3 verify_certificates.py         # independent re-verification
python3 plot_growth.py                 # growth.png
```
