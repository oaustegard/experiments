# SSUF β* hunt — results (2026-07-24)

Handoff: [claude-workspace#165](https://github.com/oaustegard/claude-workspace/issues/165).
Follow-on to the Goemans/DGG conjecture disproof (2026-07-22) and #163 (Woodall
τ=3 search, negative). This run is autonomous (scheduled GitHub-webhook trigger,
no human in the loop) — scope was kept to what could be *verified*, not what
could be *claimed*.

## TL;DR

- **Literature gate: done.** Key correction to the issue's assumed background:
  the Rybin counterexample has **no arXiv writeup** — it exists only as an
  unrefereed X/Twitter thread. Ring loading bounds (1.1D lower / 1.3D upper)
  and the TVZ planar 2·d_max result are confirmed from primary sources.
- **Calibration against Rybin's actual instance: BLOCKED.** This session's
  WebFetch returns HTTP 402 on x.com (paywalled), no working nitter mirror was
  found, and no secondary source reproduces arc-level topology or flow values.
  **No claim in this repo reproduces or exceeds β\*=16/15.** Do not read
  anything below as a new record against the real Rybin instance.
- **What IS delivered and verified:** an exact-rational β* engine
  (`engine.py`), validated against sympy's tested LP solver (not a hand-rolled
  one — see "solver history" below) and against a fully hand-derived,
  independently-constructed calibration instance (`calibration.py`, β\*=1/2,
  matches by hand to the last digit). A parametrized generalization of that
  instance (`family.py`) was grid-searched for its own maximum β\*.
- **Symbolic K4-family, odd-hole/clique gadgets, ms13 engine sweep, ring
  loading:** not attempted this run, or only partially — see "Not done" below.
  This is an honest scope cut, not a silent one.

## Literature gate findings

Full agent report preserved in git history of this file's first commit message
context; summarized here:

1. **No arXiv paper exists** for the Rybin/GPT-5.6 counterexample as of
   2026-07-24. Source: [X thread](https://x.com/DmitryRybin1/status/2079904005652893709)
   (2026-07-22), linking a shared GPT-5.6 Pro chat transcript. Secondary
   coverage (officechai.com, vibemathed.com, digg.com) confirms only: 7-node
   DAG, demands 15/10/15, fractional cost 58, min unsplittable cost 60
   (capacity violation ≤ 15). None reproduce arc-level data. One informal
   generalization exists: [@basedjensen](https://x.com/basedjensen/status/2080020575968543230)
   posted a 3-parameter integer family (b, m, g) on the same topology with an
   explicit condition `m(b−m) > (b+m)g` — could not be fetched this session
   (same x.com 402 block) and is unverified.
2. **TVZ planar bound confirmed**: Traub–VargasKoch–Zenklusen,
   [arXiv:2308.02651](https://arxiv.org/abs/2308.02651) / *Math. Prog.* 2026
   (DOI [10.1007/s10107-026-02365-x](https://link.springer.com/article/10.1007/s10107-026-02365-x)),
   proves the +2·d_max cost-preserving version for **planar** graphs. The
   published title extends this to **bounded-genus** graphs; the exact
   bounded-genus theorem statement was blocked by a Springer paywall
   (title-level confirmation only).
3. **Ring loading bounds confirmed** from primary sources: lower bound
   **1.1·D** (Skutella, [arXiv:1405.0789](https://arxiv.org/abs/1405.0789),
   *SIAM J. Discrete Math* — this paper also disproved the original SSW ≤D
   conjecture and gives an upper bound of 19/14·D ≈ 1.357D), upper bound
   **1.3·D** (Däubel, [arXiv:1904.02119](https://arxiv.org/abs/1904.02119),
   2019). Matches the issue's stated ~1.1/~1.3 figures.
4. **No evidence of active pursuit** of the specific β* question by others —
   public discussion is at the "AI disproved a conjecture" news level, not the
   quantitative-constant level. Weak signal given the topic is ~2 days old and
   fast-moving on X.

## Engine (`engine.py`)

Implements the β* definition and breakpoint algorithm from issue #165 exactly:
`Instance` (DAG-as-arc-labels, terminals with path sets + fractional split),
`compute_beta_star()` enumerates all unsplittable routings, computes the
sorted breakpoint list `{(F_r(a) − x(a)) / d_max}`, and finds the largest
breakpoint at which the convex-hull membership LP `x ∈ conv{F_r : F_r ≤ x +
β·d_max}` is still infeasible — walking breakpoints ascending and stopping at
the first *feasible* one (β\* is the breakpoint immediately below it).

**Solver history (logged for honesty, not hidden):** the first implementation
was a hand-rolled two-phase exact-rational simplex. It failed its own
simplest sanity test (`test_lp.py::test_1d_inside`) on the first run. Rather
than debug a from-scratch simplex under time pressure — where a subtle bug
could silently corrupt every downstream certificate with no visible symptom —
it was replaced with `sympy.solvers.simplex.lpmin`, part of sympy's public,
tested API. `test_lp.py` now has 5 independent geometric sanity checks (1-D
interval, 2-D triangle inside/outside, and a symmetric-triangle stable-set-gap
case structurally resembling the Rybin mechanism) — all pass.

## Self-hosted calibration (`calibration.py`)

Since the real Rybin instance is unreachable, a small independently-constructed
SSUF instance with the same qualitative mechanism (three pairwise-conflicting
cheap "Z" paths on a triangle of shared arcs, each terminal also has a
dedicated expensive "E" path) was hand-derived end-to-end:

- 3 terminals, demand 1 each, split 1/2 between Z and E.
- Hand derivation (full algebra in `calibration.py`'s docstring): β\* = 1/2,
  first-feasible β = 1. Mechanism: at β=1/2 only 4 of 8 routings are
  β-good, and the three "shared triangle arc" equality constraints force
  the fourth (EEE) routing's weight negative (−1/2) — a clean infeasibility
  certificate, structurally the same *shape* of obstruction as the odd-triangle
  stable-set gap behind the real Rybin result (per memory of the mechanism:
  "3 pairwise-conflicting zero-cost options with fractional mass > 1"), just
  with different (verifiable) numbers.
- The engine reproduces β\*=1/2 and first-feasible-β=1 exactly. **PASS.**

This validates the pipeline is internally consistent and numerically correct.
It does **not** validate against Rybin's ground truth.

## Family sweep (`family.py`)

Generalized the calibration instance to the shape issue #165 requested:
demands (1, b, 1), per-terminal Z-split (r, q, r) (symmetric in the two
degree-1 terminals). Ran a coarse-then-refine exact-rational grid search
over (b, r, q) — first an 8×3×3 coarse pass (denominators of 4), then a
refined pass around the coarse maximum.

**Finding: this family's β\* supremum is exactly 1, approached but never
attained or exceeded.** The coarse pass found β\*=3/4 at (b,r,q)=(1, 1/4,
3/4); pushing the ridge direction it sits on (b=1, r→0⁺, q=1−r) gives a
clean closed form:

| k | r=1/2^k | β\* | first-feasible β |
|---|---|---|---|
| 2 | 1/4   | 3/4   (0.7500) | 1 |
| 3 | 1/8   | 7/8   (0.8750) | 1 |
| 4 | 1/16  | 15/16 (0.9375) | 1 |
| 5 | 1/32  | 31/32 (0.9688) | 1 |
| 6 | 1/64  | 63/64 (0.9844) | 1 |
| 7 | 1/128 | 127/128 (0.9922) | 1 |

β\*(k) = 1 − 1/2^k → 1⁻ as k→∞; first-feasible β stays pinned at exactly 1
for every k tested. Varying b away from 1 (holding r=1/32, q=31/32) only
*lowers* β\* (b=1/2 and b=3/4 both give 15/16 < the b=1 ridge's 63/64 at the
same r; b≥5/4 drops further, down to 33/64 at b=2) — b=1 is the ridge.

**So within this family, β\* never reaches the original (refuted) conjectured
bound of 1, let alone the 16/15 anchor.** This is a real, if modest, negative
result about this specific topology/parametrization — not evidence about the
Rybin family, which has a different (and currently unverifiable) topology.

**Caveat:** this is a screening/ridge-following search, not an optimization
proof — the supremum=1 finding is about the explored region of this
particular constructed family, not a theorem. A background full-grid run
(denom=16 refinement) was attempted twice and killed both times before
completing (once by hitting the 550s timeout with a print-only-at-end bug
now fixed, once by an apparent background-task interruption around the
~5-minute mark) — the ridge-following spot-checks above substitute for the
incomplete grid and are individually reproducible via `family.py`'s
`beta_star_at()`.

## Not done this run (honest scope cut)

Per issue #165's priority order:

1. ~~Symbolic optimization of the known (Rybin) K4-family~~ — could not do
   this against the *real* family (topology unverified); did a substitute
   independently-constructed family instead (above).
2. **Odd-hole/clique gadget search (C5, K4 conflict graphs)** — not started.
   The mechanism note from prior sessions (nested chain s-u-v-w with 3
   tap-offs; no-gos: 3-parallel-gates impossibility, source-prefix borrowing)
   is recorded in memory but not re-derived or tested here.
3. **ms13 engine sweep** (Dirichlet sampling + polish, referenced as "Oskar
   has core.py/sweep.py from 2026-07-22") — not portable this session; that
   code was delivered directly to Oskar in a prior session and does not
   persist in any repo or muninn_utils module this session could reach.
4. **Ring loading secondary target** — explicitly deferred per the issue's
   own instruction ("only start after primary has its calibration gate green
   ... drop if primary is productive"); primary work (engine + self-hosted
   calibration + family sweep) filled the available session, so ring loading
   was not started. Bounds are confirmed in the literature-gate section above
   for whoever picks this up next.

## Recommended next step

The blocking issue for real progress against the actual β\*=16/15 anchor is
**data access**, not method: retrieve the Rybin certificate's exact graph
(arcs, capacities, flow x, costs) either (a) from Oskar directly (he uploaded
the PDF certificate in the 2026-07-22 session), or (b) by checking whether the
`claude-workspace-fuse` spoke (referenced in the #163 Woodall closure) has it
committed from that session — this scheduled session's repo scope is locked to
`oaustegard/claude-workspace` only and could not check. Once the real instance
is available, `engine.py` and `test_lp.py` are ready to calibrate against it
directly — no engine changes anticipated.
