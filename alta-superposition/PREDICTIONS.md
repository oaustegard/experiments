# Predictions, registered before the trainer was written

The question. On LAC (`oaustegard/llm-as-computer`, branch
`exp/blind-isa-recovery`, `experiments/superposition/RESULTS-A.md`) a learned
residual code compressed 24 features into 12 dimensions purely by deleting
features nothing reads, kept every survivor exactly orthogonal (Gram
off-diagonal `0.00e+00`), and fell off a cliff at `d = 11`. Tracr §5 (Lindner
et al. 2023) compressed a feed-forward RASP program, `frac_prevs`, and found
the opposite — survivors sitting in non-orthogonal directions, genuine
superposition. One machine each, so the split could be iteration versus
feed-forward, or it could be LAC versus Tracr.

This experiment puts both shapes inside **one** compiler, ALTA (Shaw et al.
2024, arXiv 2410.18077), which compiles to Universal Transformers and ships
both looped and fixed-depth example programs.

## Three compiled programs

| program | shape | residual dims `D` | live dims |
|---|---|--:|--:|
| `subleq` | looped one-instruction computer, 16 positions, values in [-16, 16] | 402 | measured, recorded in `results.json` |
| `parity_seq` | looped running parity, one iteration per position, length 12 | 32 | measured |
| `parity_ff` | fixed-depth: selector-width count of ones, then one MLP | 8 | measured |

Definitions fixed here, before any code is trained.

* **Read** dimension — a residual dimension with a non-zero row in some query,
  key or value projection, in the first FFN weight matrix, or in the output
  transform. A dimension outside this set can be written but never influences
  the computation.
* **Used** dimension — a residual dimension that is non-zero (above `1e-12`)
  somewhere on the teacher trajectory of the training or evaluation inputs.
* **Live** dimension — read *and* used. A used-but-unread dimension still
  pollutes the compressed state when written, so deleting it is work the
  optimizer has to do; an unused dimension is pure gauge and its code row is
  never multiplied by anything, so it is excluded from every Gram matrix here.
* **Survivor** — a live dimension whose code row has norm above `1e-6`.
* **Gram off-diagonal** — the largest `|G_ij|`, `i != j`, of the Gram matrix of
  the unit-normalised survivor rows of `U`.
* **Computes exactly** — the compressed model's decoded output equals the
  symbolic interpreter's output on all 16 positions, on each of 100 held-out
  evaluation inputs.
* `d_min` — the smallest `d` at which the fraction computed exactly is 1.00.

## B1 — the looped machine compresses only by deletion

`subleq` reaches `d_min` equal to its live-dimension count, and at every
`d >= d_min` the Gram off-diagonal of the survivors is zero.

**Grading.** Confirmed if `d_min` equals the live count exactly *and*
`max |off-diagonal| < 1e-8` at every working `d`. Refuted on the number if
`d_min` is below the live count (the machine found room by sharing rather than
by deleting) or above it (the optimizer failed where a solution provably
exists — the identity restricted to live dimensions). Refuted on the mechanism
if any working `d` shows an off-diagonal above `1e-3`. `parity_seq` is a second
looped arm graded the same way; it is auxiliary, and B1 is graded on `subleq`.

## B2 — the feed-forward program superposes

`parity_ff` reaches a `d_min` strictly below its live count, and at `d_min` its
survivors are in non-orthogonal directions, reproducing Tracr §5's pattern.

**Grading.** Confirmed if `d_min < live count` *and* the Gram off-diagonal at
`d_min` exceeds 0.1. Partially confirmed if it compresses below the live count
with off-diagonals between `1e-3` and 0.1. Refuted if `d_min` equals the live
count, or if it compresses with off-diagonals below `1e-3` — deletion again,
and then the LAC result is about compiled machines in general rather than about
iteration.

## B3 — the cliff is the looped machine's, not the compiler's

`subleq` at `d_min - 1` computes **no** evaluation input exactly; `parity_ff`
at `d_min - 1` degrades gracefully.

**Grading.** Confirmed if `subleq`'s exact fraction at `d_min - 1` is 0.00 and
`parity_ff`'s is at least 0.05. Refuted if `subleq` retains any exact input at
`d_min - 1`, or if `parity_ff` also drops to 0.00.

## Method, fixed in advance

Mirrors `learned.py` from the LAC run.

1. Every compiled weight is frozen. The only trainable object is the code `U`
   (`D x d`), applied to every residual write, with readout `R` applied to
   every residual read. Tied arm `R = U` is Tracr's shared-`W` convention; an
   untied arm frees `R` at the boundary widths.
2. The objective is teacher-forced on the compiled model's own trajectory, and
   is a pair of hinges rather than a regression:
   * **margin** — every attention head's selection is a hard decision. For each
     layer, head and query position, the smallest selected logit minus the
     largest unselected logit must stay above half the compiled gap. Scale-free:
     each hinge is divided by its own target.
   * **tolerance** — every read residual coordinate on the trajectory must be
     reconstructed to within that coordinate's own decision tolerance: 0.25 for
     an indicator, a quarter of the tightest bucket gap for a numerical
     variable.
3. Codes are fit by **continuation** from the identity at `d = D`, one
   dimension at a time, projecting onto the code's own top-`(d-1)` right
   singular subspace before retraining. The LAC method lesson was that random
   initialisation does not find the optimum even where one provably exists, so
   any threshold read off a random init is a fact about the optimizer.
4. Training inputs are disjoint from the 100 evaluation inputs, and are sized
   so that every dimension the evaluation set uses is also used by the training
   set. A code cannot be asked to preserve a feature it never saw.

## Registered deviation from `learned.py`

LAC hinged tolerance on "every scalar the machine rounds". ALTA's MLP is a
compiled lookup table over one-hot codes with per-coordinate thresholds rather
than a set of named rounded scalars, so the tolerance hinge here covers every
*read* residual coordinate on the trajectory, at that coordinate's own
threshold. This is the same quantity — the decision margin the machine
actually needs — enumerated by the compiler's layout instead of by hand.
