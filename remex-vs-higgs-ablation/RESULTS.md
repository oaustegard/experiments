# remex vs the QuIP#/HIGGS lineage — a 2×2×2 ablation for retrieval

*Commissioned by [oaustegard/experiments#8](https://github.com/oaustegard/experiments/issues/8).
First run 2026-08-02. **Rerun the same day under the `gating` skill** — corpora
rebuilt from scratch, gate audited and replaced, sweep re-executed behind it.*

## Question

Does remex's construction — exact fp32 norm stored out-of-band + dense Haar
rotation + per-coordinate scalar Lloyd-Max on the unit direction — buy anything
for retrieval-index compression over the randomized-Hadamard +
Gaussian-MSE-optimal-grid lineage (QuIP# → HIGGS → TurboQuant)?

Three axes differ, so the experiment is the full factorial rather than a
two-method bake-off. A head-to-head can only say *which* wins; the factorial
says *which axis* the difference lives on, and that is the actionable part.

| axis | remex side | HIGGS-lineage side |
|---|---|---|
| **A. rotation** | dense Haar, O(d²) apply | randomized Hadamard, O(d log d) |
| **B. norm** | exact fp32 norm out-of-band, quantize the unit direction | per-block scale folded into the payload |
| **C. codebook** | scalar Lloyd-Max, per coordinate | Gaussian-optimal m-dimensional grid |

`remex` = (haar, exactnorm, scalar). `HIGGS-like` = (rht, blockscale, vector).
The other six cells are the interaction terms.

---

# Part 1 — the rerun under `gating`

The first run shipped a calibration gate that passed all 8 of its checks and
had already caught a real defect: a random-init Lloyd trainer converging to
grids *worse* than the scalar quantizer at 6 and 8 bits, which would have been
written up as "scalar wins axis C at high rate." It was not a bad gate.

This rerun applies the [`gating`](https://github.com/oaustegard/claude-skills/tree/main/gating)
skill to it. The skill's question is not "does the suite pass" but "can it go
red," and it names three obligations: an anchor outside your own code, a
known-bad the gate demonstrably rejects, and a written statement of what it
cannot catch. The old gate had the first, had one instance of the second, and
did not have the third.

`calibrate.py` is kept **unmodified**, because `audit.py`'s probes are evidence
about that file and rewriting it would erase what they measured. `gate.py` is
the replacement.

## The audit: three checks could not fail

`python3 audit.py` → `audit.log`. Verdicts are the skill's four.

| check | verdict | the probe |
|---|---|---|
| **G6** RHT incoherence | **CANNOT FAIL** | Replaced *both* rotations with the identity — the worst possible incoherence — and G6 reported PASS at d=100, 768 and 1024. |
| **G3** grid beats scalar | **BLIND** | A vector arm contributing zero quantization gain (−0.002 to +0.002 dB) passed 5 of 6 assertions. The one that fired did so by sampling noise, not by a stated margin. |
| **G1** Max (1960) anchor | **BLIND** | The table covers b≤5; the sweep runs to 8. b=6 and b=8 carried no assertion at all. |
| **G8** byte budget | **BLIND** | A 32 MiB codebook — 4334× the real one — changed nothing G8 looks at. |
| **G7** known-bad reach | **BLIND** | The gate's only known-bad exercises G4's criterion alone. G3, the check axis C actually rests on, *accepts* the same under-trained grid (0.09142 < 0.11748 scalar). |

G0, G2 and G5 were not probed and are PLAUSIBLE: each has a named breaking
input and an external anchor, none observed red.

**G6 is the one worth dwelling on.** It is not a weak check, it is a *relative*
one, and the two things it relates are the two arms of the same factorial —
which share the apply/inverse plumbing, the call site, and the author. A shared
failure is exactly what such a check cannot see, and axis A would have been read
off two no-op rotations without a single warning.

**G3 is more uncomfortable, because the thing that broke it was a fix.** Adding
the unrefined product grid as a training candidate is correct — it is what makes
"the vector arm is never worse than the scalar arm" true rather than rhetorical,
and the first run's adversarial review was right to demand it. But it also means
a vector arm that does no vector quantization degrades to the scalar arm instead
of to something visibly bad, and `grid_mse < scalar_mse` with no margin cannot
tell those apart. A guarantee that makes an arm safe can make the check watching
it toothless.

## Mutation testing: the gate scored the codebook and never ran the codec

`mutate.py --target grids.py --target quantizers.py --stride 3 --max 50`
→ `mutate.log`: **36 killed, 55 survived of 91.**

The survivors clustered, and the cluster was the diagnosis. The gate measured
codebook *contents* — the MSE of a point set, by KD-tree — and never invoked
the *encoder*:

| survivor | what it breaks |
|---|---|
| `grids.py:166` `*`→`/`, `+`→`−` | `ScalarCodebook`'s decision boundaries |
| `grids.py:194` `k=1`→`k=2` | `VectorCodebook` returns the second-nearest codepoint |
| `quantizers.py:232` `==`→`!=` | swaps the two axis-B norm modes |

Those functions run on every vector of every corpus, and all three mutations
left the gate green. Three more were holes of the same family: `uniform_levels`
— the naive-uniform **floor control** that "the floor sits below remex
everywhere" rests on — was never called; `MAX_1960_MSE` could lose a row to a
mutated dict key and the gate would report PASS over fewer rates; and
`min(BLOCK, d//2)`→`max` collapses d=100 to a single block, turning the
per-block-scale arm into a global scale, with the byte check none the wiser
because `side == 2·nblocks` holds just as well for nblocks=1.

Two survivors were tautologies **in the rebuilt gate**: payload bytes compared
between two arms computed from the same expression (so both moved together),
and a `total` field nothing read.

`verify_kills.py` re-runs the 9 real mutants and confirms each dies — a
permanent fixture rather than a transient experiment, so the day someone widens
a tolerance for an unrelated reason it fails. It also lists 7 confirmed
equivalent mutants with the reason each is unobservable, because an unexplained
survivor and an equivalent mutant look identical in a report. Two are worth
knowing: the sign in `_trunc_second_moment` cannot be observed because
Σ(fb−fa) telescopes to zero for any level set symmetric about 0, and numpy
treats `reshape(-2, B)` exactly like `reshape(-1, B)`.

## The rebuilt gate

`gate.py`, on the skill's `Gate` harness (vendored as `_gate_harness.py`), which
mechanically refuses to report PASS without a registered known-bad it rejected
*and* at least one stated coverage limit — exit 2, INCONCLUSIVE, not 0.

Final run: **PASSED — 139 checks, 5 known-bad rejected, 14 coverage limits
stated** (`gate.log`).

What changed:

- **Absolute incoherence brackets.** Each rotation is bracketed against the
  E[max|coord|] of a uniformly random unit vector, computed by drawing and
  normalising Gaussians — a path touching neither rotation. The identity sits
  3.6–9× outside it.
- **A margin derived from measured noise.** `paired_gain` scores the trained
  grid and the lifted scalar grid on the *same* sample stream, so the common
  fluctuation cancels and the difference carries its own standard error. The
  grid must clear the degenerate baseline by 3 se, not merely land on the right
  side of an inequality. The axis-C criterion is one function, called by both
  the live check and the known-bad, so the known-bad cannot drift from it.
- **Panter–Dite at the unanchored rates.** The optimal fixed-rate scalar
  quantizer approaches 2.7207·2⁻²ᵇ from below, a hard upper bracket where
  Max (1960) has run out.
- **The codec is actually run**: idempotence (`Q(Q(x)) == Q(x)`), membership
  (every output is a codepoint), the encoder attaining the point set's own
  distortion, the arm round-trip landing on the codebook MSE, and the 1-bit
  exact-norm arm reproducing relative norms with zero spread.
- **Five known-bads instead of one**: the under-trained grid (retained), the
  identity rotation, a barely-trained zero-gain vector arm, the recorded pre-fix
  b=8 scalar MSE, and an arm that drops its codebook from shared bytes.
- **The gate blocks.** `run_ablation.py` runs it and aborts on non-zero exit,
  including exit 2. Previously it was step 3 of a documented command list, which
  is advisory. A gate that does not block is a report.

### Four checks in the new gate went red, and all four were mine

Worth recording, because it is the honest shape of adding brackets:

1. **A paired standard error compared against an unpaired one.** They estimate
   different quantities; the check was meaningless as written. Replaced with a
   comparison of the analytic paired se against the observed spread of the same
   difference across independent streams.
2. **"The vector arm always carries more shared bytes."** False at d=100 and
   2 bits, where remex's 40 KiB Haar matrix outweighs the 20 KiB codebook. The
   claim that actually holds is about the *codebook* term; the total is now a
   per-bit-width note, which is how the amortization table has to be read anyway.
3. **`rht d=1024 spike incoherence: 0.03125 < 0.03125`** — a strict lower bound
   at an *attainable* optimum. At a power-of-two d the Hadamard block spans the
   whole vector, so one round maps a coordinate spike to exactly ±1/√d. The
   bound is right; the strictness was wrong. This one blocked the sweep, which
   is the wiring working. Fixed to be inclusive, and a random-vector probe added
   because the spike probe is degenerate for the RHT at power-of-two d — FWHT
   flattens a delta by construction however badly the other stages behave.
4. **The zero-gain known-bad was not zero-gain.** To avoid a degenerate
   zero standard error I built it from one Lloyd iteration on 2,000 samples
   rather than from the unrefined product grid. At m=8 with K=65536 that
   relocates ~63,000 empty-cell codepoints toward the mode and earns a real
   +0.10 dB, so the gate correctly *accepted* it and the known-bad failed. It
   passed in fast mode (m=2, K=16) and only failed at full size — a reminder
   that a known-bad has to be validated at the configuration it will run in.

That fourth failure exposed something worth more than the fix. Probing for a
jitter scale that would give a genuinely zero-gain arm showed the paired
standard error **shrinks as the two codebooks converge**: a product grid
perturbed by N(0, 1e-3) gains +0.0001 dB against a 3-se margin of 1.2e-06, and
is accepted. The margin is statistically correct and practically empty — it
certifies *the gain is real*, not *the gain is worth having*. Real grids here
gain 0.35–1.41 dB; an arm gaining 0.01 dB would pass. The only practical floor
in this gate is the E8-ball anchor, and it constrains one configuration (m=8 at
2 bits/coord) out of nine. That is now a stated coverage limit rather than an
assumption, and it is the sharpest remaining weakness in the gate.

---

# Part 2 — the ablation

## Setup

**Scoring is against our own fp32 exact search, never human qrels** (METHODS.md
principle 4). Metrics: recall@10 and recall@100 versus the fp32 top-k, per-query
Spearman ρ, and relative reconstruction MSE as a secondary diagnostic only.

**Asymmetric setting**: documents are compressed, queries stay fp32. That is
what retrieval-index compression means in deployment, applied identically to
every arm.

| corpus | docs | queries | d | ‖x‖ CV | source |
|---|---|---|---|---|---|
| `arxiv768` | 750 | 150 | 768 | 1.4% | arXiv ML abstracts, BAAI/bge-base-en-v1.5 |
| `glove100` | 20,000 | 1,000 | 100 | 20.2% | ANN-benchmarks `glove-100-angular` |
| `nfcorpus1024` | 2,000 | 400 | 1024 | 2.7% | BEIR NFCorpus, BAAI/bge-large-en-v1.5 |

**Bit widths** 1, 2, 3, 4, 6, 8. **Seeds** 5 rotation seeds per arm (2 for the
rotation-free control). **Metrics** cosine and raw inner product. **Controls**
fp32 exact (ceiling), naive uniform scalar quantization with no rotation
(floor), and LM+QJL — the TurboQuant `prod` variant — as a replication control.

Everything is **data-oblivious**: rotations come from a seed, codebooks are
fitted to the standard normal, nothing is fitted to the corpus. remex's scalar
Lloyd-Max needs no calibration set, so giving the vector arm a corpus-fitted
codebook would confound axis C with a fit/transfer advantage — which
`recall-per-byte` and `rotation-decorrelation` have both already shown reverses
under an honest protocol.

## Reproduction against the first run

All three corpora were rebuilt from scratch. Two reproduce the first run's
numbers to four decimals; one does not, for a known reason.

| corpus | reproduces? | why |
|---|---|---|
| `glove100` | **exactly** — axis-C peak +0.0348 @ 2b (cosine) and +0.0398 @ 3b (ip), both identical to run 1 | deterministic slice of a fixed `.hdf5` |
| `nfcorpus1024` | **exactly** — +0.0178 @ 2b, +0.0239 @ 3b, identical | deterministic first-2,000 of `mteb/nfcorpus` |
| `arxiv768` | **no** — 2-bit cosine Δ moved +0.0156 → +0.0196 | abstracts are a fresh draw from the HF mirror; the first run already flagged its absolute values as non-comparable |

The codebook table reproduces throughout: b=2/m=8 gain +1.22 dB both runs,
b=8/m=2 +0.63 dB both runs, b=1/m=5 +0.35 dB both runs.

That split is the useful part. The conclusions below rest on effects that are
identical across two independently rebuilt deterministic corpora, and directionally
identical on the third.

## What the vector arm is worth

Held-out MSE per dimension of the codebook the sweep actually uses, against the
scalar quantizer at the same rate. Every row is gated (`gate.log`).

| bits | m | K | grid MSE/dim | scalar Lloyd-Max | gain | × Shannon |
|---|---|---|---|---|---|---|
| 1 | 8 | 256 | 0.323622 | 0.363380 | +0.50 dB | 1.294 |
| 1 | 5 | 32 | 0.335180 | 0.363380 | +0.35 dB | 1.341 |
| 2 | 8 | 65536 | 0.088750 | 0.117482 | +1.22 dB | 1.420 |
| 2 | 5 | 1024 | 0.094763 | 0.117482 | +0.93 dB | 1.516 |
| 3 | 5 | 32768 | 0.024997 | 0.034548 | +1.41 dB | 1.600 |
| 3 | 4 | 4096 | 0.026132 | 0.034548 | +1.21 dB | 1.672 |
| 4 | 4 | 65536 | 0.007257 | 0.009501 | +1.17 dB | 1.858 |
| 6 | 2 | 4096 | 0.000551 | 0.000644 | +0.68 dB | 2.259 |
| 8 | 2 | 65536 | 0.000036 | 0.000041 | +0.63 dB | 2.337 |

Every grid beats the scalar quantizer by more than 3 se of the paired
estimator, none beats the Shannon bound, and the ratio to the bound rises
monotonically with rate — the signature of fixed-rate quantization approaching
its Zador constant. Reference points: Max (1960) table 1 reproduced to the
printed digits at 1–5 bits; Panter–Dite bracketing 6 and 8 bits; E8's normalised
second moment (0.0716821, Conway & Sloane) reproduced to 8.5e-4 relative; a
tuned ball-shaped E8 codebook at 2 bits/coordinate that the trained m=8 grid
must beat by ≥1%; and HIGGS §4.3's own practical envelope (p ∈ [1,5],
n ∈ [9,4096]), which these grids meet or exceed at every bit width.

## Results

recall@10 against fp32 exact search, mean over 5 rotation seeds. Full tables —
recall@100, Spearman ρ, per-seed min/max, byte itemisation — in `tables.md`.

### The short answer

**Only axis C moves.** Pooled over three corpora and six bit widths:

| axis | cosine | inner product |
|---|---|---|
| **A** rotation: Haar → RHT | +0.0005 ± 0.0016 | −0.0003 ± 0.0024 |
| **B** norm: exact fp32 → per-block scale | +0.0014 ± 0.0014 | +0.0009 ± 0.0016 |
| **C** codebook: scalar → Gaussian-optimal grid | **+0.0113 ± 0.0100** | **+0.0146 ± 0.0139** |

Axes A and B are indistinguishable from zero at a seed-to-seed spread of
±0.001–0.004. Axis C is an order of magnitude larger, one-signed, and present on
all six corpus×metric combinations.

### Axis C is a low-rate effect that closes completely

| corpus / metric | peak Δ recall@10 | at | Δ at 8 bits |
|---|---|---|---|
| glove100 / inner product | +0.0398 | 3 bits | +0.0015 |
| glove100 / cosine | +0.0348 | 2 bits | +0.0004 |
| arxiv768 / inner product | +0.0282 | 2 bits | +0.0010 |
| nfcorpus1024 / inner product | +0.0239 | 3 bits | +0.0008 |
| arxiv768 / cosine | +0.0196 | 2 bits | +0.0001 |
| nfcorpus1024 / cosine | +0.0178 | 2 bits | +0.0002 |

Peak at 2–3 bits, monotone decay, gone by 8. It is also **dimension-dependent**
— roughly twice as large at d=100 as at d=768 or d=1024, which is what the
scalar-vs-vector gap should do: at higher d the rotated coordinates are closer
to i.i.d. Gaussian, exactly the regime where a scalar quantizer is least
penalised.

Head to head at the sharpest point (2 bits, cosine, matched actual bytes):

| corpus | B/vec remex | B/vec HIGGS-like | remex | HIGGS-like | Δ |
|---|---|---|---|---|---|
| glove100 | 29 | 29 | 0.598 | 0.636 | +0.038 |
| arxiv768 | 196 | 204 | 0.807 | 0.832 | +0.025 |
| nfcorpus1024 | 260 | 272 | 0.814 | 0.834 | +0.020 |

### The one place remex wins: MIPS at 1 bit

Axis C is not uniformly positive. Under **inner product at 1 bit** it goes
negative on both encoder corpora — arxiv768 −0.0063, nfcorpus1024 −0.0043 —
while staying positive on glove100 (+0.0231). Both negative values reproduce the
first run in sign; nfcorpus1024's reproduces exactly.

The mechanism is an axis-B × axis-C interaction. At 1 bit the scalar codebook
emits ±c on every coordinate, so the code's norm is constant; combined with an
exactly-stored fp32 norm, remex's reconstruction satisfies ‖x̂‖ = 0.79788·‖x‖
with **standard deviation zero** across documents. Uniform shrinkage does not
change a ranking, so remex reproduces relative document norms perfectly. The
vector arms cannot: their reconstruction-to-true norm ratio carries real
per-document noise on the quantity MIPS ranks by.

That zero-spread property is now a gate check rather than a claim in prose — it
is a property of `Arm.encode_decode`, which mutation testing showed the old gate
never called.

Whether the noise matters depends on how much the corpus's *true* norms vary.
GloVe's spread is 20.2%, so quantizer norm noise is negligible against it and
the vector codebook's better geometry wins. The BGE corpora spread 1.4–2.7% —
the same order as the noise — so the noise dominates and the constant-norm
property wins.

This is the sharpest thing the factorial bought that a head-to-head could not:
remex's advantage here is not the scalar codebook and not the exact norm, but
their *interaction*, appearing exactly where axis B looked moot on its own.

### Controls behave

`fp32` = 1.000 by construction. The naive uniform floor sits below remex
(glove100/cosine at 3 bits: 0.757 vs 0.774), confirming the rotation and the
Lloyd-Max levels are both doing work. **LM+QJL replicates**: strictly dominated
at every bit width and corpus (glove100/cosine at 2 bits: 0.373 vs remex's
0.598), reproducing the settled 2026-04-02 result. Both controls are now gated
directly, not merely observed.

### Axis A: the wall clock says the opposite of the prediction

Rotation apply, 4096 vectors at d ≤ 1024 and 512 above:

| d | Haar (dense) | RHT | ratio |
|---|---|---|---|
| 100 | 0.4 ms | 20.6 ms | Haar **50× faster** |
| 768 | 9.4 ms | 223.6 ms | Haar **24× faster** |
| 1024 | 16.3 ms | 183.9 ms | Haar **11× faster** |
| 4096 | 31.1 ms | 72.8 ms | Haar 2.3× faster |
| 8192 | 145.6 ms | 217.2 ms | Haar 1.5× faster |

The asymptotics are real and visible — the ratio moves from 50× to 1.5× as d
grows by two decades — but the crossover is nowhere near the dimensions anyone
runs retrieval at. This is a fact about numpy, not about the algorithm: the
dense rotation is one BLAS `sgemm` against decades of tuning, while the FWHT is
a Python loop over strided slices. A fused FWHT would change it entirely.
Reported because the pre-registered prediction was the other way round.

Haar's *build* cost is the opposite story and is not in the ratio above: 24.1 s
at d=8192 against the RHT's 0.8 ms. That is per-index, not per-query.

### Shared bytes invert the comparison at these corpus sizes

The headline tables exclude the rotation and the codebook, because they are
shared across the index — the convention both lineages use. That convention is
right in the limit and misleading here, and it is **not symmetric**: remex's
shared cost is one d×d rotation, while the vector arm additionally carries a
K×m codebook reaching 1 MiB.

At glove100's 20,000 documents:

| bits | arm | headline B/vec | shared B/vec | **true B/vec** | N for shared <5% |
|---|---|---|---|---|---|
| 3 | remex | 42 | 2.0 | **43.5** | 19,293 |
| 3 | HIGGS-like | 42 | 32.9 | **74.4** | 316,800 |
| 4 | remex | 54 | 2.0 | **56.0** | 14,839 |
| 4 | HIGGS-like | 60 | 52.5 | **112.5** | 350,192 |

Counted honestly the recall-per-byte ordering **reverses**:

| arm | true B/vec | recall@10 |
|---|---|---|
| remex @ 4 bits | 56.0 | 0.876 |
| remex @ 6 bits | 81.0 | **0.965** |
| HIGGS-like @ 3 bits | 74.4 | 0.804 |
| HIGGS-like @ 4 bits | 112.5 | 0.893 |

remex at 6 bits beats HIGGS-like at 4 bits on both axes at once — fewer true
bytes *and* higher recall — and remex at 4 bits beats HIGGS-like at 3 bits the
same way. The vector arm needs roughly 350,000 vectors before its codebook
amortizes under 5% of per-vector cost.

`shared_bytes()` is now asserted term by term. In the first run it was computed
and reported but gated by nothing, which the audit flagged as G8's blind spot.

## Which predictions failed

1. **"A → null. Haar ≈ RHT on recall; RHT 10–100× faster at d=768–1024."**
   *Confidence 0.8.* — **Half right, and the half that failed is the interesting
   one.** Recall is null as predicted (+0.0005 ± 0.0016 cosine, −0.0003 ± 0.0024
   ip). The speed claim is refuted with the sign reversed: RHT is 11–24×
   *slower* at those dimensions in numpy.
2. **"B → metric-dependent. Exact-norm irrelevant under cosine, helps under
   inner product."** *Confidence 0.6.* — **Failed as a main effect.** Cosine
   +0.0014, inner product +0.0009 — statistically indistinguishable, and if
   anything favouring the block-scale side. Two of three corpora come from
   encoders trained under cosine, whose raw norms barely vary (CV 1.4–2.7%
   against GloVe's 20.2%), so inner product is nearly the same problem as cosine
   there. **But see the 1-bit MIPS result**: exact-norm does win, on precisely
   those low-spread corpora, through an interaction with the 1-bit scalar
   codebook rather than on its own. The prediction failed as stated and was
   right for a reason it did not state.
3. **"C → remex loses to a properly-implemented Gaussian-optimal grid at 2–3
   bits, converging by 4–6 bits."** *Confidence 0.55.* — **Held, with
   convergence later than predicted.** remex loses at 2–3 bits on every corpus
   and metric; convergence is at 6–8 bits rather than 4–6.
4. **"remex's surviving advantage is implementation simplicity, not
   distortion."** *Confidence 0.5.* — **Held, and understated.** remex is also
   the better recall-per-byte choice below ~350k vectors once the shared
   codebook is counted, and 11–50× faster to apply in numpy.

## What this means for remex

remex is not distinctive on axes A or B: the rotation and the norm handling are
free choices that cost nothing either way. Its distinctiveness is entirely axis
C, and there it is **behind** a properly-built Gaussian-optimal grid by
0.02–0.04 recall@10 in the 2–3 bit regime, shrinking with dimension and gone by
8 bits.

That is a real loss, and a small one against what it buys: a numpy-only,
calibration-free, data-oblivious codec with a 2 KiB side table instead of a
1 MiB one, faster to apply at every dimension anyone indexes at, and winning on
true bytes-per-vector below a few hundred thousand documents. If the index is
large and the bit budget is 2–3 bits, the HIGGS lineage is the right answer.
Otherwise the gap is not what should decide it.

## Caveats

- **The 6- and 8-bit axis-C numbers are m=2 results.** `K_MAX = 2¹⁶` forces the
  sub-vector dimension to 2 at those rates. The m=2 ceiling is about 1.3 dB of
  the 4.3 dB scalar→Shannon gap. Does not affect the 1–4 bit conclusions, which
  use m=4–8.
- **Axis B rests on one corpus.** Only `glove100` has real norm spread.
- **`nfcorpus1024` is 2,000 of 3,633 documents**, capped deliberately —
  bge-large on CPU ran at 0.2–0.5 docs/s here.
- **`arxiv768` is not comparable to the first run's absolute values** — the
  abstracts are a fresh draw.
- **Seed variance was benign.** The pre-registration warned of catastrophic
  rotation-seed outliers in this family. Across 5 seeds × every cell, the worst
  seed is within 0.01–0.02 recall@10 of the mean. A negative result on that
  specific risk.
- **What the gate still cannot catch** is stated in `gate.log` under
  `[cannot catch]` — 14 entries. The load-bearing ones: every codebook check
  scores against N(0, I), so if the rotated corpus coordinates are not Gaussian
  every grid is calibrated for the wrong source and all checks still pass;
  byte accounting is checked against arithmetic, not bytes written to disk; the
  incoherence bound catches a rotation that has stopped working but not one
  that is merely mediocre; and the recall pipeline itself is outside the gate,
  anchored only by the fp32 and LM+QJL controls inside the sweep.

## Reproducing

```bash
python3 build_corpora.py     # ~1.5 h, mostly bge-large on CPU
python3 audit.py             # audits calibrate.py; writes audit.log
python3 gate.py              # the gate; exit 1 FAILED, exit 2 INCONCLUSIVE
python3 verify_kills.py      # the 9 mutants the gate's newer checks must kill
python3 run_ablation.py      # runs the gate first and ABORTS unless it passes
python3 run_ablation.py timing
python3 summarize.py > tables.md
python3 plot.py
python3 beta_check.py

# the full mutation pass (~45 min)
python3 /mnt/skills/user/gating/scripts/mutate.py \
    --target grids.py --target quantizers.py --stride 3 --max 50 \
    -- python3 gate.py --fast
```

Regenerable artifacts (`assets/`, `data/`) are gitignored. Lint gate:
`uvx ruff@0.16.0 check .` from **this directory** — passes. (The first run
claimed it also passed from the repo root; it does not. The root reports 1,089
errors, all in other experiment directories — `ms13-campaign` alone accounts for
582 — and none in this one. Corrected rather than repeated.)
