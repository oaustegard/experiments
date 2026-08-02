# remex vs the QuIP#/HIGGS lineage — a 2×2×2 ablation for retrieval

*Commissioned by [oaustegard/experiments#8](https://github.com/oaustegard/experiments/issues/8).
Run 2026-08-02 on CCotw.*

**Status: in progress — this file is being assembled as the sweep completes.**

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

## Setup

**Scoring is against our own fp32 exact search, never human qrels.** Human
labels conflate "is the base method good" with "did my approximation damage
it" (METHODS.md principle 4), and this repo has already watched that
saturate from both ends in `jina-remex-vs-remax`. Metrics are the ones that
experiment established: recall@10 and recall@100 versus the fp32 top-k, plus
per-query Spearman ρ over the whole corpus, plus relative reconstruction MSE
as a secondary diagnostic only.

**Asymmetric setting**: documents are compressed, queries stay fp32. That is
what retrieval-index compression means in deployment, and it is the setting
both lineages target. It is applied identically to every arm.

**Corpora** — three, differing in dimensionality and anisotropy:

| name | docs | queries | d | source |
|---|---|---|---|---|
| `arxiv768` | 750 | 150 | 768 | arXiv ML abstracts, BAAI/bge-base-en-v1.5 |
| `glove100` | 20,000 | 1,000 | 100 | ANN-benchmarks `glove-100-angular` |
| `nfcorpus1024` | 3,633 | 400 | 1024 | BEIR NFCorpus medical abstracts, BAAI/bge-large-en-v1.5 |

**Bit widths** 1, 2, 3, 4, 6, 8 per coordinate. **Seeds** 5 rotation seeds per
arm (2 for the rotation-free control), reporting min and spread, not just mean.
**Metrics** cosine and raw inner product. **Controls** fp32 exact (ceiling),
naive uniform scalar quantization with no rotation (floor), and LM+QJL — the
TurboQuant `prod` variant — as a replication control.

Everything is **data-oblivious**: rotations come from a seed, codebooks are
fitted to the standard normal, nothing is fitted to the corpus. This is not
incidental. remex's scalar Lloyd-Max needs no calibration set, so giving the
vector arm a corpus-fitted codebook would confound axis C with a fit/transfer
advantage — and both `recall-per-byte` and `rotation-decorrelation` in this
repo have already shown that advantage reverses under an honest protocol.

## Deviations from the pre-registered plan

Recorded because the issue pre-registered a specific setup and these are the
places the run departs from it.

1. **The 750-abstract arXiv set from the 2026-07-08 remax_kb codec eval is not
   in this repo**, and the spoke checkouts that held it are not present. The
   continuity corpus was rebuilt to the same shape (750 arXiv ML abstracts,
   d=768) from a different draw of abstracts. Numbers here are therefore *not*
   directly comparable to that eval's absolute values, only to each other.
2. **The live arXiv API 429s through this container's egress proxy.** Abstracts
   come from the `CShorten/ML-ArXiv-Papers` HuggingFace mirror instead; the API
   path is retained in `build_corpora.py` and is tried first.
3. **The BEIR NFCorpus zip host times out** through the same proxy; the
   `mteb/nfcorpus` HuggingFace mirror is used.
4. **The vector codebook is capped at 2¹⁶ codepoints** (`K_MAX`), which forces
   sub-vector dimension m=2 at 6 and 8 bits. This was flagged as a possible
   self-inflicted strawman on the vector arm, then checked against the source:
   HIGGS §4.3 states its own practical configuration space as grid dimension
   **p ∈ [1,5]** and grid size **n ∈ [9, 4096]**. This experiment's grids run
   to m=8 and K=2¹⁶, i.e. at or beyond the published envelope at every bit
   width, so the cap does not weaken the vector arm relative to the method it
   stands in for. It does still mean the 6- and 8-bit axis-C numbers are m=2
   results, and a higher-dimensional grid would close a little more of the
   remaining gap there — but that is past where either method is interesting,
   since both are within 0.02 recall@10 of fp32 by then.
5. **The first `arxiv768` encoding produced unit-norm vectors, which made the
   inner-product condition degenerate, and had to be redone.** BGE ships a
   `Normalize` module as the last stage of its sentence-transformers pipeline,
   and it overrides `encode(normalize_embeddings=False)` — so the supposedly
   raw vectors came out at exactly ‖x‖ = 1.0000, σ = 0. The symptom was
   unmissable once looked at: the cosine and inner-product tables for that
   corpus were *byte-identical in every cell*. With no norm to store, axis B's
   entire prediction is untestable, since the difference between "store the
   exact norm" and "fold a scale into the payload" is vacuous when the norm is
   a constant. The module is now stripped and the true norms kept.

   This turns out to matter for reading axis B at all. Even unnormalised, BGE's
   norms barely move — CV = 1.4% at d=768 — because the model is *trained*
   under cosine, so its norm carries almost no information. GloVe, by contrast,
   has CV = 20%. So the corpora do not sample "inner product" uniformly: on
   modern text encoders inner-product retrieval is nearly the same problem as
   cosine, and axis B is close to architecturally moot. Only `glove100`
   applies real pressure to it. That is a finding about the deployment target,
   not only a caveat about the setup — but it does mean the axis-B result rests
   on one corpus of three, and it is reported that way.
6. **The exact-norm arm quantizes with a Gaussian Lloyd-Max at σ = 1/√d, not
   the exact Beta marginal.** TurboQuant §3 fits its per-coordinate quantizer
   to the Beta distribution that a rotated *unit* vector actually has. The
   Gaussian is the correct asymptotic limit, but it is an approximation
   applied to remex's own side of the comparison, so `beta_check.py` measures
   it rather than assuming it is free: excess MSE is ≤0.007% at 2 bits and
   ≤0.43% at 6 bits for d=100, and ~0% for d ≥ 768. It does not materially
   handicap remex at any dimension used here.

## Calibration gate

The issue is explicit that a null on axis C is only readable if the vector arm
is credible:

> If axis C shows **no** difference, that is evidence the VQ arm is
> under-implemented, **not** evidence that scalar is optimal. Check the VQ arm
> against a published number before concluding anything.

So `calibrate.py` runs before the sweep and its verdict is a precondition on
reading any axis-C result. Per METHODS.md principle 2 it is **two-sided**: it
must certify the good implementations clean *and* reject a deliberately broken
one. A one-sided "nothing looked wrong" pass is what let `svgview` ship seven
green tests over an input path that was never connected.

*(Gate output pasted below once the final run completes.)*

### The gate caught a real defect, not a synthetic one

The first working build trained the Gaussian grids with Lloyd seeded from a
random sample of the source — the textbook LBG initialization. Gate check G3
("the trained grid must beat the scalar quantizer at the same rate") failed at
6 and 8 bits:

| rate | scalar Lloyd-Max | random-init grid (held-out) | verdict |
|---|---|---|---|
| 6 bits (m=2, K=4096) | 0.0006442 | 0.0008284 | grid 29% **worse** |
| 8 bits (m=2, K=65536) | 0.0000413 | 0.0000771 | grid 87% **worse** |

(The 8-bit scalar figure here is the corrected one — the value in use at the
time was 0.0000479, itself wrong by +16%, which is the second defect the
adversarial review turned up. It made this gap look smaller than it was.)

Left in, that would have been written up as "scalar wins axis C at high rate"
— precisely the wrong conclusion the issue warns about, and the kind that is
very hard to catch afterwards because it is directionally plausible (one
*expects* the two to converge at high rate).

Diagnosis followed METHODS.md principle 1, verifying with a deliberately
disjoint code path before blaming the subject. The empirical MSE instrument
(sampling + KD-tree nearest neighbour) was checked against the closed-form
scalar answer (exact integration against the normal density) by lifting the
scalar levels into an m-dimensional product grid, where the two must agree:

| rate | closed-form scalar | KD-tree measurement of the product grid |
|---|---|---|
| 2 bits | 0.1174819 | 0.1174547 |
| 4 bits | 0.0095011 | 0.0094960 |
| 6 bits | 0.0006443 | 0.0006473 |

The instrument was exonerated, so the grid was the only suspect left. It was
not sample starvation either — raising the training set from 48 to 1,953
samples per codepoint only moved held-out MSE from 0.000854 to 0.000769, still
worse than scalar. Nor was it a bad lattice choice: a tuned A2 hexagonal ball
codebook scored 0.000916, worse still, because a uniform-density lattice is
the wrong construction for a Gaussian at fixed rate — the optimal point
density goes as f^(m/(m+2)), and the scalar Lloyd-Max quantizer already has
that companding built in while a lattice ball does not.

The fix is to seed Lloyd from `product_init` — the scalar quantizer's own
levels lifted to m dimensions, which has exactly (2^bits)^m = K points in every
configuration this experiment uses. Because Lloyd is monotone non-increasing in
training distortion, seeding there makes the vector arm **provably no worse
than the scalar arm**, so axis C can only measure genuine vector-quantization
gain. `train_gaussian_grid` now trains both candidates and keeps whichever wins
on held-out MSE; random-init wins at low rate (where shaping gain is large and
the product grid's rectangular boundary is a real handicap) and product-init
wins at high rate. That check is now G0/G3 in the gate.

This is the entry that belongs in METHODS.md regardless of how the ablation
comes out.

## Adversarial review

The issue scheduled an independent review *before* the writeup rather than
after, on two specific questions: whether the VQ arm is implemented at
published quality, and whether the bit budget is matched honestly including
every side channel. A second agent read the harness with no stake in the
outcome and was told to try to break it.

It did. Five findings were blocking, and two of them were confirmed by direct
measurement before anything was changed:

**A stale grid, and the reason it survived.** `grid_m2_K65536.npz` — the
8-bit vector codebook — was still the artifact of the *pre-fix* trainer. Its
held-out MSE was 7.71e-5 against the scalar quantizer's 4.13e-5: the 8-bit
vector arm was **2.11 dB worse than scalar**, which would have reproduced
exactly the fake "scalar wins axis C at high rate" result the gate had already
caught once. It survived a `rm -rf` of the grid cache because a background
trainer that was being killed rewrote the file moments after the delete. The
file was identifiable only by its schema (an `mse_train` key, no `init` key).

The root cause is not the race, it is the cache key: grids were keyed on
`(m, K)` — on the *problem*, not on the *method*. A cache keyed that way cannot
notice that the code which produced its contents has changed. Grids are now
keyed and stamped with a `GRID_VERSION`, and a file whose stamp does not match
is deleted rather than trusted.

**A wrong published number, in the direction that hides the defect.**
`lloyd_max_1d` returned the distortion via the fixed-point identity
MSE = 1 − Σpᵢyᵢ², which holds only when the levels *are* the centroids of the
cells their boundaries induce. Lloyd has converged by 6 bits but not by 8
(20,000 iterations still leave max|level − centroid| ≈ 5e-6), so at 8 bits the
identity was evaluated slightly off the fixed point and returned **4.791e-5
against a true 4.127e-5, 16% high**. Max (1960)'s published table stops at 5
bits, so gate check G1 could never have caught it — and an inflated scalar MSE
makes G3's "the vector grid must beat scalar" test *more permissive* exactly
where the vector arm is weakest. The corrected value now sits just under the
Panter–Dite asymptote (2.7207·2⁻²ᵇ = 4.151e-5), as it should.

**A guarantee that was argued rather than enforced.** The claim that seeding
Lloyd from the scalar product grid makes the vector arm "provably no worse
than the scalar arm" did not hold as coded. Lloyd is monotone in *training*
distortion, but selection happens on *held-out* distortion, and the largest
grids get only ~61 samples per codepoint, where the train/held-out gap reaches
~14%. Refinement really can land worse than its own starting point, and the
unrefined product grid was never itself a candidate — so there was nothing to
fall back to. It is a candidate now, which makes the bound real instead of
rhetorical. This one stings: it is the same species of error as the bug the
gate had just caught, and I had written the justification confidently enough
not to test it.

**A confound that hit one arm only.** At d=100 and 4 bits the block size (50)
was not a multiple of the sub-vector dimension (4), so 2 of every 25
sub-vectors straddled a block boundary and had their halves scaled by
different fp16 factors before being quantized by a grid trained on N(0, I₄).
That corrupts only the `blockscale+vector` cell — the HIGGS-like arm — and so
confounds axes B and C rather than degrading anything uniformly. Block size is
now required to tile the sub-vector dimension.

**A gate that never certified the grids that mattered.** G3 hard-coded
`pick_m(b, 768)`, so the m=5 grids behind *every* `glove100` number at 1, 2 and
3 bits were never checked against scalar or against Shannon at all. The gate
now certifies every distinct grid used across all three corpus dimensions.

Two further findings changed what the experiment can claim rather than what it
computes, and both are carried into the results below: shared bytes are not
negligible at these corpus sizes (see the amortization table), and the RHT's
asymptotic advantage does not survive contact with numpy (see axis A).

The full verdict — `ACCEPTABLE-WITH-CAVEATS` on the VQ arm,
`HONEST-WITH-CAVEATS` on the per-vector budget and `NOT-MATCHED` once shared
bytes are counted — is reproduced in the PR body.

### What the vector arm is worth, after all of that

The point of the gate and the review is that axis C is only readable if these
numbers are real. They are the held-out MSE per dimension of the codebook the
sweep actually uses, against the scalar quantizer at the same rate:

| bits | m | K | grid MSE/dim | scalar Lloyd-Max | gain | × Shannon bound |
|---|---|---|---|---|---|---|
| 1 | 8 | 256 | 0.323349 | 0.363380 | +0.51 dB | 1.293 |
| 1 | 5 | 32 | 0.334978 | 0.363380 | +0.35 dB | 1.340 |
| 2 | 8 | 65536 | 0.088641 | 0.117482 | +1.22 dB | 1.418 |
| 2 | 5 | 1024 | 0.094744 | 0.117482 | +0.93 dB | 1.516 |
| 3 | 5 | 32768 | 0.024992 | 0.034548 | +1.41 dB | 1.600 |
| 3 | 4 | 4096 | 0.026124 | 0.034548 | +1.21 dB | 1.672 |
| 4 | 4 | 65536 | 0.007254 | 0.009501 | +1.17 dB | 1.857 |
| 6 | 2 | 4096 | 0.000551 | 0.000644 | +0.68 dB | 2.255 |
| 8 | 2 | 65536 | 0.000036 | 0.000041 | +0.63 dB | 2.340 |

Every grid beats the scalar quantizer at its own rate, none beats the Shannon
bound, and the ratio to the bound rises monotonically with rate — the
signature of fixed-rate quantization approaching its Zador constant, and a
sanity check that the numbers are not accidents. The gain peaks at 3 bits and
falls off at 6 and 8 bits, but that fall-off is **m=2's ceiling, not
vector quantization's**: the sub-vector dimension drops to 2 there because
2^(bits·m) must stay under 2¹⁶. See the caveats.

Two of these rows only look right because of the fixes above. The 8-bit row
was 0.0000771 before the stale codebook was caught — 87% *worse* than scalar
rather than 0.63 dB better.

The reference points that make them meaningful:

- **Max (1960)** table 1, reproduced to the printed digits at 1–5 bits by the
  scalar arm (gate G1).
- **E8's normalised second moment**, 0.0716821 (Conway & Sloane), reproduced to
  3.7e-4 relative by the lattice machinery (gate G2).
- **A tuned ball-shaped E8 codebook** at 2 bits/coordinate — the shaping
  QuIP#'s E8P codebook uses — scoring 0.09110 MSE/dim, which the trained m=8
  grid has to beat (gate G4).
- **HIGGS §4.3's own practical envelope**, grid dimension p ∈ [1,5] and grid
  size n ∈ [9, 4096]. This experiment's grids run to m=8 and K=2¹⁶, at or
  beyond that envelope at every bit width, so the vector arm is not a
  weakened stand-in for the method it represents.
