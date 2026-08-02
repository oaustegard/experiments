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
   sub-vector dimension m=2 at 6 and 8 bits. See the caveats section — this is
   a real limit on what the 6/8-bit axis-C numbers can claim.

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
| 6 bits (m=2, K=4096) | 0.0006443 | 0.0008284 | grid 29% **worse** |
| 8 bits (m=2, K=65536) | 0.0000479 | 0.0000771 | grid 61% **worse** |

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
