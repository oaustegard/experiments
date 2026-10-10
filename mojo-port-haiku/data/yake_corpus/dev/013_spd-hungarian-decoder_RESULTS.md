# SPD's Hungarian decoder against four cheaper decoders on MSLR-WEB10K

Test of the decoder in *Single Pass Decoding for Generative Reranking*
(Laftchiev et al., Meta; arXiv:2609.01807, v1 as **hLLM**, v2 as **SPD**).
Pre-registration in [`PLAN.md`](PLAN.md), written and committed before any
model was trained.

## The finding

On 1,785 MSLR-WEB10K reranking slates of 50 items, sorting the score matrix's
**first column** ranks as well as solving the assignment problem over the whole
matrix: NDCG@10 0.5186 versus 0.5194, paired-bootstrap difference
+0.0008 [−0.0028, +0.0044]. That holds across three training seeds and across
four Sinkhorn temperatures, eight comparisons in all, every one of them with a
CI containing zero.

The Hungarian solve does beat the repair strategy the paper names as its
alternative — row-argmax with collision repair, +0.0076 NDCG@10
[+0.0039, +0.0113]. It does not beat a sort.

The reason is structural and is the pre-registered G3 diagnostic: the trained
matrix carries 80.9% of its Frobenius energy in one singular value. A rank-1
matrix `M[i,j] = a_i · b_j` with `b` decreasing in position has an optimal
assignment equal to `argsort(-a)` by the rearrangement inequality, so on such a
matrix the Hungarian algorithm is a fixed-cost identity on a sort. `M` here is
not exactly rank-1. The 19% that is not accounts for the margin over repair.
It does not produce a margin over sorting.

## The four questions and the backbone substitution

Four questions from `PLAN.md`:

- **Q1** Holding `M` fixed, does Hungarian decoding beat cheaper decoders?
- **Q2** Does the `N × K` matrix beat a scalar score plus a sort?
- **Q3** Is the trained `M` effectively rank-1?
- **Q4** Does the 0.008 ms solver figure reproduce, and how does it scale?

The backbone is replaced. Per-item vectors come from an MLP over 136
learning-to-rank features rather than from an LLM prefill, so nothing here
speaks to the paper's Section 6.1 capacity-gap claim, its LoRA ablation, or its
28 ms latency. The head, the Sinkhorn cross-entropy objective and the decoder
are the paper's.

## Setup as built

**Data.** MSLR-WEB10K (`philipphager/MSLR-WEB10k`), 136 features, graded
relevance 0–4. Slates simulate a reranking stage: top 50 candidates per query
by BM25 over the whole document (feature 110), for every query with at least 50
candidates. 5,323 train slates, 1,785 test slates. Test label mix
[41377, 30333, 14748, 1987, 805] over grades 0–4.

**Teacher.** `HistGradientBoostingRegressor`, 400 iterations, fit pointwise on
graded relevance. Its argsort is the permutation the students distill, computed
once offline (the paper's Phase 1).

**Students.** Shared per-item encoder (136 → 256 → 128, LayerNorm), then:

| arm | head | output | loss | params |
|---|---|---|---|---|
| M1 | self-attention, L=2, 4 heads | `M` in R^50×50 | Sinkhorn CE, 20 iters, τ=1 | 0.51 M |
| M2 | linear probe, no cross-item path | `M` in R^50×50 | Sinkhorn CE, 20 iters, τ=1 | 0.16 M |
| M3 | self-attention, L=2, 4 heads | one score per item | ListMLE vs the same permutation | 0.45 M |

20 epochs, AdamW, lr 1e-3 cosine, batch 64 slates, seed 20260908.

**Decoders**, all reading an identical `M`: D1 Hungarian
(`scipy.optimize.linear_sum_assignment`), D2 row-greedy with repair, D3
column-greedy, D4 sort by softmax-expected position, D5 sort by column 0.

## Diagnostics G1 to G3

`PLAN.md` requires these first, because if `M` were collision-free or exactly
rank-1 then D1 = D2 by construction and the metric comparison would measure
nothing.

| | M1 (self-attention) | M2 (linear probe) |
|---|---|---|
| G1 slates with any row-argmax collision | 100% | 100% |
| G1 mean colliding items per 50-item slate | 35.9 | 39.8 |
| G2 slates where D1 ≠ D2 as permutations | 100% | 100% |
| G3 rank-1 energy, mean | 0.809 | 0.960 |
| G3 rank-1 energy, 5th–95th pct | 0.719 – 0.899 | 0.881 – 0.996 |

The comparison is not vacuous: the assignment constraint binds on every slate
and the two decoders disagree on every slate.

The linear probe's matrix is 0.960 rank-1. That head has no path between items
at all in this setup — stricter than the paper's Figure 2a, whose probe sits on
a backbone that attended across candidates during prefill — so the reading is
narrow: with no cross-item mechanism anywhere, 96% of `M`'s energy is a single
score × position-profile product, and the solver has almost nothing to solve.
Two self-attention layers move it to 0.809.

## Decoder comparison

Means over the 1,785 test slates:

| arm : decoder | NDCG@1 | NDCG@10 | NDCG@50 | R@1 | R@10 | MRR | τ vs teacher |
|---|---|---|---|---|---|---|---|
| M1 : D1 Hungarian | 0.4739 | 0.5194 | 0.7367 | 0.0343 | 0.2787 | 0.8523 | 0.6946 |
| M1 : D2 row-greedy | 0.4724 | 0.5118 | 0.7325 | 0.0357 | 0.2742 | 0.8646 | 0.6498 |
| M1 : D3 column-greedy | 0.4739 | 0.5017 | 0.7244 | 0.0362 | 0.2802 | 0.8680 | 0.4987 |
| M1 : D4 expected position | 0.4819 | 0.5236 | 0.7396 | 0.0355 | 0.2812 | 0.8631 | 0.7002 |
| M1 : D5 column 0 | 0.4739 | 0.5186 | 0.7355 | 0.0362 | 0.2828 | 0.8684 | 0.6245 |
| M2 : D1 Hungarian | 0.4730 | 0.5136 | 0.7334 | 0.0340 | 0.2762 | 0.8521 | 0.6795 |
| M3 : scalar + sort | 0.4879 | 0.5166 | 0.7355 | 0.0364 | 0.2806 | 0.8733 | 0.7068 |
| teacher (GBT) | 0.5030 | 0.5338 | 0.7436 | 0.0359 | 0.2823 | 0.8704 | 1.0000 |
| BM25 first stage | 0.2381 | 0.3341 | 0.6325 | 0.0258 | 0.2367 | 0.7252 | 0.2101 |

Paired bootstrap on NDCG@10, 10,000 resamples over slates, M1:

| comparison | difference | 95% CI |
|---|---|---|
| D1 − D2 row-greedy | +0.0076 | [+0.0039, +0.0113] |
| D1 − D3 column-greedy | +0.0177 | [+0.0124, +0.0232] |
| D1 − D4 expected position | −0.0042 | [−0.0070, −0.0013] |
| D1 − D5 column 0 | +0.0008 | [−0.0028, +0.0044] |
| D1 − M3 scalar + sort | +0.0028 | [−0.0015, +0.0072] |
| D1 − teacher | −0.0144 | [−0.0198, −0.0089] |

Every student sits far above the BM25 order it reranks (0.33 → 0.52) and a
little below its teacher (0.534), which is what distillation into a smaller
student is supposed to look like. Arms are not tying because nothing learned.

On NDCG@1 no decoder separates from any other; every CI spans zero. The
paper's headline quality metrics are NDCG@1 and Recall@1, and at that depth
this experiment cannot distinguish the decoders at all.

## Sinkhorn temperature

τ is the lever most likely to produce a genuinely two-dimensional `M`: τ → 0
sharpens the relaxation toward a permutation matrix. Retraining M1 at four
values (NDCG@10):

| τ | rank-1 energy | collisions | D1 | D2 | D4 | D5 | D5 − D1 |
|---|---|---|---|---|---|---|---|
| 0.25 | 0.780 | 46.7 | 0.5148 | 0.4801 | 0.5132 | 0.5171 | +0.0023 [−0.0012, +0.0058] |
| 0.5 | 0.797 | 41.2 | 0.5166 | 0.4979 | 0.5193 | 0.5157 | −0.0009 [−0.0049, +0.0029] |
| 1.0 | 0.809 | 35.9 | 0.5194 | 0.5118 | 0.5236 | 0.5186 | −0.0008 [−0.0043, +0.0029] |
| 4.0 | 0.817 | 33.3 | 0.5198 | 0.5163 | 0.5211 | 0.5191 | −0.0008 [−0.0042, +0.0027] |

The near-rank-1 structure moves by 0.04 across a 16× range of τ. It is not an
artifact of leaving τ at 1.

D4's advantage over D1 should not be quoted as a result. It is +0.0042 [+0.0013, +0.0070] at τ=1 and inside the noise at every other τ,
and a sweep of D4's own decode-time softmax temperature at fixed weights
(0.05 → 10) crosses zero twice, with D1 ahead below T=0.25 and above T=4.
D5, which has no temperature and no other hyperparameter, ties D1 everywhere.

## Three seeds

| seed | rank-1 energy | D1 | D2 | D4 | D5 | M3 |
|---|---|---|---|---|---|---|
| 0 | 0.809 | 0.5194 | 0.5118 | 0.5236 | 0.5186 | 0.5166 |
| 1 | 0.786 | 0.5186 | 0.4853 | 0.5227 | 0.5188 | 0.5161 |
| 2 | 0.815 | 0.5205 | 0.4915 | 0.5244 | 0.5195 | 0.5169 |

D2 is the unstable arm — 0.485 to 0.512 across seeds, a 0.027 spread against
0.002 for D1 and D5. The Hungarian's advantage over naive repair is real and
its size depends on the draw. D5 lands within 0.001 of D1 in all three.

## The matrix versus a scalar score

M3 uses the same encoder and the same two self-attention layers, emits one
number per item, distills the same teacher permutation with ListMLE, and is
decoded by sorting. Against M1 decoded by its best cheap decoder:

| M1:D4 − M3:sort | difference | 95% CI |
|---|---|---|
| NDCG@10 | +0.0070 | [+0.0030, +0.0110] |
| NDCG@50 | +0.0041 | [+0.0017, +0.0065] |
| NDCG@1 | −0.0059 | [−0.0194, +0.0074] |
| Recall@10 | +0.0006 | [−0.0018, +0.0030] |
| Kendall τ vs teacher | −0.0066 | [−0.0088, −0.0044] |

The `N × K` formulation is worth its cost on NDCG@10 and NDCG@50, and loses on
agreement with the teacher it was distilled from. Whatever the matrix buys, it
is not delivered by the solver: the same matrix decoded by a sort of one column
gets the same NDCG@10 as the assignment solve.

## Rectangular K

Equation 1 sums over `i = 1..N` with `π ∈ Π_N`, which requires K = N; the
method statement puts `M` in `R^{N×K}`. At K < N the assignment emits K ordinals
rather than the N the abstract promises, and the paper leaves the ordering of
the remaining N − K items unstated. Here they fall back to their best score.
That tiebreak is this experiment's choice; the paper specifies none.

Truncating the trained 50-column matrix at decode time (NDCG@10):

| K | ordinals the solver emits | D1 | D4 | D5 |
|---|---|---|---|---|
| 1 | 1 | 0.5186 | 0.3341 | 0.5186 |
| 5 | 5 | 0.5198 | 0.4827 | 0.5186 |
| 10 | 10 | 0.5142 | 0.4870 | 0.5186 |
| 25 | 25 | 0.5204 | 0.5175 | 0.5186 |
| 50 | 50 | 0.5194 | 0.5236 | 0.5186 |

D1 varies by 0.006 over a 50× range of K, inside the seed spread. The position
dimension of `M` carries little at decode time — which is the same observation
as G3 from a different direction. (This is decode-time truncation of a matrix
trained at K = 50, not training at each K.)

## Solver cost

`scipy.optimize.linear_sum_assignment` is a Jonker–Volgenant
shortest-augmenting-path implementation, the same family as the paper's LAPJV.
Times are per solve, on 4 vCPU.

| N | random `M` | rank-1 `M` | banded `M` |
|---|---|---|---|
| 50 | 0.064 ms | 0.067 ms | 0.066 ms |
| 100 | 0.264 ms | 0.411 ms | 0.268 ms |
| 250 | 1.90 ms | 5.37 ms | 1.70 ms |
| 500 | 8.49 ms | 42.8 ms | 7.05 ms |
| 1000 | 40.2 ms | 311 ms | 29.5 ms |
| 2000 | 225 ms | 3072 ms | 126 ms |

At N=50 this container measures 0.064 ms against the paper's 0.008 ms. An 8×
gap between scipy on a shared 4-vCPU container and a tuned LAPJV on a server
core is unremarkable, and the paper's conclusion survives either number: at
N=50 the solver is a rounding error next to a 28 ms prefill.

Two things the table adds. The empirical exponent on random matrices from
N=50 to N=1000 is 2.16, matching LAPJV's O(N²) average case rather than the
O(N³) worst case the paper quotes. And **near-rank-1 matrices are the solver's
bad case, not its easy one** — 3072 ms at N=2000 against 225 ms for random,
13.6×, because near-ties make shortest-augmenting-path search longer. G3 says
the trained matrices are near rank-1, so a deployment that grows the slate is
on the expensive curve, not the benign one. At N ≤ 50 none of this matters.

## Controls

**Positive control on the measurement** (`controls.py`). `M[i,j] = −|i−j| + σ·ε`
has a known optimal assignment equal to the true ranking. Mean NDCG@10 over
1,000 synthetic slates:

| σ | collisions | rank-1 energy | D1 | D2 | D3 | D4 | D5 |
|---|---|---|---|---|---|---|---|
| 1 | 12.9 | 0.722 | 0.9921 | 0.9799 | 0.9423 | 0.9975 | 0.9858 |
| 4 | 17.1 | 0.698 | 0.9166 | 0.8749 | 0.8194 | 0.9431 | 0.8895 |
| 16 | 18.0 | 0.463 | 0.6868 | 0.6476 | 0.5980 | 0.7168 | 0.5982 |

D1 beats D2, D3 and D5 by 0.04 to 0.09 NDCG@10 whenever the matrix has real
two-dimensional structure. The harness detects a Hungarian advantage where one
exists, so the tie on trained matrices is a property of those matrices.

**Negative control.** On 500 draws of `M = a bᵀ` with `b` decreasing, D1
returns exactly `argsort(-a)` in 500 of 500. G3 has the interpretation
`PLAN.md` claims for it.

**Degradation sweep.** D1 − D2 on M1 at five checkpoints: +0.0268 at epoch 1,
+0.0124 at epoch 2, +0.0049 at epoch 5, +0.0103 at epoch 10, +0.0076 at epoch
20. Rank-1 energy falls from 0.884 to 0.809 over the same span. The Hungarian
helps an undertrained matrix roughly three times as much as a converged one,
and the matrix moves away from rank-1 as training proceeds.

## Untested claims

- Nothing about LLM prefill hidden states. The backbone is an MLP over LTR
  features. If real prefill states produce a matrix with much more than 19%
  off-rank-1 energy, the tie reported here would not transfer. That is the most
  likely way this result is wrong.
- Nothing about the 28 ms / 64× latency claims, which are properties of an
  A100 and a 0.6 B backbone.
- Nothing about how often an autoregressive teacher emits an invalid
  permutation. The GBT teacher here always emits a valid one, so the part of
  Section 6.2's argument that rests on teacher invalidity is untested.
- One dataset, one slate size, one teacher family, three seeds.

## Costs

Wall-clock on 4 vCPU, no GPU: data download 346 MB, slate build and teacher fit
94 s, each student 60–90 s for 20 epochs, evaluation 53 s, solver benchmark 84 s,
controls 11 s, adversarial passes 85 s. Ten model trainings in total. Under
15 minutes of compute for the whole directory.

## Container fixes

`pyarrow` is absent from the container image and pandas 3.0.5 needs it for
parquet; one `pip install --break-system-packages`. Nothing else.
