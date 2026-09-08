# Pre-registration — does the Hungarian decoder in arXiv:2609.01807 earn its place?

Written **before** any model was trained or any metric computed. Committed
before `run_*.py` produced output. Amendments, if any, are appended at the
bottom with a date and a reason; nothing above this line is edited after the
first result lands.

Date: 2026-09-08. Paper: arXiv:2609.01807, *Single Pass Decoding for
Generative Reranking* (Laftchiev et al., Meta). v1 titles the method **hLLM**;
v2 renames it **SPD**. Same method. This directory uses SPD.

## What the paper claims

A generative reranker must emit `N` ordinals. Autoregressive decoding spends
one sequential forward pass per token. SPD instead reads an `N x K`
item-position score matrix `M` off the LLM's prefill hidden states with a
2-layer self-attention head, and decodes the ordinals as

    pi* = argmax_{pi} sum_i M[i, pi(i)]                                (Eq. 1)

solved exactly by the Hungarian algorithm (LAPJV in their implementation).
Reported: 28 ms end to end, 64x over the reasoning teacher, 3.1x over the
no-reasoning teacher, quality within 0.001 AUC. Section 6.2 claims the
decoder's structural guarantees "account for part of the list-level advantage
over the autoregressive teacher".

## What is untested in the paper

Every reported ablation varies the *head* (self-attention / linear probe /
slot-query, Table 4) or the *training signal and backbone* (Table 3). No
ablation varies the **decoder**. `M` is produced, and the Hungarian algorithm
is the only thing ever applied to it. Table 5 shows the solver costs 0.008 ms;
it does not show what is lost without it.

So: given the paper's own trained score matrix, what does the combinatorial
solve buy over sorting?

## The question this directory tests

**Q1 (decoder).** Holding `M` fixed, does Hungarian decoding produce a better
ranking than cheap alternatives that ignore the assignment constraint?

**Q2 (formulation).** Does the `N x K` matrix formulation beat a scalar
relevance score plus a sort — the standard reranker — trained to distill the
same teacher permutation?

**Q3 (structure).** Is the trained `M` effectively rank-1? If
`M[i,j] ~ a_i * b_j` with `b` monotone in position, the rearrangement
inequality makes the optimal assignment *provably* equal to sorting by `a_i`,
and the Hungarian solve is a fixed-cost identity.

**Q4 (solver cost).** Reproduce the 0.008 ms @ N=50 figure and measure how it
scales.

## What this cannot test, and is not claimed to

- Whether LLM **prefill hidden states** carry enough ranking signal (their
  Section 6.1 capacity-gap claim). The backbone is replaced here.
- The LoRA-vs-frozen interaction (Table 3) as it applies to real hidden states.
- The 28 ms / 64x latency numbers, which are properties of an A100 and a 0.6B
  backbone this container does not have.
- How often an autoregressive teacher actually emits an invalid permutation.

The substitution is deliberate and narrow: Q1-Q3 are questions about `M` and
about what is done to `M`. The head, the training objective and the decoder
are the paper's. Only the source of the per-item vectors changes.

## Setup

**Data.** MSLR-WEB10K (`philipphager/MSLR-WEB10k` on HuggingFace), 136
query-document features, graded relevance 0-4, 6,000 train queries /
6,000 test queries. Real learning-to-rank data with real labels.

**Slates.** Simulate a reranking stage: for each query with at least 50
candidates, take the top 50 by feature 110 (BM25, whole document, 0-indexed
109) — a first-stage retriever's output. `N = 50`, matching the paper's
Amazon Beauty slate size and its solver-timing `N`. Fixed seed.

**Teacher.** `HistGradientBoostingRegressor` fit pointwise on graded relevance
over the train split. Teacher permutation = argsort of its scores, descending.
Computed once, offline, stored — the paper's Phase 1.

**Student.** Shared per-item encoder MLP `136 -> 128` standing in for the
prefill hidden state `h_i`. Then:

| arm | head | output | loss | decoded by |
|---|---|---|---|---|
| M1 | self-attention, L=2 (the paper's winner) | `M` in R^{50x50} | Sinkhorn CE, Eq. 3 | D1-D5 |
| M2 | linear probe (their Fig. 2a) | `M` in R^{50x50} | Sinkhorn CE, Eq. 3 | D1-D5 |
| M3 | self-attention, L=2 | scalar score per item | ListMLE vs the same teacher permutation | sort |

M3 is the control for Q2: same encoder, same cross-item attention, same
teacher, no matrix and no solver.

**Decoders**, all reading an identical `M`:

- **D1 Hungarian** — `scipy.optimize.linear_sum_assignment(-M)`. The paper's.
- **D2 row-greedy + repair** — each item takes its best free position, items
  processed by descending max score. The "repair" strategy the paper says it
  replaces.
- **D3 column-greedy** — each position takes its best free item, positions in
  order.
- **D4 expected-position sort** — sort ascending by `sum_j j * softmax_j(M[i])`.
- **D5 column-0 sort** — sort descending by `M[i,0]`, the affinity for rank 1.

**Metrics.** NDCG@{1,10,50}, Recall@{1,10} (label >= 1 counts as relevant),
MRR, and Kendall tau against the teacher permutation. Paired bootstrap over
test slates, 10,000 resamples, 95% CI on the paired difference.

## Pre-registered readings

State what the measurement reads if SPD is **entirely correct**, before
building it:

| diagnostic | if the decoder is essential | if it is decoration |
|---|---|---|
| G1 collision rate of row-argmax before repair | well above 0 | at or near 0 |
| G2 fraction of slates where D1 != D2 | large | at or near 0 |
| G3 top singular value's share of `M`'s Frobenius energy | well below 1 | at or near 1 |
| D1 - D2 NDCG@10, paired bootstrap CI | excludes 0, D1 ahead | contains 0 |
| M1 - M3 NDCG@10 | excludes 0, M1 ahead | contains 0 |

These readings are distinguishable in both directions, which is the bar item 1
of `thesis-discipline-check` sets. G1 and G3 are checked and reported **first**,
because if `M` is rank-1 or collision-free then D1 == D2 by construction and the
NDCG comparison measures nothing. In that case the finding is the structure of
`M`, not a defect in the decoder.

## Controls, scheduled here rather than added after

1. **Positive control on the measurement.** Build an `M` that is deliberately
   not rank-1 — genuine position-specific preferences plus noise — and confirm
   D1 beats D2 on it. If the harness cannot detect a Hungarian advantage where
   one is constructed to exist, a null on the trained `M` is uninterpretable.
   This runs before the trained-`M` comparison is believed.
2. **Student-quality control.** Report absolute NDCG for every arm against the
   teacher and against the BM25 first-stage order. Arms that tie because the
   student learned nothing are a different result from arms that tie because
   the decoder does nothing.
3. **Degradation sweep.** Evaluate the D1 - D2 gap at training checkpoints
   (epochs 1, 2, 5, 10, 20). If the Hungarian helps an undertrained `M` and
   stops helping a converged one, that is the characterization, not a
   refutation, and it is the answer that gets reported.
4. **Adversarial pass before writeup.** A hostile read of RESULTS.md against
   the raw numbers, scheduled now, run before the README and the PR body are
   written. Both prior instances in this repo were saved by a scheduled
   adversary and neither by re-reading.

## Thesis dating

The conclusion is not formed. What is formed, at 2026-09-08 12:30 EDT, after
reading v1 in full and before writing any code, is the **suspicion** that `M`
may be close to rank-1 and the solve therefore close to a sort. That suspicion
is what G3 is built to kill or confirm, and G3 has a reading in both
directions. If G3 comes back well below 1 and D1 wins, the suspicion was wrong
and that is what RESULTS.md will say.

## Prior art

Searched: Sinkhorn permutation learning, differentiable sorting, Hungarian
decoding in ranking. Found PiRank (Swezey et al.), Sinkhorn Policy Gradient
(Emami & Ranka), Ranking via Sinkhorn Propagation (Adams & Zemel), SoftSort,
differentiable sorting networks, DETR's matching loss, FIRST (single-token
listwise decoding), ListT5 (tournament sort). All of these either use Sinkhorn
or Hungarian for *training* or propose a different decoding structure
entirely. Did not find, in four searches, a published ablation that holds a
trained ranking score matrix fixed and varies only the decoder. Absence after
four queries is not proof; it is four queries.
