# modernbert-bidirectionality — adaptation test, plan and predictions

Written 2026-09-21 before any arm trained. Follows the ablation results in
[`RESULTS.md`](RESULTS.md): on a trained encoder, an 8-token lookahead in every
layer retained 65–79% of the bidirectional benefit and bidirectional attention
in the global layers alone retained 33–43%. Oskar: *"Run the ettin-decoder-32m
adaptation test."*

## Question

Starting from a causal decoder, which attention pattern gives the most of the
bidirectional benefit for a fixed adaptation budget: bidirectional attention in
a subset of layers (the dQwen3.5 cut), or a short lookahead window in every
layer?

## Setup

- **Start:** `jhu-clsp/ettin-decoder-32m`, the causal twin of the
  `ettin-encoder-32m` probed above: same 10-layer recipe (global attention at
  layers 0, 3, 6, 9; ±64-token sliding window elsewhere), same tokenizer, same
  2T-token data, trained with next-token prediction. 32.1M parameters.
- **Ceiling:** `ettin-encoder-32m`, masked-LM CE 2.128 / top-1 0.594 on the
  eval set below. Trained bidirectionally on the same data; no amount of
  adaptation here is expected to reach it.
- **Objective:** masked-LM at the masked position (predict token `i` from the
  hidden state at `i`, which holds `[MASK]`), the encoder's own objective and
  the one the eval set uses. 15% of positions masked, always with `[MASK]`.
- **Data:** wikitext-103 train, packed to 256-token sequences in corpus order
  (`data_train.py`, 24,000 sequences, 6.14M tokens). Every arm sees the same
  rows in the same order with the same masks.
- **Budget:** 1,465 steps × 16 × 256 = 6.0M tokens per arm, about 35 minutes
  each on 4 vCPU. AdamW, lr 1e-4 peak, 5% warmup, linear decay to 10%,
  weight decay 0.01, clip 1.0, seed 20260921.
- **Eval:** the ablation's 256 validation sequences with the identical 9,728
  masked positions, under the arm's own attention mask, at step 0 and every
  250 steps.

## Arms

| arm | full-attention layers (0,3,6,9) | sliding layers (1,2,4,5,7,8) |
|---|---|---|
| `causal` | j ≤ i | j ≤ i |
| `global` | open | j ≤ i |
| `look8` | j ≤ i+8 | j ≤ i+8 |
| `both` | open | j ≤ i+8 |
| `bidir` | open | open |

All masks are AND-ed with the model's own ±64 window on sliding layers.
`causal` is the control: it measures how much of every arm's gain is learning
the `[MASK]` task rather than reading right context. `bidir` is the LLM2Vec
recipe and the adaptation ceiling. `global` is the dQwen3.5 cut at this
model's ratio (4 of 10 layers). `look8` is the alternative the ablation
favoured. `both` tests whether the two add.

## Metric

`closed = (CE_causal − CE_arm) / (CE_causal − CE_bidir)` at the final step:
the share of the adaptation-reachable benefit an arm gets. Reported alongside
raw CE, top-1, and the gap to the encoder ceiling.

## Predictions (confidence)

- **P1 (75%)** The `causal` control ends between 3.3 and 4.3 nats. Predicting
  a masked token from its own position with only left context is next-token
  prediction with one extra hop, which the decoder already does; the task is
  learnt in the first few hundred steps.
- **P2 (65%)** `look8` beats `global` by at least 0.3 nats at the final step.
- **P3 (70%)** `bidir` is the best arm and ends between 2.3 and 3.0 nats: it
  does not reach the encoder (2.13) in 6M tokens but gets within a nat.
- **P4 (60%)** `both` ends within 0.15 nats of `bidir`: global layers plus a
  lookahead everywhere is nearly as good as bidirectionality everywhere.
- **P5 (70%)** The ordering causal > global > look8 > both ≥ bidir (by CE)
  holds at every eval from step 250 on.
- **P6 (60%)** `look8` closes ≥ 60% of the causal-to-bidir gap and `global`
  ≤ 45%, mirroring the untrained ablation (0.65 vs 0.43 on the encoder).

If P2 and P6 hold, the ablation result transfers to adaptation: a lookahead
window everywhere is the better cut. If `global` catches `look8` (P2 fails),
training repairs what the mask removes and the untrained ablation was a
misleading floor; that is the outcome that would most change the earlier
conclusion.

## Not measured here

Larger budgets (6M tokens is under 0.3% of the pretraining run); the DeltaNet
mechanics that a hybrid would need for a lookahead in its recurrent layers
(a block-local reverse scan, as in arXiv 2607.02805); downstream tasks;
decoders larger than 32M.
