# Embedding inversion on bekko-a8m — CPU proof of concept

Status: **running** (pre-registered 2026-09-01, results land below as stages
finish; see `logs/` for stage sentinels).

Question: can a vec2text-shaped inverter recover text from a
bekko-embedding-v1-a8m vector on this account's stack, and how much of that
survives the 1-bit code remax_kb actually stores?

## Pre-registration

Written before any stage ran. The numbers section below is empty until they do.

**Setup.** 42,000 short English strings (half questions from NQ + MS MARCO,
half sentences split from NQ Wikipedia passages; ≤ 40 bekko tokens), split
40,000 / 1,000 / 1,000 by string. Encoder: bekko-a8m default int8 ONNX,
mean-pooled, L2-normed, float384. Inverter: t5-small with each vector projected
to 8 soft tokens (`model.py`). Two conditions:

- `float` — the model and verifier see the float embedding.
- `bin1` — both see the dequantized centered-sign code (384 bits), the
  information budget of a remax_kb 1-bit index.

Per condition: zero-step model (3 epochs), then a corrector trained for 1 epoch
on the zero-step model's own greedy hypotheses over the training set, warm
started from the zero-step weights. Evaluation: beam 4, five correction rounds,
the verifier (bekko, under the same condition) picks among candidates and the
incumbent is kept unless a candidate re-embeds closer.

**Controls.** Nearest training string by verifier cosine (what memorization
alone buys), and the zero-step top beam without verifier selection.

**Predictions, in order of confidence.**

1. Verifier selection among beams raises exact match over the top beam at
   round 0, and rounds 1–5 raise it further, with most of the gain in round 1.
2. `bin1` lands below `float` at every round on exact match, and the gap is
   larger than the gap in cosine, since the sign code is closer to the
   float vector in cosine than in recoverable content.
3. Exact match falls with length in both conditions; under `bin1` the 17+
   word bucket is near zero while token F1 stays well above zero, i.e.
   paraphrase not verbatim.
4. Both conditions beat the nearest-training-string control on exact match by
   at least 20 points; if they do not, the inverter is a retrieval system.

Not predicted, reported either way: the absolute exact-match level. vec2text
reports 92% on 32-token GTR inputs after 5M pairs and a GPU; this is 40k pairs,
t5-small, 4 vCPU, so the number here is a floor for the recipe, not a ceiling.

**Budget.** Measured 2.0 s/step at bs=64 len=16 and 6.6 s/step at len=40 on
4 vCPU; the plan is ~4–5 h wall clock for both conditions. Each stage commits
its small artifacts and pushes when it lands.

## Results

_(pending)_

## Files

- `encoder.py` — bekko ONNX wrapper (torch-free), `SignBits` 1-bit code
- `build_data.py` — corpus, splits, embeddings → `data/` (gitignored)
- `model.py` — `Inverter` (zero-step / corrector) on t5-small
- `train.py`, `evaluate.py` — training and the correction loop
- `run_all.sh` — staged driver with sentinels and per-stage commits
- `results_<cond>.json` — per-round metrics and every per-item hypothesis
- `logs/train_*.json` — per-epoch losses
