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

### float condition (landed 04:26 UTC; bin1 still running)

Test set, n = 1,000. Exact match is case-insensitive after whitespace
normalisation; token F1 is bag-of-words; BLEU is sacrebleu corpus BLEU; cosine
is bekko cosine between the hypothesis's re-embedding and the target.

| arm | exact | token F1 | BLEU | cosine |
|---|---|---|---|---|
| nearest training string (control) | 0.001 | 0.271 | 6.5 | 0.546 |
| zero-step, top beam | 0.008 | 0.358 | 6.7 | 0.549 |
| zero-step, verifier picks among 4 beams (round 0) | 0.014 | 0.384 | 7.5 | 0.597 |
| round 1 | 0.022 | 0.405 | 8.7 | 0.627 |
| round 2 | 0.024 | 0.408 | 8.9 | 0.633 |
| round 3 | 0.024 | 0.408 | 9.0 | 0.635 |
| round 4 | 0.024 | 0.409 | 9.0 | 0.635 |
| round 5 | 0.024 | 0.409 | 9.0 | 0.635 |

The corrector found a closer candidate for 57% of items in round 1, 22% in
round 2, 6%, 1%, then 0.2%. Exact matches by round: 14, 22, 24, 24, 24, 24.

By length of the target, final round:

| words | n | exact | token F1 | cosine |
|---|---|---|---|---|
| ≤ 6 | 267 | 0.071 | 0.523 | 0.685 |
| 7–10 | 247 | 0.020 | 0.479 | 0.673 |
| 11–16 | 204 | 0.000 | 0.355 | 0.625 |
| 17+ | 282 | 0.000 | 0.277 | 0.561 |

Final-hypothesis cosine quantiles: p10 0.43, median 0.62, p90 0.87. 16% of
items end above 0.8 and 7% above 0.9.

Not retrieval: no final hypothesis is a training string, none equals the
nearest-neighbour string, and the inverter's outputs sit 0.09 cosine closer to
the target than the closest string in the 40k training set. One test string
turned out to be a case-variant duplicate of a training string (see
`ERRORS.md`).

Training: zero-step dev loss 3.02 → 2.84 → 2.78 over three epochs (35 min
each at 3.3 s/step); corrector one epoch to dev 2.74 (61 min at 5.9 s/step).
Evaluation with five rounds: 17 min.

What the outputs look like (target / round 0 / round 5 / nearest training
string):

> why do those who do not collect social security benefits pay more for medicare
> what are the benefits of social workers who pay for medicare
> what are the benefits of social workers who pay for medicare (cos 0.78)
> do medicare and social security count on your federal income tax

> definition of home equity loan
> definition of mortgage equity
> definition of mortgage equity (cos 0.79)
> Bank of America Home Loans Bank of America Home Loans is the mortgage unit of Bank of America.

> President Friedrich Ebert knew that Germany was in an impossible situation.
> Germany was unable to understand the possibility of a germany.
> Germany was unable to understand the possibility of a germany. (cos 0.64)
> Movable type had been hitherto unknown in Europe.

Exact recoveries are all short questions: *how much do nyc detectives make*,
*what is the correct blood pressure*, *what country is nassau in*.

**Against the predictions, float only.** (1) holds in direction: verifier
selection +0.6 points exact over the top beam, round 1 +0.8 more, and rounds 2–5
add 0.2 in total. (3) holds: exact match is 7.1% at ≤ 6 words and zero past 10,
while token F1 at 17+ words is 0.28. (4) fails: 2.4 points over the control, not
20. The model is not a retrieval system (see above) but it is not an inverter in
the vec2text sense either — it is a paraphraser that lands on topic. (2) waits
on the bin1 arm.

**Why the level is where it is.** The zero-step model reaches cosine 0.55 on
its own *training* set (greedy) and 0.547 on dev — the same number, so this is
underfitting, not overfitting. Dev loss was still falling at epoch 3. vec2text's
zero-step model sits near 0.9 cosine before any correction, after 5M pairs on a
GPU; this run had 40k pairs, 125× fewer, and a base at the wrong end of the
learning curve gives the corrector little to correct from. The correction loop
itself did what the paper says it does — the largest gain in round 1, then
convergence — so the recipe transfers; the budget did not.

## Files

- `encoder.py` — bekko ONNX wrapper (torch-free), `SignBits` 1-bit code
- `build_data.py` — corpus, splits, embeddings → `data/` (gitignored)
- `model.py` — `Inverter` (zero-step / corrector) on t5-small
- `train.py`, `evaluate.py` — training and the correction loop
- `run_all.sh` — staged driver with sentinels and per-stage commits
- `results_<cond>.json` — per-round metrics and every per-item hypothesis
- `logs/train_*.json` — per-epoch losses
