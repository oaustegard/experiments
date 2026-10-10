# Embedding inversion on bekko-a8m — CPU proof of concept

Status: **done** (pre-registered 2026-09-01 20:33 EDT, both arms landed by
2026-09-02 03:58 EDT; 7.4 h wall clock on 4 vCPU).

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
t5-small, 4 vCPU, so the number here is a floor for the recipe.

**Budget.** Measured 2.0 s/step at bs=64 len=16 and 6.6 s/step at len=40 on
4 vCPU; the plan is ~4–5 h wall clock for both conditions. Each stage commits
its small artifacts and pushes when it lands.

## Results

### float condition (landed 04:26 UTC)

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
the vec2text sense either: a paraphraser that lands on topic. (2) is
scored below.

**Why the level is where it is.** The zero-step model reaches cosine 0.55 on
its own *training* set (greedy) and 0.547 on dev — the same number, so this is
underfitting, not overfitting. Dev loss was still falling at epoch 3. vec2text's
zero-step model sits near 0.9 cosine before any correction, after 5M pairs on a
GPU; this run had 40k pairs, 125× fewer, and a base at the wrong end of the
learning curve gives the corrector little to correct from. The correction loop
itself did what the paper says it does: the largest gain in round 1, then
convergence. The shortfall is in the base model's training budget.

### bin1 condition (landed 07:57 UTC)

Same protocol, but the model and the verifier see only the dequantized
384-bit centered-sign code of every vector. "cosine" in this table is
therefore cosine between two ±1 sign vectors, which equals 1 − 2·(fraction of
disagreeing bits); 0.385 means 69% of bits agree. It is not on the float
table's scale. The float-space comparison below puts both arms on one ruler.

| arm | exact | token F1 | BLEU | cosine |
|---|---|---|---|---|
| nearest training string (control) | 0.001 | 0.257 | 6.1 | 0.369 |
| zero-step, top beam | 0.003 | 0.301 | 5.0 | 0.297 |
| zero-step, verifier picks among 4 beams (round 0) | 0.005 | 0.318 | 5.3 | 0.347 |
| round 1 | 0.007 | 0.334 | 6.0 | 0.379 |
| round 2 | 0.009 | 0.336 | 6.2 | 0.384 |
| round 3 | 0.009 | 0.336 | 6.1 | 0.385 |
| round 4 | 0.009 | 0.336 | 6.1 | 0.385 |
| round 5 | 0.009 | 0.336 | 6.1 | 0.385 |

Improved fraction per round: 53%, 16%, 3%, 0.5%, 0.1%. Exact matches by round:
5, 7, 9, 9, 9, 9. Final code-space cosine quantiles: p10 0.213, median 0.365,
p90 0.568.

| words | n | exact | token F1 | cosine |
|---|---|---|---|---|
| ≤ 6 | 267 | 0.019 | 0.409 | 0.417 |
| 7–10 | 247 | 0.016 | 0.411 | 0.423 |
| 11–16 | 204 | 0.000 | 0.279 | 0.374 |
| 17+ | 282 | 0.000 | 0.244 | 0.328 |

No final hypothesis is a training string or the nearest-neighbour string.

**Apples to apples.** Re-embedding both arms' final strings with bekko and
scoring against the float target (`results_bin1.json → float_space`):

| strings from | float-space cosine to target |
|---|---|
| float arm, round 5 | 0.635 |
| float arm, nearest training string | 0.546 |
| bin1 arm, round 5 | 0.529 |
| bin1 arm, round 0 | 0.495 |
| bin1 arm, nearest training string | 0.515 |

And in the other direction, scoring the float arm's strings with the sign
code: they agree with the target on 71.8% of bits, against 69.2% for the arm
that was trained on the code. The float-trained inverter beats the
code-trained one on the code's own metric.

Training: zero-step dev loss 3.27 → 3.11 → 3.05 (float: 3.02 → 2.84 → 2.78,
a steady 0.27 nats/token behind); corrector to 3.01 (same 0.04 gain as float).

Exact recoveries: *what is aneurysm*, *how many active satellites are there*,
*most famous gemstones*, *It reached the top 10 in Canada.* — the last is the
only exact sentence recovered in either arm.

> was texas chainsaw massacre filmed in texas
> where did the chinese shootings take place
> where did the shootings in texas take place (code 0.35, float 0.54)
> The Texas Chain Saw Massacre The Texas Chain Saw Massacre premiered in Austin, Texas on October 1, 1974, …

> The production then moved to Utah's Arches National Park to shoot more of the opening.
> The shooting took place at the National Parks in Washington, D.C., and in Washington, D.C..
> The shooting took place at the National Parks. (code 0.33, float 0.55)
> Filming also took place that month near Park City, Utah.

## Prediction results

1. **Holds, both arms.** Verifier selection among beams raises exact match
   over the top beam (0.8 → 1.4 float, 0.3 → 0.5 bin1); round 1 adds the most
   (+0.8 and +0.2 points; 57% and 53% of items improved); rounds 2–5 add 0.2
   points in each arm and converge by round 3.
2. **Holds.** bin1 lands below float at every round on exact match (0.9 vs 2.4
   at round 5, a 63% relative drop) and the gap in float-space cosine is
   smaller (0.529 vs 0.635, 17% relative). The sign code keeps most of the
   direction and loses most of the recoverable content.
3. **Holds.** Exact match is zero past 10 words in both arms; under bin1 the
   17+ bucket is 0.000 exact with token F1 0.244.
4. **Fails, both arms.** 2.4 and 0.9 points over the control, not 20. The
   models are not retrieval (no output is a training string; float-arm
   outputs sit 0.09 closer than any training string) but they are not
   inverters at this budget either. bin1 round 0 is *below* the retrieval
   control on its own verifier (0.347 vs 0.369) and only the corrector lifts it
   past.

## Recipe transfer at this budget

The mechanism transfers: verifier-in-the-loop selection and iterative
correction behave exactly as vec2text describes, with the same shape of gains.
The level does not, and the reason is on the training curve, not in the loop:
the zero-step base scores the same cosine on its own training set as on dev,
dev loss was still falling at the last epoch, and vec2text's base starts near
0.9 cosine where this one starts at 0.55. A corrector can only refine what the
base already nearly has. On this account's stack the relevant number is the
1-bit arm: from a remax_kb index a 40k-pair inverter recovers the topic of a
short question about a third of the time by token overlap and the exact
string under 1% of the time, and it does no better than the float arm even at
matching the code bits. Scaling the base (more pairs, more epochs, a GPU) is
the next step if exact recovery is the goal; the code is ready for it
(`train.py --epochs`, `build_data.py --n-train`).

## Files

- `encoder.py` — bekko ONNX wrapper (torch-free), `SignBits` 1-bit code
- `build_data.py` — corpus, splits, embeddings → `data/` (gitignored)
- `model.py` — `Inverter` (zero-step / corrector) on t5-small
- `train.py`, `evaluate.py` — training and the correction loop
- `run_all.sh` — staged driver with sentinels and per-stage commits
- `results_<cond>.json` — per-round metrics and every per-item hypothesis
- `logs/train_*.json` — per-epoch losses
