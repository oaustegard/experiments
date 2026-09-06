# Latent calculator: a depth-3 tool port on two frozen small models

A frozen SmolLM2-135M reads an exact arithmetic result injected into its
residual stream at layer 16, with no tokens in either direction, on 0.60 of
in-distribution prompts and 0.34 of prompts with a held-out 5-digit operand,
given the true operands. Frozen Monad (56.7M, port at layer 29) reads it on
0.39 and 0.26. Both models read a categorical result (`cmp`: greater, less,
equal) at 1.00 and numeric results at a rate that falls with digit count:
SmolLM2 add 0.60, sub 0.62, mul 0.22, from 0.93 on two-digit operands to
0.39 on six-digit. The frozen models alone score 0.001. The same result
delivered as text tokens never lands in the answer format (exact 0.000) and
appears somewhere in the prose 0.39 (Monad) and 0.19 (SmolLM2) of the time,
so the latent port beats the text route on both models at zero inserted
tokens and about half the CPU milliseconds per answer.

Query formation fails on both models, and on three of four operators the
consumption failure hides it. No layer of either model carries the two operands
in a linearly decodable form at the query token: exact recovery of operator
plus twelve digit slots peaks at 2% (Monad) and 3.6% (SmolLM2), flat across
depth, while the operator alone reads at 1.00 from layer 1. The trained query
head reaches 5.7% and 12.3% exact; the calculator run on its output is right
0.27 and 0.31 of the time. Feeding that learned query to the port drops
SmolLM2 from 0.60 to 0.30 in distribution and from 0.34 to 0.22 on the
held-out length. `cmp` survives the bad query (0.94 and 0.85) because the
order of two numbers survives digit errors that the calculator does not.

The delayed arm, injected one position later at the first answer token,
works only when that token carries no information: SmolLM2 begins every
number with a bare space token, and its delayed arm scores add 0.53 and
mul 0.23 in distribution, near the synchronous 0.60 and 0.22. It scores 0.00
on `cmp`, whose answer is a single token, and Monad's delayed arm scores 0.01
on everything because Monad's first answer token already holds two digits.

Predictions are in [`PREDICTIONS.md`](PREDICTIONS.md), registered before
the first training run; grades are at the end.

## Setup

Both models frozen, fp32, CPU, 4 threads. Trained parts: a two-layer MLP
query head on the residual stream at layer `k` at the last prompt token
(operator plus two operands as six right-aligned digit slots each, 11 classes
per slot), a Python calculator on the argmaxed query, and a result encoder
(sign, 12 digit slots, kind slot, shared digit embedding, MLP) producing one
vector of size hidden. 793k trainable parameters on Monad, 1.06M on SmolLM2.
The encoder trains on the frozen model's next-token loss over the answer
tokens, gradient flowing through the frozen upper layers; layers below `k`
are cached, so Monad trains at 0.41 s and SmolLM2 at 1.1 s per batch of 32,
three epochs over 20,000 rows.

Data: twelve templates, four operators, operand lengths {1, 2, 3, 4, 6};
`test_in` is 2,000 unseen values at training lengths, `test_len5` 2,000 rows
with at least one 5-digit operand, an interpolation holdout (PREDICTIONS.md
says why the spec's 5-6 extrapolation split was changed). Exact match is
greedy generation against the answer string, stopping at the first non-empty
line. `contains` scores the answer string appearing anywhere in the
generation; it exists because the text baseline answers in prose
(`∴ Final answer: **5505**`, `5505 5505 5505`).

Injection arms: `residual` adds the vector to the residual stream after
layer `k` at the query position; `kv` gives layer `k+1`'s attention one extra
key/value slot built from the vector, visible from the query position on;
`delayed` is `residual` one position later. Baselines: `none`, the frozen
model; `text`, the calculator result inserted as tokens after the prompt.
Query modes: `oracle` (true operands to the calculator) and `learned` (the
query head's argmax). The encoder always trains on oracle results.

`k` was to be the shallowest layer whose linear probe recovers the full query
above 0.95 on validation. No layer did on either model, so the fallback took
the best: 29 of 64 for Monad, 16 of 30 for SmolLM2.

## Choosing k: the probe curves

Linear probe at the last prompt token; validation exact query recovery, mean
digit-slot accuracy, operator accuracy, and slot accuracy on `test_len5`.

Monad:

| layer | exact query | slot | operator | len-5 slot |
|---|---|---|---|---|
| emb | 0.000 | 0.46 | 0.40 | 0.31 |
| 1 | 0.002 | 0.55 | 1.00 | 0.39 |
| 13 | 0.004 | 0.58 | 1.00 | 0.41 |
| 29 | 0.020 | 0.64 | 1.00 | 0.45 |
| 45 | 0.006 | 0.59 | 1.00 | 0.41 |
| 63 | 0.011 | 0.61 | 1.00 | 0.40 |

SmolLM2:

| layer | exact query | slot | operator | len-5 slot |
|---|---|---|---|---|
| emb | 0.000 | 0.46 | 0.40 | 0.31 |
| 1 | 0.012 | 0.64 | 1.00 | 0.47 |
| 7 | 0.014 | 0.66 | 1.00 | 0.44 |
| 16 | 0.036 | 0.69 | 1.00 | 0.46 |
| 22 | 0.007 | 0.63 | 1.00 | 0.40 |
| 29 | 0.012 | 0.63 | 1.00 | 0.39 |

The operator is a single linearly readable feature from the first layer on in
both models. The operand digits are not assembled at the query token at any
depth. Units digits are the worst slots (0.41-0.50) and leading digits the
best (0.84-0.93): both models keep the size of a number at the query position
and lose its low-order digits. SmolLM2 tokenizes one digit per token and
Monad chunks digits variably (` 12`, `3`, `45`); the curves have the same
shape, so tokenization is not the cause. Monad's upper 35 layers add nothing
to this readout.

The MLP query heads: Monad exact 0.057 on validation, 0.058 on `test_in`,
0.000 on `test_len5`; SmolLM2 0.123, 0.098, 0.001. Operator 1.00 on both. The
calculator on the learned query is right 0.27 / 0.22 (Monad, `test_in` /
`test_len5`) and 0.31 / 0.22 (SmolLM2).

## Results

Exact match; `contains` on `test_in` / `test_len5`; tokens per answer (prompt
+ inserted tool tokens + generated); CPU ms per answer at batch 1.

Monad, k = 29:

| arm | query | test_in | test_len5 | contains | tokens | tool tok | ms |
|---|---|---|---|---|---|---|---|
| none | – | 0.001 | 0.000 | 0.03 / 0.01 | 19.7 | 0 | 438 |
| text | oracle | 0.000 | 0.000 | 0.39 / 0.29 | 26.7 | 4.8 | 513 |
| text | learned | 0.000 | 0.000 | 0.08 / 0.02 | 26.6 | 4.8 | 632 |
| residual | oracle | **0.387** | **0.262** | 0.39 / 0.26 | 12.9 | 0 | 155 |
| residual | learned | 0.255 | 0.222 | 0.26 / 0.22 | 12.9 | 0 | 222 |
| kv | oracle | 0.252 | 0.238 | 0.26 / 0.24 | 14.2 | 0 | 214 |
| kv | learned | 0.228 | 0.211 | 0.24 / 0.21 | 14.2 | 0 | 282 |
| delayed | oracle | 0.013 | 0.004 | 0.05 / 0.01 | 19.6 | 0 | 421 |
| delayed | learned | 0.006 | 0.001 | 0.04 / 0.01 | 19.5 | 0 | 499 |

SmolLM2, k = 16:

| arm | query | test_in | test_len5 | contains | tokens | tool tok | ms |
|---|---|---|---|---|---|---|---|
| none | – | 0.001 | 0.000 | 0.03 / 0.02 | 25.9 | 0 | 489 |
| text | oracle | 0.000 | 0.000 | 0.19 / 0.18 | 32.7 | 6.1 | 554 |
| text | learned | 0.000 | 0.000 | 0.05 / 0.02 | 32.7 | 6.1 | 622 |
| residual | oracle | **0.604** | **0.341** | 0.61 / 0.34 | 18.0 | 0 | 235 |
| residual | learned | 0.298 | 0.220 | 0.30 / 0.22 | 18.0 | 0 | 309 |
| kv | oracle | 0.337 | 0.277 | 0.34 / 0.28 | 18.0 | 0 | 238 |
| kv | learned | 0.262 | 0.221 | 0.26 / 0.22 | 18.0 | 0 | 318 |
| delayed | oracle | 0.268 | 0.107 | 0.30 / 0.13 | 18.8 | 0 | 263 |
| delayed | learned | 0.052 | 0.002 | 0.06 / 0.00 | 18.8 | 0 | 352 |

By operator and by maximum operand length on `test_in`, oracle query:

| model | arm | add | sub | mul | cmp | len 2 | len 3 | len 4 | len 6 |
|---|---|---|---|---|---|---|---|---|---|
| Monad | residual | 0.23 | 0.24 | 0.10 | 1.00 | 0.65 | 0.54 | 0.34 | 0.27 |
| Monad | kv | 0.03 | 0.04 | 0.03 | 0.93 | 0.30 | 0.28 | 0.26 | 0.22 |
| SmolLM2 | residual | 0.60 | 0.62 | 0.22 | 1.00 | 0.93 | 0.82 | 0.61 | 0.39 |
| SmolLM2 | kv | 0.14 | 0.17 | 0.07 | 1.00 | 0.60 | 0.36 | 0.31 | 0.27 |
| SmolLM2 | delayed | 0.53 | 0.31 | 0.23 | 0.00 | 0.52 | 0.42 | 0.31 | 0.09 |

Same, learned query:

| model | arm | add | sub | mul | cmp |
|---|---|---|---|---|---|
| Monad | residual | 0.05 | 0.04 | 0.04 | 0.91 |
| SmolLM2 | residual | 0.12 | 0.08 | 0.08 | 0.94 |
| SmolLM2 | kv | 0.05 | 0.04 | 0.05 | 0.94 |

On `test_len5` with oracle operands the SmolLM2 residual arm scores add
0.14, sub 0.19, mul 0.02, cmp 1.00; Monad's scores 0.00-0.02 on the numeric
operators. `cmp` rows are about a quarter of each split, so every aggregate
above 0.25 is carrying numeric answers.

Reading the failures. The residual arm's wrong numeric answers are numbers of
the right sign and length with wrong digits (`-819118` for `-821918` on
SmolLM2, `-881714` on Monad). The kv arm's wrong answers are often copies of
an operand from the prompt (`5496` for `5496 + 9`, `-828708` for
`6790 - 828708`): given a slot it cannot decode, layer k+1's attention reads
a number it can. Monad's delayed arm generates the frozen model's own prose
(`Let me try: 1653 - 2064 = -123`); SmolLM2's delayed arm answers `a` on
every `cmp` row and gets the number right when the digits come after the
blind first token.

Training signal (teacher-forced answer-token accuracy on validation after
three epochs): Monad residual 0.59, delayed 0.46, kv 0.44; SmolLM2 residual
0.83, delayed 0.75, kv 0.66. SmolLM2 residual went 0.60, 0.76, 0.83 across
epochs and was still climbing.

Latency. The latent arms are faster than `none` mostly because they generate
fewer tokens (Monad 3.7 against 10.5, SmolLM2 5.9 against 13.7): the
injected vector also steers the model into a terse answer. Per-token cost is
unchanged.

## Predictions, graded

* **D1 confirmed** on both models: frozen, 0.001 on `test_in` and 0.000 on
  `test_len5`.
* **D2 ungraded by its own rule** (no layer reached 0.95 extraction), and the
  oracle arms show it would have failed: consumption caps at 0.60 / 0.34
  (SmolLM2) and 0.39 / 0.26 (Monad), not 0.90.
* **D3 confirmed.** Residual-oracle exact match against the text arm's
  `contains` (its exact match is zero on both models): Monad 0.387 vs 0.395
  and 0.262 vs 0.293, within 5 points; SmolLM2 0.604 vs 0.185 and 0.341 vs
  0.182, ahead by 42 and 16. Zero inserted tokens, 155 vs 513 ms and 235 vs
  554 ms per answer.
* **D4 refuted as stated on `test_len5`, confirmed in distribution on
  SmolLM2.** Oracle minus learned on `test_len5`: 4 points (Monad), 12
  (SmolLM2), under the 15 the prediction named. On `test_in` for SmolLM2 the
  gap is 31 points (0.604 vs 0.298), and on `cmp`, the one operator where
  consumption is solved, it is 6-15 points on both models. Query formation
  fails (calculator right 0.22-0.31 of the time from the learned query), and
  on the held-out length the consumption failure the prediction assumed away
  hides it on the numeric operators.
* **D5 refuted on the aggregate, with the mechanism visible.** Delayed loses
  25 points on Monad and 23 on SmolLM2 (`test_len5`, oracle). Where the first
  answer token is a space (SmolLM2 numbers) the delayed arm sits within 7
  points of synchronous on add and mul in distribution; where it carries the
  answer (`cmp`, Monad's two-digit first token) the arm scores zero.

## Interpretation

One injected vector carries one categorical fact into a frozen 56M or 135M
model reliably, and a multi-digit number at a rate that falls with digit
count. The wider model (576 against 256 hidden) reads numbers better at every
length, and its encoder was still improving at three epochs, so the ceiling
on SmolLM2 is not established here. The text route is worse at this size on
both models: the frozen model surrounds the number with prose or repeats it
and never lands the bare answer.

The KV-slot arm underperforms the residual add on both models, in line with
the persistent-memory paper on frozen GPT-2 (2603.22329), where KV extension
failed and direct injection routes worked.

The async-by-one form loses nothing when the first answer token is a
format token and loses the answer when that token is the answer. A fixed
lead-in token before every answer would remove the loss.

On the query side both models keep the operator and the magnitude of the
operands at the query token and drop the low-order digits. A query head that
attends over the prompt positions, where the digits are, is the next arm; so
is injecting the result at every answer position rather than one, and
SmolLM2-135M-Instruct as a third model for a fairer `text` baseline.

## Files

`data.py`, `model_utils.py`, `probe.py`, `query_head.py`, `train_port.py`,
`eval.py`, `run_all.sh`, `test_latent_calculator.py` (21 tests), `results/`
(one JSON per probe, query head, and eval configuration), `journal.jsonl`
(per-epoch training records). `data/` and `ckpt/` are regenerable and
gitignored. Whole run: 5.4 h on 4 CPU cores.
