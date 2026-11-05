# RESULTS: EmbeddingGemma 2 as a decision model people train themselves

A decision model answers typed questions about a piece of state (a support thread, an invoice, an
agent's trace) with a probability for each option: pick a label (`choice`), rate on a rubric (`score`),
or say whether a statement is true (`noul`). TypeSafe's Jev is the hosted example. `browser-decider`
built a 10 MB one on Ettin-17m that matches Jev on the Typed Decisions benchmark (0.740 against 0.727)
but falls to 0.353 on a workflow it never saw and to 0.279 on a separate set of 104 unrelated questions
(bank intents, guardrail yes/no predicates, rubric scores).

The question here: does EmbeddingGemma 2 (EG2), kept frozen with a small trained "head" on top, work
on new questions without training, and how many labelled examples does a new workflow need? The
encoder is the 4-bit ONNX build the austegard.com pages already run. Accuracy throughout means the
benchmark's own metric: the top option agrees with the top option of the gold distribution (the mean of
three teacher samples). The teacher agrees with itself 0.735 of the time, and always picking each
question's most common answer scores 0.470.

## Answer

Out of the box, EG2 classifies topic- and intent-style questions well and is not a decision model. With no training it scores 0.721 on 104 unrelated questions, where the trained Ettin decider scores 0.279. <!-- @evidence[[results/summary.json#full.a0.ood_accuracy]] -->
Most of that comes from bank-intent questions (0.875). <!-- @evidence[[results/summary.json#full.a0.ood_by_subset.banking]] -->
On Typed Decisions, whose states are records of numbers and events, it scores 0.378, below the 0.470 you get by always picking each question's most common answer. <!-- @evidence[[results/summary.json#full.a0.td_test_accuracy]] -->

It needs labelled cases of each workflow. A one-million-parameter head trained on a workflow's own cases reaches 0.584 from 64 cases and 0.661 from 256. <!-- @evidence[[results/summary.json#full.a4.a0_n256]] -->
Training on all four workflows gives 0.678, against Ettin's 0.740. <!-- @evidence[[results/summary.json#full.a1.td_test_accuracy]] -->
Training a 256-case head took 18 s on one CPU thread in PyTorch; a browser trainer was not built. <!-- @evidence[[results/summary.json#full.timing.n256_seconds_1thread]] -->

Training the head on a broad mix of 727 other task sources did not make it general.
It scored 0.327 on Typed Decisions, worse than no training. <!-- @evidence[[results/summary.json#full.a3.td_test_accuracy]] -->
The untested next lever is training the encoder itself on a broad mix, as tasksource-decider-nano
(150M) does: 0.7515 on the Typed Decisions leaderboard.

## Findings

1. **The zero-shot score depends on the kind of question.** Zero-shot EG2 gets 0.875 on banking intents, 0.458 on yes/no guardrail
   predicates (a 50/50 set, so chance), 0.500 on 16 rubric scores, and 0.378 on Typed Decisions. Similarity between
   a state and an option's description works when the answer is "which topic is this"; it does not read
   numbers, rules or truth. (`results/a0.json`)
2. **A head on frozen EG2 matches the published frozen-EG2 result.** 0.678 ± 0.010 over three seeds,
   KL 0.182, Brier 0.105, ECE 0.109; mrm8488's frozen phase scored 0.677. On the unrelated set it
   scores 0.667 (zero-shot 0.721), where Ettin's trained decider fell to 0.279; the part it keeps is
   banking (0.766), while predicates (0.528) and rubric (0.479) stay near chance. Starting the head at
   the zero-shot score is the likely reason it keeps more than Ettin; that was not tested
   separately. (`results/a1.json`)
3. **Nothing transfers to an unseen workflow.** Trained on three workflows, tested on the fourth: 0.372,
   the same as no training (0.378) and as Ettin (0.353). Typed Decisions has 20 distinct questions; a
   head trained on 15 of them has nothing general to learn. (`results/a2.json`)
4. **A broad mix teaches the head the mix, not the skill.** On 7,921 rows from 727 tasksource sources,
   the head improved on held-out mix rows (0.424 to 0.500, seed 0) and got worse everywhere else: 0.327
   ± 0.008 on Typed Decisions over three seeds, and 0.609 on the unrelated set (zero-shot 0.721), where
   seeds ranged from 0.452 to 0.712. (`results/a3.json`, `results/a3_diag.json`)
5. **Labels per workflow are what buy accuracy.** Head trained only on n cases of one workflow: 0.503 at
   16, 0.584 at 64, 0.661 at 256; the all-workflow head scores 0.678. Starting from the mix-trained head
   instead of zero-shot is no better (0.478, 0.575, 0.655). (`results/a4.json`)
6. **The 4-layer cut keeps specialist accuracy and loses generality.** With the 25 MB cut from
   `eg2-layer-student` as the encoder, the trained head scores 0.665 (−0.013), but the unrelated set falls
   from 0.667 to 0.442, and zero-shot from 0.721 to 0.587. It embeds states 5 times faster on CPU
   (86 s against 442 s for 1,672 states). (`results/cut4/`)
7. **The head is cheap enough to train in a browser.** 984,324 parameters; 6.2 s for 64 cases and 18.1 s
   for 256 on one CPU thread in PyTorch, on cached embeddings. No gradient touches the encoder, so a
   page needs the forward pass it already has plus a small trainer. (`results/timing.json`)

## Method

**Fixture.** Typed Decisions (`../browser-decider/data/td/`, apache-2.0): four synthetic workflows, 300
train and 100 test cases each, five typed questions per case, gold = mean of three teacher samples.
Scored with the dataset's own `score_one` via `../browser-decider/run/common.py`. The unrelated set is
`../browser-decider/data/decider_eval.jsonl` (64 banking77 intents, 24 guardrail predicates, 16 rubric
scores), converted as `browser-decider/run/eval_ood.py` does.

**Encoder.** `onnx-community/embeddinggemma-2-ONNX` `model_q4.onnx` on 4 vCPU (`run/enc.py`), states
rendered as `key: value` lines and capped at 1,024 tokens, cached once per (prompt, text). The cut is the
4-layer, 32K-vocabulary, 3-bit-table build served at austegard.com/bsky/image-search/model/ort/
(distilled onto EG2 query vectors, then fine-tuned on Bluesky alt text). The 8-layer cut exists only as
a PyTorch checkpoint and was not exported.

**Zero-shot (A0).** P(option) = softmax(τ_type · cos(state, option text)). State prompt (2), option
prompt (3 choices) and option template (4 choices) were chosen by Typed Decisions *train* accuracy, τ per type by train
KL; test and the unrelated set were scored once. Choice: `task: classification` on both sides, option
text "instructions Answer: name. description".

**Head (A1–A4).** `field = norm(s + MLP([s, q, s⊙q]))`, `logit_i = τ_type · cos(field, o_i) + r([s⊙o_i,
q⊙o_i])`, both MLPs ending in zero-initialised layers so the untrained head is A0 (checked: 0.378). Soft
cross-entropy against gold, AdamW lr 1e-3, 40 epochs (30 for the mix), best epoch by validation KL on 10%
of training cases (20% in A4), per-type temperature fitted on that slice. A1: all four workflows, 3
seeds. A2: leave one workflow out, 4 folds × 3 seeds. A3: 7,921 rows sampled from shard 0 of
`tasksource/tasksource-jev-typed-decisions` (≤ 12 per source, no Typed Decisions rows), states capped at
512 tokens, 3 seeds. A4: n ∈ {16, 64, 256} cases of one workflow × 3 draws, from A0 or from A3 seed 0.
A5: A0 and A1 with the cut; A3 was not rerun on the cut because it was negative on full EG2.

**Caveats.** The unrelated set is small (104 questions, 16 of them rubric), so its per-subset numbers move
by several points between seeds. The 4-bit build agrees with full-precision EG2 at cosine about 0.99, so
these numbers are for the build the pages ship, not the published weights.

**Compute.** All on the container's 4 vCPU: state embedding ~7.5 min per prompt (full) and ~1.5 min
(cut), mix embedding ~60 min, every head run under a minute. No GPU.

## Log

### 2026-11-05: A0 to A5

Registered `PLAN.md` before any run (commit 5798526). Predictions against results:

| prediction | result |
|---|---|
| P1 A0 Typed Decisions 0.40–0.55; banking 0.55–0.80, predicates 0.40–0.60, rubric ≤ 0.40 | 0.378, 0.875, 0.458, 0.500: missed on three of four, both directions |
| P2 A1 0.64–0.71 | 0.678: held |
| P3 A2 0.40–0.52 | 0.372: missed, no better than zero-shot |
| P4 A3 0.50–0.62 | 0.327: missed, worse than zero-shot |
| P5 from A3, 64 cases within 0.03 of A1; from A0 needs 256 | A3 start 0.575 (−0.10): missed; A0 start at 256 0.661 (−0.017): held |
| P6 4-layer cut costs 0.02–0.05 on A1, more on A3 | −0.013 on A1, and −0.225 on the unrelated set; A3 not run on the cut |

The A0 sweep over 24 prompt/template pairs spread only from 0.289 to 0.376 on train, so the choice of
prompt matters little next to what similarity can and cannot answer. A3's best epoch was 3 to 5 for
every seed: validation KL on mix rows bottomed out early while the head was still moving away from the
zero-shot geometry the other test sets need. A check confirmed the pipeline was not at fault: the
untrained head routed through the mix code reproduces A0's 0.378 exactly.
