# PLAN: eg2-decider — EmbeddingGemma 2 as a user-trainable decision model

Registered 2026-11-05, before any arm ran.

## Question

`browser-decider` built a 10 MB Typed Decisions model on Ettin-17m: 0.740 on the benchmark, 0.353 on a
workflow it never saw (below the 0.470 label prior), 0.279 on unrelated schemas where Jev scores 0.894.
It is a specialist. Can EmbeddingGemma 2 (EG2) do better on the part that matters for an end user, a
decision model that works on a new schema out of the box and reaches specialist accuracy from a few
dozen labelled cases, with a head small and cheap enough to train in a browser tab?

Three sub-questions:

1. How far does EG2's zero-shot similarity go as a decision model (no training on the schema)?
2. Does a small head on frozen EG2 transfer to unseen workflows, and does training it on a broad mix of
   tasks (instead of four workflows) change that?
3. How many labelled cases of a new workflow does such a head need, and what does a pruned EG2
   (8- or 4-layer cut from `eg2-layer-student`) cost?

## Prior art (searched 2026-11-05)

- `mrm8488/embeddinggemma-2-ft-typed-decision`: EG2 bi-encoder (state, question, option embedded
  separately, state flattened to `key: value` lines, Classification prompt) + ~0.5M schema-conditioned
  head whose prior term is zero-shot cosine. Frozen encoder 0.677, end to end 0.764 on Typed Decisions
  test (hyperparameters chosen on test). No unseen-workflow number.
- `Rev-gen1-EmbeddingGemma-2-Frozen` (leaderboard #64): frozen EG2, trained on a mix including
  filtered `train`, 0.6015. Weights and method private.
- `tasksource/tasksource-decider-nano`: 150M cross-encoder trained on 480k requests from dozens of
  sources (0.5% Typed Decisions), 0.7515. Its training data, `tasksource/tasksource-jev-typed-decisions`
  (1M flat state/question/options/target rows), is public and is this plan's mix.

What this adds: unseen-workflow and unrelated-schema numbers for a frozen-EG2 head, a mix-trained head
measured on workflows it never saw, a label curve, and the cost of the pruned encoders, all on the q4
ONNX graph the browser runs.

## Fixture

- Typed Decisions (`../browser-decider/data/td/`, apache-2.0): 4 workflows × 300 train / 100 test
  cases × 5 questions. Scorer = the dataset's own `score_one` (via `../browser-decider/run/common.py`):
  accuracy vs gold argmax, KL from gold, Brier, 10-bin ECE. Prior 0.470, Jev 0.727, Ettin decider 0.740.
- Unrelated schemas: `../browser-decider/data/decider_eval.jsonl` (104 items: 64 banking77 intents, 24
  guardrail predicates, 16 rubric scores), converted exactly as `browser-decider/run/eval_ood.py`
  does. Jev 0.894, Ettin decider 0.279. Reported per subset. banking77 *train* is in the tasksource
  mix, so A3 on the banking subset is not unseen; the predicate and rubric subsets are.
- Encoder: `onnx-community/embeddinggemma-2-ONNX` `model_q4.onnx`, CPU, the graph the pages run.
  States rendered as `key: value` lines, cap 1024 tokens.

## Arms

- **A0 zero-shot.** Option probability = softmax(τ_type · cos(state, option text)). The state prompt,
  option-text template and τ per question type are chosen on TD *train* only; TD test and the
  unrelated set are scored once with that choice.
- **A1 frozen head, specialist.** Head on cached embeddings: `field = norm(s + MLP([s, q, s⊙q]))`,
  `logit_i = τ_type · cos(field, o_i)`, MLP output initialised to zero so the untrained head is A0.
  Trained on all four workflows' train, soft CE vs gold, 3 seeds.
- **A2 frozen head, leave one workflow out.** Same head, trained on three workflows, scored on the
  fourth. 4 folds × 3 seeds.
- **A3 frozen head, mix-trained.** Same head trained on a sample of tasksource-jev-typed-decisions
  (Typed Decisions rows removed). Scored on TD test (every workflow unseen) and the unrelated set.
- **A4 label curve.** Start from A3 (and from A0), train on n ∈ {16, 64, 256} cases of one held-out
  workflow, score that workflow's test. 4 workflows × 3 draws.
- **A5 pruned encoder.** A1 and A3 with the 8-layer and 4-layer `eg2-layer-student` cuts as the encoder.

Rounds: A0–A2 first (CPU here). A3–A5 need more embedding than 4 vCPUs do quickly (~0.6 s per
250-token state); they go to Colab or the GPU Bridge.

## Predictions

| | prediction |
|---|---|
| P1 | A0 on TD test 0.40–0.55 (near the 0.470 prior). Unrelated set: banking 0.55–0.80, predicates 0.40–0.60, rubric ≤ 0.40 |
| P2 | A1 0.64–0.71 (mrm8488's frozen phase 0.677) |
| P3 | A2 0.40–0.52: above Ettin's 0.353 because the cosine prior survives, but within 0.05 of the prior |
| P4 | A3 on TD test 0.50–0.62 (Rev-gen1 frozen EG2 0.6015 saw filtered TD train; A3 does not) |
| P5 | A4 from A3: 64 cases reach within 0.03 of A1 on that workflow; from A0 they need 256 |
| P6 | A5: 8-layer cut costs ≤ 0.02 on A1, 4-layer 0.02–0.05; both cost more on A3 than on A1 |

## Addendum 2026-11-05: Ettin label curve (registered before it ran)

Oskar asked how the 4-layer cut compares with Ettin; nothing measured Ettin with few labels. Arm A6: stock
Ettin-17m fine-tuned as browser-decider's cross-encoder (`browser-decider/run/train_enc.py --n-cases`) on
n ∈ {64, 256} cases of one workflow, the same case draws as A4 (`default_rng(100 + draw)`, 20% of them
for validation), 3 draws × 4 workflows, scored on that workflow's test. 40 epochs at n = 64 and 15 at
n = 256 (about 600 to 1,000 optimiser steps each), best epoch by validation KL. Colab GPU.

| | prediction |
|---|---|
| P7 | Ettin at 64 cases 0.48–0.57, below EG2's head (0.584): it trains 17M parameters from an MLM start |
| P8 | Ettin at 256 cases 0.62–0.70, within 0.04 of EG2's head (0.661) |
