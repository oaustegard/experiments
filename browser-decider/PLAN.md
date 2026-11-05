# PLAN: a Jev-shaped decision model small enough for a browser page

Registered 2026-11-05, before any training run.

## Question

Oskar asked whether the 8-bit Monad (56M causal LM, 65 MB, already on
austegard.com/ai-tools/monad-chat/) could become a decision model in the shape of
Jev, running in the browser. He then asked about Ettin-17m, the encoder that
ettin-native shrank to 10.2 MB (int4 token table, int8 matmuls, hand-written WGSL
engine) and that came out on top of the size/quality frontier on
austegard.com/bsky/embedders/. Jev answers typed questions (`choice`, `score`,
`noul`) about a state with a probability distribution from one forward pass, with
no generated tokens. Ettin arms run first; Monad is the comparison arm.

## Fixture

`LocalLLaMA/typed-decisions` (apache-2.0): four synthetic workflows, 300 train and
100 test cases each, five typed questions per case, every option described in
`criteria`. Gold is the mean of three teacher samples. The scorer is the
dataset's own `score_one` (`data/td/score_with_jobs.py`): accuracy = argmax match,
KL from gold, Brier, 10-bin ECE over 2,000 test decisions.

Published reference points (dataset card, 2026-11-05): prior 0.470, teacher
self-agreement 0.735, Jev 1.13.0 zero-shot 0.727. Fitted on `train`: Bekko 17M
0.483, MiniLM-L6 22M 0.587, ModernBERT-base 0.646 (both per-workflow, the card's
own), SafeDecide-140M (mmBERT-small) 0.754, gemma3-270m 0.761, Ettin-400M full
fine-tune 0.8025. Zero-shot below 1B: d3-edge 0.6B 0.425, THX-01 0.3B 0.377.

Second, out-of-schema fixture: the hub's `tests/data/decider_eval.jsonl` (104
items: banking77 card intents, agent-guardrail predicates, rubric scores), where
Jev scores 0.894 and Haiku `decider` 0.846.

## Model

Cross-encoder, one sequence per question:
`[CLS] <type>: <instructions> [SEP] [MASK] <opt>: <desc> [MASK] <opt>: <desc> … [SEP] <state JSON> [SEP]`.
A small MLP scores the final hidden state at each `[MASK]`; softmax over the
question's options. Loss is cross-entropy against the gold distribution (soft
labels). Temperature per question type fitted on a 10% validation slice of
train cases. `noul` questions without criteria get `true` / `false` options with
fixed descriptions.

## Arms

1. **E-spec**: Ettin-17m fine-tuned on all four workflows' train, scored on test
   (specialist mode). Two inits: the 10 MB build's checkpoint (`e17c`, retrieval
   tuned on Bluesky posts) and stock `jhu-clsp/ettin-encoder-17m` (MLM).
2. **E-lowo**: leave one workflow out, train on three, test on the fourth's test
   (general mode on an unseen schema). Mean over the four.
3. **E-ood**: the arm-1 model on `decider_eval.jsonl`.
4. **M-spec**: Monad (PleIAs/Monad) with the same head and data, specialist mode.
5. **Browser**: the winning Ettin arm on the WGSL engine with the head in JS,
   timed on Oskar's Mac through the GPU Bridge; download size at brotli-11.

## Predictions

- P1. E-spec (better init) accuracy 0.72, range 0.68 to 0.76.
- P2. Stock MLM init beats `e17c` by at least 0.02 on E-spec.
- P3. E-lowo mean accuracy at most 0.55: training on three schemas does not buy a
  fourth, so general mode sits near the 0.470 prior and far below Jev's 0.727.
- P4. M-spec within 0.03 of E-spec.
- P5. E-ood accuracy at most 0.60, against Jev's 0.894.
- P6. Browser: five questions (one case) in at most 30 ms on the Mac's WebGPU,
  with the download within 0.5 MB of the 10.2 MB embedder build.

## Decision rule

Ship a browser demo if E-spec reaches 0.70 and the page meets P6. A general
(any-schema) in-browser decider is claimed only if E-lowo reaches 0.65.
