# gemma-proxy-tuning

Can a fine-tune of a small Gemma 3 be transplanted onto Gemma 4 31B at decode
time, without touching the 31B's weights?

**Status: preconditions measured, experiment not run.** The vocabulary
precondition holds. The logit-scale precondition does not, and the fix is
known. Cross-generation delta transfer is untested by anyone, which is the
actual open question.

Reproduce every number below with `python3 tokenizer_diff.py`; its output is
[`tokenizer_diff.json`](tokenizer_diff.json).

## Proxy-tuning needs shared indexing and comparable scale

Proxy-tuning ([Liu et al., COLM 2024](https://arxiv.org/abs/2401.08565)) steers a
large model you cannot fine-tune by adding a logit-space delta harvested from a
small model you can:

```
p(x_t) = softmax[ s_base + alpha * (s_expert - s_antiexpert) ]
```

`s_expert` is a small model after tuning, `s_antiexpert` is that same checkpoint
before tuning. Emulated fine-tuning ([Mitchell et al.](https://arxiv.org/abs/2310.12962))
derives the identical expression from KL-constrained RL, framing it as decoupling
pre-training scale from fine-tuning scale; its up-scaling case *is* proxy-tuning.
Published results close 88–91% of the gap to true fine-tuning, but every one of
them moves a delta **across scale within a single pretraining run**, Llama-2-7B-chat
steering Llama-2-70B. Gemma 3 to Gemma 4 crosses a generation.

Summing three logit vectors requires two things of them: shared indexing, and
comparable scale.

## Vocabulary

`google/gemma-3-*` is gated behind an accepted license (401 on `resolve`), so the
Gemma 3 tokenizer comes from the `unsloth` mirrors, which republish it verbatim.
The Gemma 4 weights are apache-2.0 and ungated.

| | Gemma 3 | Gemma 4 |
|---|---|---|
| model type / vocab size | BPE, 262,144 | BPE, 262,144 |
| merge rules | 514,906 | 514,906 — byte-identical |
| ids carrying the same token | — | 255,938 (97.63%) |
| tokens present in both as strings | — | 262,125 (99.993%) |
| **ordinary text tokens that changed id** | — | **0** |

All 6,187 tokens that moved are special or reserved. The ids where the two
vocabularies disagree form exactly two contiguous ranges:

| ids | count | Gemma 3 | Gemma 4 |
|---|---|---|---|
| `[46, 106]` | 61 | `<unused40>`… | `<\|tool>`, `<\|think\|>`, `<\|channel>`, `<\|turn>`… |
| `[255999, 262143]` | 6,145 | `<start_of_image>`, `<start_of_turn>`… | `<\|image>`, `<audio\|>`… |

Ids 107 through 255,998 are identical throughout. Confirmed by encoding rather
than by trusting the map: six probe strings spanning English, Python, Norwegian,
math notation, Japanese, and emoji produce identical id sequences under both
tokenizers.

So the delta can be added on a shared index once those two ranges are masked.
This is the constraint that kills cross-*family* transfer. Monad and Baguettotron
share 157 of 8,192 ids ([`../monad-specdec/RESULTS.md`](../monad-specdec/RESULTS.md));
here it costs two slice assignments.

## Logit softcapping

Every Gemma 4 config sets `final_logit_softcapping: 30.0`. No Gemma 3 config
has it.

| model | softcap | | model | softcap |
|---|---|---|---|---|
| gemma-3-270m | `None` | | gemma-4-E2B | `30.0` |
| gemma-3-1b-pt | `None` | | gemma-4-12B | `30.0` |
| gemma-3-27b-it | `None` | | gemma-4-26B-A4B | `30.0` |
| | | | gemma-4-31B(-it) | `30.0` |

Gemma 2 capped, Gemma 3 dropped it for QK-norm, Gemma 4 brought it back. What
Gemma 4 emits is `tanh(l/30) * 30`, bounded and compressed hardest exactly where
the model is most confident. A Gemma 3 delta is unbounded and linear. Adding one to the
other is arithmetic across incommensurable scales, and it degrades worst on the
tokens the base model is surest about, which is where a steering method most
needs to be trustworthy.

Invert before adding: `atanh(clamp(l/30, -1+eps, 1-eps)) * 30` recovers the
pre-cap logits, then apply the delta, then re-cap or don't. `atanh` diverges at
the boundary, so without the clamp the inversion overflows on saturated tokens.
This also means top-k logprobs are not enough. The recipe needs raw logits or a
reliable inverse, which rules out the black-box API case the proxy-tuning paper
demonstrates on GPT-3.5.

## Checkpoints for the recipe

| slot | checkpoint | note |
|---|---|---|
| expert | `google/gemma-3-1b-pt`, fine-tuned | `270m` also has a base/it pair; 270M is far below anything published |
| anti-expert | `google/gemma-3-1b-pt`, untouched | satisfies "the checkpoint the expert was tuned from" |
| base | `google/gemma-4-31B` | the **base**, not `-it` |

Steer the base rather than `31B-it`. The delta encodes generic-base to tuned, and
Liu et al. measured degradation when that contrast is applied to a model already
adapted; their case was CodeLlama.

Overhead is more favorable than the published setup: 1B expert plus 1B
anti-expert against a 31B base is roughly 6% additional FLOPs per token, against
20% for the paper's 7B/70B pairing. Wall-clock depends on whether the three
forwards run concurrently.

## The open question

Shared vocabulary is necessary, not sufficient. Gemma 3 and Gemma 4 differ in
pretraining corpus, post-training recipe, and architecture. Gemma 4 spans dense,
MoE (`26B-A4B` activating 4B of 26B), Per-Layer Embeddings on the E-series, and
an encoder-free `12B`. A delta computed in one model's logit geometry has no
guarantee of meaning the same thing in another's, and no published work tests
transfer across a generation boundary rather than across scale.

`alpha` is where that shows up. If cross-generation transfer works at all, there
is a window where the steered behavior improves without the base's factuality
collapsing. If no such window exists, that is the result.

Cheap to settle: one 1B fine-tune, then an alpha sweep against the target
behavior and a factuality control. Nothing here needs a GPU cluster.

## Relation to speculative decoding

This started from asking whether speculative decoding could merge two models. It
cannot. Verification is rejection sampling against the target, so the output
distribution is exactly the target's and the drafter contributes latency only.
Proxy-tuning and EFT are the mechanisms that do change the distribution.

They compose in the other direction: EFT applies speculative decoding *to* the
merged ensemble, the small fine-tuned model drafting for the combination, at 2.5x
with identical samples. Speculative decoding is the accelerator on a merge, never
the merge.

## Files

- [`tokenizer_diff.py`](tokenizer_diff.py) — downloads both tokenizers and all
  eight configs, diffs vocabularies, runs the encode check, reads softcapping
- [`tokenizer_diff.json`](tokenizer_diff.json) — its output, every number above
