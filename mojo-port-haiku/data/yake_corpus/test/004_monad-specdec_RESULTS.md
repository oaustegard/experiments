# Monad as a speculative decoder for Baguettotron

Monad drafting for Baguettotron is slower than running Baguettotron alone, at
every draft length tested. The best configuration (γ=1) reaches 0.90× baseline
throughput; γ=8 reaches 0.42×. The draft model is a decent predictor and the
verification loop is lossless — the cost is that Monad is only 2.1× faster per
token than the model it drafts for, and speculative decoding needs a much
larger gap than that.

Both models come from PleIAs, both are `LlamaForCausalLM` trained on SYNTH.
Monad is 56.7M parameters (64 layers × 256 hidden, 8,192-token vocabulary);
Baguettotron is 321M (80 layers × 576 hidden, 65,536-token vocabulary).

## Setup

The two models do not share a tokenizer. 7,397 of Monad's 8,192 tokens exist as
strings in Baguettotron's vocabulary, but only 157 share an id, and those 157
are the ASCII prefix. Both are byte-level BPE with the same pre-tokenizer and the same four
special tokens, so the vocabularies are cousins from one training pipeline, not
one truncated from the other.

That rules out comparing draft ids against target logits directly.
`specdec.py` uses string-level exact match (Timor et al. 2025): the drafter
extends the committed text by γ of its own tokens, the result is detokenized,
re-tokenized under Baguettotron's vocabulary, and verified position by position
against Baguettotron's own greedy argmax. Each model keeps a KV cache addressed
by token ids rather than by length, because re-tokenizing a growing string moves
the boundaries near its end; on each round the cache is cropped to the
longest shared prefix and only the divergent suffix re-runs.

Every run in the table below produced token-for-token identical output to plain
greedy decoding, which is the check that the loop is doing real verification and
not silently accepting.

Measurements are CPU, 4 threads, fp32, batch 1, 5 prompts × 48 new tokens.
Baseline and speculative runs are interleaved per configuration, because the
container's shared cores drift 20–30% over a long run and an early baseline
compared against a late speculative run measures the machine.

## Results

| γ | acceptance | accepted/round | speedup | identical to greedy |
|---|---|---|---|---|
| 1 | 0.546 | 0.53 | 0.90× | yes |
| 2 | 0.474 | 0.84 | 0.84× | yes |
| 3 | 0.399 | 1.07 | 0.71× | yes |
| 4 | 0.381 | 1.33 | 0.64× | yes |
| 8 | 0.249 | 1.63 | 0.42× | yes |

Measured speedups sit close to what the Leviathan et al. formula predicts from
the measured acceptance and cost ratio (0.90 measured vs 0.97 predicted at γ=1),
so the implementation's own overhead accounts for only a few percent.

## Where the time goes

Three factors compound. **Depth sets latency, not parameter count.** Monad
has 1/5.7 of Baguettotron's parameters but 64/80 of its layers, and decodes a
token in 57.0 ms against 119.9 ms. That is a ratio of 2.1×, a per-step cost
ratio of c = 0.476.

Truncating each model's layer stack and fitting a line through the depths gives
0.804 ms per layer for Monad at width 256, and 1.264 ms per layer for
Baguettotron at width 576. A transformer layer's FLOPs go as width², so a
compute-bound decode would show a per-layer ratio near 2.25² = 5.06. The measured
ratio is 1.57. At these widths and batch 1, fixed overhead dominates per-layer cost (kernel
dispatch, normalization, cache handling), which makes that cost nearly
independent of width. A narrower model of the same depth is barely cheaper.

**A smaller vocabulary needs more draft steps.** Monad averages 3.25 characters
per token against Baguettotron's 4.12, so covering one target token takes 1.27
draft steps. That lifts the effective cost of drafting one target token's worth
of text from c = 0.476 to c = 0.602.

**Acceptance lands just under break-even.** At γ=1 the loop breaks even when
acceptance equals the effective cost ratio. Break-even is 0.602; measured
acceptance is 0.546.

## Monad as a predictor

0.546 acceptance is respectable for a model 5.7× smaller reading through a
different vocabulary. For comparison, Baguettotron's own first N layers agree
with its full stack on 2.1% of tokens at 20 layers, 7.3% at 40, and 25.0% at 60
Layer-skip self-speculation with no trained early-exit head predicts
Baguettotron far worse than Monad does. Projected through the same formula, none
of those depths beats baseline either.

Monad's cost is what disqualifies it here. Its agreement rate is fine.

## Sizing a viable draft model

Hold Monad's measured acceptance and remove the granularity penalty by sharing
Baguettotron's tokenizer: break-even at γ=1 drops to c = 0.476 and the formula
gives 1.05×. That is still inside noise.

Reaching a useful 1.3× at γ=2 needs c ≈ 0.21, or about 22 ms per token. At
Monad's measured 0.804 ms per layer that is roughly a 24-layer draft at width
256 that shares the target's vocabulary.

That projection assumes acceptance stays at 0.546 for a much shallower model,
which it would not. Treat 24 layers as a floor on the draft budget rather than a
recipe.

## Limits

CPU only, 4 threads, fp32, batch 1, greedy decoding, 5 prompts, 48 new tokens
each. Acceptance is measured on those 5 prompts and will move with domain.

The GPU regime is untested here. The depth argument gets worse rather than
better on a GPU: per-layer kernel launch overhead is a larger share of a much
shorter layer time, which pushes the per-layer cost ratio further toward 1 and
the whole speedup further toward the 64/80 depth ratio.

Related, from `METHODS.md`: the memory-bandwidth ceiling entry covers the GPU
case where speculation lets a quoted throughput exceed the bandwidth bound. That constraint is bandwidth; this one is per-layer overhead.

## Files

| File | What |
|---|---|
| `specdec.py` | Cross-tokenizer speculative decoding loop and cached runners |
| `run_eval.py` | Interleaved baseline vs speculative sweep over γ → `results.json` |
| `bench_latency.py` | Per-token decode latency for both models → `latency.json` |
| `depth_scaling.py` | Baguettotron latency vs truncated depth → `depth_scaling.json` |
| `depth_scaling_monad.py` | Per-layer marginal cost, both models → `depth_scaling_both.json` |
| `self_spec.py` | Layer-skip self-speculation arm → `self_spec.json` |
| `analyze.py` | Granularity, cost ratios, theory vs measured → `analysis.json` |
| `diag_determinism.py` | Confirms the target is schedule-deterministic |
| `test_cache.py` | Confirms KV crop-and-reuse matches a fresh run |

Reproduce with `python3 bench_latency.py && python3 run_eval.py && python3 analyze.py`.
