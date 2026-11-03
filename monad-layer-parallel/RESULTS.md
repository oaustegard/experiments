# monad-layer-parallel — results

Pleias' Monad is a 56.7M-parameter language model, 64 layers deep and only 256 wide. `monad-weight-edit` removed most of
its repetition loops by editing its weights; the edited model is `A8p` (called F below). This experiment asks how much faster
and smaller training-free weight surgery can make `A8p` in a browser before its answers get worse. Two surgeries were tried
and combined: quantizing weights to 8 or 4 bits, and folding pairs of adjacent layers into single layers of twice the width
(Layer Parallelism), which cuts the number of sequential steps per token. Everything ran in Chrome on Oskar's Mac through
onnxruntime-web's WebGPU backend. The plan, with its predictions, was registered before any arm ran: [`PLAN.md`](PLAN.md).

## Answer

How much faster and smaller can training-free surgery make the loop-edited Monad without hurting answers? With 8-bit
weights alone, 1.8× faster (11.8 → 6.6 ms per token) and 43% smaller, with no difference detected on the test set.
The 8-bit model decodes in 6.6 ms per token on the Mac (152 tokens/s against 85 for fp32) <!-- @evidence[[results/summary.json#bench.A8p_q8.B1]] -->
and ships 65 MB instead of 114. <!-- @evidence[[results/summary.json#sizes_mb.A8p_q8.shipped]] -->
On the 500-prompt test set it gets 17.4% right against 17.2% for fp32. <!-- @evidence[[results/summary.json#test.A8p_q8.all.acc]] -->

4-bit weights reach 5.5 ms but triple the loop rate to
18.6% <!-- @evidence[[results/summary.json#test.A8p_q4.all.loop]] -->; 4-bit plus 16
folded layer pairs collapsed to 5.8% of answers correct. Folding 8 pairs, tested in fp32, stays within the
plan's rule (≤2 points less accurate, ≤3 points more loops) (16.0% correct <!-- @evidence[[results/summary.json#test.A8p_P8.all.acc]] -->, loops 7.8%), but in 8-bit it
saves only 0.7 ms, and the zero padding on unpaired layers (so every layer has one shape)
grows the download to 109 MB. <!-- @evidence[[results/summary.json#sizes_mb.A8p_P8_q8.shipped]] -->

A 4-bit quantizer that kept loops near fp32's rate would change this: at 4 bits a double-width layer decodes as fast as a
normal one, so folding would pay.

## Findings

F below is the loop-edited model in fp32; "screen KL" is the divergence of an arm's next-token distribution from F's on 16
dev prompts; p-values are McNemar's exact test on per-prompt outcomes against F.

1. 8-bit weights showed no loss: 87 vs 86 correct of 500, p = 1.0 <!-- @evidence[[results/summary.json#stats_vs_F.A8p_q8.all.correct.p]] -->, loops 7.0% against 5.8% (p = 0.48), at 6.6 ms/token against 11.8. One run on one test set: no detectable difference, not proven equivalence (Round 3).
2. 4-bit weights bring the loops back: 18.6% of answers loop against 5.8%, p = 3.6e-11 <!-- @evidence[[results/summary.json#stats_vs_F.A8p_q4.all.loop.p]] -->; overall 12.8% correct, p = 0.035. A screen KL of 0.13 was already enough to do this (Round 2, Round 3).
3. GPTQ and HQQ do not repair Monad's 4-bit error: screen KL 0.267 and 0.136 against 0.132 for round-to-nearest <!-- @evidence[[results/summary.json#screen_kl.A8p_gptq]] --> (Round 3).
4. 4-bit plus 16 folded pairs collapses: 5.8% correct, loops 42.4% <!-- @evidence[[results/summary.json#test.A8p_P16_q4.all.loop]] -->, despite a screen KL of 0.251, under the plan's 0.35 gate for test-set runs (Round 3).
5. Folding 8 pairs passes the decision rule, without a margin: 80 vs 86 correct (p = 0.61), loops 39 vs 29 (p = 0.23); GSM8K falls from 12 to 5 correct, p = 0.092 <!-- @evidence[[results/summary.json#stats_vs_F.A8p_P8.gsm8k.correct.p]] --> (Round 5, run in fp32 as a stand-in for 8-bit).
6. In int4 a double-width layer costs what a normal one does, so folding converts depth directly into latency: 16 pairs 4.4 ms and 24 pairs 3.6 ms <!-- @evidence[[results/summary.json#bench.A8p_P24_q4.B1]] -->, both within 0.1 ms of the plan's cost model (0.6 ms + 0.0775 ms per layer); 8 pairs missed it, 5.3 against 4.9 (Round 2).
7. The int8 weights decode slowly at batch 8: 53.9 ms per step against 12.2 for int4 <!-- @evidence[[results/summary.json#bench.A8p_q8.B8]] -->. Raising the kernel's accuracy_level option to 4 (int8 activations) did not change it (6.8 ms at batch 1). The cause looks like onnxruntime-web's int8 MatMulNBits kernel, but it was not isolated (Round 2).
8. Wider parallel groups cost more KL per layer of depth saved than pairs: depth 48 at KL 0.180 for pairs, 0.256 for groups of three <!-- @evidence[[results/summary.json#depth.groups.3.kl]] --> (Round 4).
9. Parareal over depth, a layer-level form of speculative decoding (guess segment outputs cheaply, then correct them in parallel), gave nothing on 8 dev answers: 16 segments of 4 layers reach KL 4.12 only at iteration 3 <!-- @evidence[[results/summary.json#depth.parareal.16.3.kl]] -->, which already cost as many layer steps as running all 64 in order (Round 4).

## Method

**Models.** F is `A8p` from `monad-weight-edit`, exported in fp32 with onnxruntime-genai's model builder and shipped as
bf16 (114 MB). Quantized arms use the builder's int8 and int4 (round-to-nearest, symmetric, block 32) MatMulNBits weights
with fp32 activations. GPTQ and HQQ int4 arms go through onnxruntime's `MatMulNBitsQuantizer` (`quant_alt.py`).

**Folding.** `fold.py` turns a pair of adjacent layers (A, B) into one standard Llama layer with 8 heads and a 1,536-wide MLP:
q/k/v and gate/up rows are concatenated, o_proj and down_proj columns stacked, so both attention blocks read the same input
x and both MLPs read h = x + attn_A(x) + attn_B(x). The merged RMSNorm weight is the mean of the pair's two; each layer's ratio
to that mean is folded into its own input projections. Unpaired layers are zero-padded to the same width so the builder sees
one uniform config. Pairs are chosen greedily by lowest single-pair KL on dev answers, without overlaps. Every folded fp32
model matched the PyTorch pair wrapper (`../monad-weight-edit/lp_probe.py`) to a maximum logit difference below 1e-3.

**Screen.** KL from F's next-token distribution, regenerated on 16 dev prompts on CPU (`screen_kl.json`).

**Latency.** Decode milliseconds per token on the Mac (Chrome 156, Metal), median of three runs, batch 1 and batch 8
(`../monad-weight-edit/bridge.py bench`). Tokens per second is 1000 / ms.

**Test.** The pinned 500-prompt test set of `monad-weight-edit` (200 MMLU, 150 GSM8K, 150 IFEval), greedy, batch 8 on the
Mac, scored by that experiment's `summarize.py`: accuracy per task and loop rate (the answer ends in a repeating cycle at the
length limit). Paired comparisons against F use McNemar's exact test on the per-prompt outcomes. F's own test run is the one
`monad-weight-edit` reported.

**Depth probe.** `depth_probe.py`, on CPU, teacher-forced on 8 dev answers of 256 tokens: (a) groups of 2, 3 or 4 consecutive
layers run in parallel, chosen like pairs, at the same total depth; (b) Parareal over depth, a layer-level form of
speculative decoding that guesses each segment's output with the segment's layers run in parallel, then corrects with
the exact layers run from all guesses at once.

## Log

### Round 1 — premises (2026-11-03)

Recorded in `PLAN.md` and `results/premises.json`. Batch 8 costs about the same as batch 1 in fp32, but random-weight
models of other shapes show that is because weights are read once per step, not because decoding is dispatch-bound: an fp32
layer of Monad's width costs 0.172 ms, one of twice the width 0.228 ms. In int4 the double-width layer costs the same as a
normal one (0.0775 against 0.075 ms), which is what makes folding pairs pay. Pairs cost about a tenth of the KL of deleting
the same number of layers (8 pairs 0.055 against 0.570 for 8 deleted layers).

### Round 2 — arms built and screened

The first fold multiplied each layer's raw RMSNorm weights into its projections and set the merged norm to 1. Monad's norm
weights span 16×, and those scaled matrices then quantized badly in blocks of 32: int4 + 16 pairs screened at KL 0.54.
Keeping the mean of the pair's norm weights as the merged norm and folding only the ratios brought it to 0.25, which met
prediction 3's additivity (0.27 ± 0.05: int4 0.13 plus 16 pairs 0.12).

| arm | screen KL | ms/token B=1 | tok/s | ms/step B=8 | shipped MB |
|---|---|---|---|---|---|
| F fp32 | 0 | 11.8 | 85 | 12.7 | 114 |
| int8 | 0.003 | 6.6 | 152 | 53.9 | 65 |
| int8, accuracy_level 4 | — | 6.8 | 147 | 54.0 | 65 |
| int4 | 0.132 | 5.5 | 182 | 12.2 | 32 |
| int4 HQQ | 0.136 | — | — | — | — |
| int4 GPTQ | 0.267 | — | — | — | — |
| fp32 + 8 pairs | 0.062 | — | — | — | — |
| fp32 + 16 pairs | 0.120 | — | — | — | — |
| int8 + 8 pairs | 0.066 | 5.9 | 169 | 73.6 | — |
| int8 + 16 pairs | 0.122 | 5.0 | 200 | 63.2 | — |
| int8 + 24 pairs | 0.347 | 4.3 | 233 | 52.6 | — |
| int4 + 8 pairs | 0.184 | 5.3 | 189 | 14.4 | 56 |
| int4 + 16 pairs | 0.251 | 4.4 | 227 | 12.3 | 49 |
| int4 + 24 pairs | 0.506 | 3.6 | 278 | 10.5 | 42 |

Prediction 1 (latency within 0.3 ms of 0.6 + 0.0775 × layers): int4 + 16 pairs 4.4 against 4.3 and int4 + 24 pairs 3.6
against 3.7 met it; int4 + 8 pairs, 5.3 against 4.9, missed. Per the decision rule the cost model is not extrapolated to
other depths.

### Round 3 — test set

| arm | correct | MMLU | GSM8K | IFEval | loops | correct vs F (p) |
|---|---|---|---|---|---|---|
| F | 17.2% | 27.0% | 8.0% | 13.3% | 5.8% | — |
| int8 | 17.4% | 26.5% | 8.0% | 14.7% | 7.0% | 87 vs 86 (1.0) |
| int4 | 12.8% | 25.5% | 2.0% | 6.7% | 18.6% | 64 vs 86 (0.035) |
| int4 + 16 pairs | 5.8% | 12.5% | 1.3% | 1.3% | 42.4% | 29 vs 86 (4.6e-9) |

int4 failed prediction 2 (within 2 points of F, loops under 8%): it is 4.4 points down and loops on 18.6% of answers, three
times F's rate, most of it on IFEval (41%), the task where the loop edit had done its work. The decision rule then called for a
GPTQ arm before pairing; GPTQ screened worse than round-to-nearest (0.267 against 0.132), and HQQ was no better (0.136). int4
+ 16 pairs, screened at 0.25, under the 0.35 gate, collapsed: loops on 42%, 5.8% correct. Prediction 3 failed. The gate was
set from KL alone, and a screen KL of 0.13 already tripled the loop rate. int8 passed on every task.

### Round 4 — further parallelism and layer-level speculation

Oskar asked whether wider groups could use more of the GPU and whether speculative decoding could run at the layer level.
Depth probe at depth about 48: pairs KL 0.180, groups of three 0.256, groups of four (depth 46) 0.447. Wider groups cost more
for the same depth. Parareal over depth is exact after as many iterations as segments, but each iteration fixes about one
segment: the coarse guesses of segment boundaries are off by roughly the size of the attention sink, so KL stays between 4 and
17 until the last iteration (16 segments of 4 layers: 10.6, 16.8, 9.3, 4.1 after 0 to 3 iterations), by which point it has
cost more layer steps than running all 64 in order. No gain.

### Round 5 — 8 pairs on the test set

The 8-pair fold had not been planned for the test set: the plan's gate admitted the fastest arm under KL 0.35, and that
arm collapsed. After round 3 the KL screen could no longer pick arms, so the 8-pair fold was tested directly. It ran in fp32
as a stand-in for 8-bit + 8 pairs, since the two screened alike (0.062 and 0.066) and 8-bit alone matched fp32 on the test
set. Six IFEval batches overran the 600 s job cap and were rerun as two further jobs.

| arm | correct | MMLU | GSM8K | IFEval | loops | correct vs F (p) | loops vs F (p) |
|---|---|---|---|---|---|---|---|
| F | 17.2% | 27.0% | 8.0% | 13.3% | 5.8% | — | — |
| fp32 + 8 pairs | 16.0% | 29.0% | 3.3% | 11.3% | 7.8% | 80 vs 86 (0.61) | 39 vs 29 (0.23) |

It sits inside the decision rule's bounds (−1.2 points, +2.0 points of loops). GSM8K fell from 12 to 5 correct (p = 0.09).
That is not significant at this sample size but is the largest per-task move. In 8-bit the fold buys 0.7 ms (5.9 against
6.6) and costs 44 MB of download (109 against 65 MB), because the zero padding on unpaired layers is stored as int8 weights
and scales. In int4 the same padding cost about 3% per layer. The recommended build is 8-bit alone.
