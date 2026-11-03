# monad-layer-parallel — results

Pleias' Monad is a 56.7M-parameter language model, 64 layers deep and only 256 wide. `monad-weight-edit` removed most of
its repetition loops by editing its weights; the edited model is `A8p` (called F below). This experiment asks how much faster
and smaller training-free weight surgery can make `A8p` in a browser before its answers get worse. Two surgeries were tried
and combined: quantizing weights to 8 or 4 bits, and folding pairs of adjacent layers into single layers of twice the width
(Layer Parallelism), which cuts the number of sequential steps per token. Everything ran in Chrome on Oskar's Mac through
onnxruntime-web's WebGPU backend. The plan, with its predictions, was registered before any arm ran: [`PLAN.md`](PLAN.md).

## Answer

ANSWER_TBD

## Findings

FINDINGS_TBD

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

P8_TBD
