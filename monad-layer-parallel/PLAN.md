# monad-layer-parallel — plan

Follow-up to [`monad-weight-edit/`](../monad-weight-edit/RESULTS.md), which removed most of Pleias Monad's repetition loops
by editing its weights. The question here is speed and size: how fast and how small can training-free weight surgery make
the loop-edited Monad (`A8p`) in a browser before its answers get worse? Registered 2026-11-03, before any arm below has run.

## What the premise measurements found

Monad has 64 layers, each 256 wide with 4 attention heads of 64 and an MLP 768 wide. All latency is decode time per token on
Oskar's Mac (Chrome 156, Metal, onnxruntime-web 1.30 WebGPU), batch 1, from `results/premises.json`. Quality is KL to the
full model's next-token distribution, measured on the greedy answers the full model gave to the 192 dev prompts of
`monad-weight-edit`. Models are exported with onnxruntime-genai's model builder.

- **Batch 8 costs about the same as batch 1 in fp32 (13.0 vs 11.8 ms), but that does not make decoding dispatch-bound.** Weights
  are read once per step whatever the batch. Random-weight models of other shapes separate the costs: an fp32 layer of
  Monad's width costs 0.172 ms and a layer of twice its width (8 heads, a 1,536-wide MLP) 0.228 ms, plus 0.8 ms per token
  that does not depend on depth. About 0.115 ms of each layer is a fixed per-layer cost that width does not change; the rest
  grows with the bytes of weight read.
- **4-bit weights change the regime.** With int4 weights (MatMulNBits, fp32 activations) the 64-layer model decodes in 5.4 ms
  instead of 11.8, and a layer of twice the width costs the same as a normal one: 0.0775 against 0.075 ms per layer, plus 0.6 ms
  fixed. 8-bit weights give 7.6 ms. At batch 8 the int8 kernel is slow (54 ms); int4 stays at 12.2 ms.
- **Quantization costs quality, spread over the whole model.** int8: KL 0.002. int4, round-to-nearest, block 32: KL 0.111;
  the builder's k-quant mixed mode 0.091. `monad-weight-edit` found that MLP neurons in layers 22, 24 and 29 build the
  attention sink, the very large vector Monad keeps on its first token. Keeping those three layers' MLP output matrices at full
  precision changed nothing (0.109, against 0.110 for layers 40, 45 and 50), and neither did the output head (0.102). Quantizing only the
  MLPs gives 0.059 and only attention 0.067, so the error accumulates over all 64 layers rather than sitting in a few matrices.
- **Running adjacent layers as parallel pairs costs about a tenth of deleting layers.** Layer Parallelism (Calvo González et al.
  2025, arXiv 2502.02790) computes two consecutive layers from the same input: both attention blocks read x, both MLPs read the
  summed result. Pairs chosen by lowest single-pair KL: 8 pairs 0.055, 16 pairs 0.158, 24 pairs 0.451, 28 pairs 10.7. Deleting
  8 layers costs 0.570 and 16 layers 1.361. The costliest pairs are (0,1) at 7.35 and pairs inside layers 22–31, which hold the
  sink-building neurons and the attention blocks that write the loop direction in `monad-weight-edit`; the KL ranking avoids them without being told.

The LP paper reports that fusing a pair on one GPU gives no speed-up, because large models are compute- and
bandwidth-bound; its gain comes from fewer all-reduces across GPUs. Monad in int4 on a browser GPU is the opposite case: a
fused pair is one layer of twice the width and costs the same as one normal layer. Did not find single-device pair fusion
for small models in three searches (LP; ReplaceMe, arXiv 2505.02819; LayerMerge for CNNs).

## The experiment

A training-free transform of `A8p`: pairs of adjacent layers are folded into single layers of twice the width (concatenated
QKV, gate and up projections; stacked o_proj and down_proj columns; each layer's RMSNorm weight folded into its own input
projections), then all weights are quantized to int4. Unpaired layers are zero-padded to the same width so the
onnxruntime-genai builder sees one uniform config; in int4 the padding costs about 3% per layer. Before export, the folded
fp32 model must match the PyTorch pair wrapper in `../monad-weight-edit/lp_probe.py` to a maximum logit difference of 1e-3.
int4 means the builder's default: round-to-nearest, symmetric, block 32.

| arm | layers | weights | predicted ms/token | predicted size |
|---|---|---|---|---|
| F (reference, existing) | 64 | fp32 | 11.8 measured | 227 MB (114 MB shipped as bf16) |
| Q8 | 64 | int8 | 7.6 measured | 73 MB |
| Q4 | 64 | int4 | 5.4 measured | 36 MB |
| Q4-P8 | 56 | int4, 8 pairs | 4.9 | ~63 MB |
| Q4-P16 | 48 | int4, 16 pairs | 4.3 | ~54 MB |
| Q4-P24 | 40 | int4, 24 pairs | 3.7 | ~45 MB |

Predicted latency is 0.6 ms + layers × 0.0775 ms. Sizes count every layer at double width, since padded single layers store
zeros; zero blocks would shrink further under HTTP compression, which is not counted. Pairs are chosen by single-pair KL on
the dev set, recomputed for `A8p`. Every arm is screened by dev-set KL against F on CPU, and every arm's latency is benched on
the Mac at batch 1 and 8. The pinned 500-prompt test set (200 MMLU, 150 GSM8K, 150 IFEval, greedy, from `monad-weight-edit`)
then runs on the Mac for F, Q4, and the fastest paired arm whose dev KL is at most 0.35, scoring loop rate and accuracy per
task. On that set F loops on 5.8% of answers and gets 17.2% right overall.

## Predictions

1. Every paired arm's batch-1 latency is within 0.3 ms of the cost model: Q4-P16 4.3 ms, Q4-P24 3.7 ms (2.7× and 3.2× faster
   than F).
2. Q4 alone is within 2 points of F's overall test accuracy and loops on under 8% of answers.
3. Q4-P16's dev KL is 0.27 ± 0.05 (the two transforms' KLs add), so it passes the 0.35 gate and goes to the test set, where it
   is within 3 points of F's overall accuracy. Q4-P24's dev KL is above 0.35, so it is screened out.

## Decision rules

- The recommended build is the fastest tested arm with overall accuracy no more than 2 points below F and a loop rate no more
  than 3 points above F.
- If Q4 itself fails prediction 2, add a GPTQ int4 arm (calibration on the dev set, closed-form error compensation, no
  gradient descent) before any pairing.
- If any paired arm misses prediction 1, the cost model is wrong for real weights; report the measured latencies and do not
  extrapolate to other depths.

## Ruled out by the premises

- Width pruning (fewer heads or MLP neurons): in int4 a layer's width does not change its cost.
- Deleting layers: costs about ten times the KL of pairing for the same depth.
- Sink-aware mixed precision: no effect on int4 error (above).
