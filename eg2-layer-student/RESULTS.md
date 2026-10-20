# RESULTS: a query encoder cut from EmbeddingGemma 2's own first layers

Issue: [#103](https://github.com/oaustegard/experiments/issues/103), following
[#102](../eg2-query-student/RESULTS.md). Run 2026-10-20 in a 4-vCPU CCotw container, about
8 hours of CPU training, no GPU and no API spend. What broke is in [`ERRORS.md`](ERRORS.md).

## Answer

Can EmbeddingGemma 2 (EG2) cut to its first k layers be the browser query encoder for
EG2-indexed posts? For retrieval, mostly. Its first 4 layers, distilled for two epochs and
shipped as q4 ONNX with a 32K-token vocabulary, download as 29.0 MB <!-- @evidence[[results/final.json#/L2 k4_ep2_v32k q4/mb]] -->
and find a synthetic query's source post 90% of the time <!-- @evidence[[results/final.json#/L2 k4_ep2_v32k q4/synth_test/hit@10/0]] -->
against EG2's 91.9%, a paired change of -0.019. <!-- @evidence[[results/final.json#/L2 k4_ep2_v32k q4/synth_test/hit@10_minus_eg2/0]] -->
On real queries it shares 0.588 of EG2's top 10. <!-- @evidence[[results/final.json#/L2 k4_ep2_v32k q4/real_test/overlap@10/0]] -->
The 2-layer build is 21.8 MB <!-- @evidence[[results/final.json#/L2 k2_ep2_v32k q4/mb]] -->
with hit@10 0.889 <!-- @evidence[[results/final.json#/L2 k2_ep2_v32k q4/synth_test/hit@10/0]] -->
and agreement 0.546, <!-- @evidence[[results/final.json#/L2 k2_ep2_v32k q4/real_test/overlap@10/0]] -->
smaller and better on all three than #102's best student
(24.4 MB, 0.794, 0.482). <!-- @evidence[[results/final.json#/#102 A3 leaf_cont165k epoch0 int8/real_test/overlap@10/0]] -->

Agreement stays far below pruned EG2's 0.932 (102.9 MB). <!-- @evidence[[results/final.json#/#102 A4 pruned EG2 K=32000/real_test/overlap@10/0]] -->
Training had not converged: a second epoch added 0.036 at k = 4. <!-- @evidence[[results/pairs.json#second_epoch_k4.real_test.0]] -->

## Findings

1. **A cut of EG2 beats an unrelated student at the same size.** k = 2, q4, 21.8 MB:
   0.546 real, 0.667 synthetic, hit@10 0.889; <!-- @evidence[[results/final.json#/L2 k2_ep2_v32k q4/synth_test/overlap@10/0]] -->
   #102's leaf-mt student, int8, 24.4 MB: 0.482, 0.513, 0.794. <!-- @evidence[[results/final.json#/#102 A3 leaf_cont165k epoch0 int8/synth_test/hit@10/0]] -->
   (Round 3, `run/final.py`)
2. **Untrained, the cut is useless.** Top-10 agreement is 0.014 or less for every k up to 8,
   0.411 at k = 20, <!-- @evidence[[results/l0.json#/20/real_test/overlap@10/0]] -->
   and 1.000 only at k = 24. The intermediate layers are not in the output space; the
   training does the work: k = 4 went from 0.003 to 0.573 in one epoch. <!-- @evidence[[results/l1_k4.json#/evals/step2572/real_test/overlap@10/0]] -->
   (Round 1, Round 2)
3. **Training time is worth more than depth.** With one epoch each, k = 8 beats k = 4 by
   0.032 on real queries; <!-- @evidence[[results/pairs.json#k8_vs_k4_one_epoch.real_test.0]] -->
   k = 4 with two epochs against k = 8 with one: +0.004 [−0.002, +0.011], <!-- @evidence[[results/pairs.json#k4_two_epochs_vs_k8_one.real_test.0]] -->
   at 29.0 against 44.1 MB. <!-- @evidence[[results/final.json#/L2 k8 q4/mb]] -->
   (Round 3, `run/pairs.py`)
4. **The remaining error is structured.** The best student, k = 4 after two epochs in fp32,
   reaches cosine 0.948 to EG2 <!-- @evidence[[results/final.json#/L1 k4_ep2 step2572/real_test/cos_to_eg2]] -->
   and 0.608 agreement; <!-- @evidence[[results/final.json#/L1 k4_ep2 step2572/real_test/overlap@10/0]] -->
   in #102, isotropic noise at cosine 0.958 kept 0.81. The student errs along the
   directions that decide this corpus's ranking. (Round 3)
5. **Training on the cut vocabulary wins back part of what the cut costs.** Cutting the
   32K vocabulary after training costs 0.028 real (0.608 to 0.580). <!-- @evidence[[results/final.json#/L2 k4_ep2 q4/real_test/overlap@10/0]] -->
   Training the second epoch on the cut tokenizer instead: +0.008 [+0.002, +0.015] real, <!-- @evidence[[results/pairs.json#train_on_cut_vs_cut_after.real_test.0]] -->
   +0.016 synthetic, <!-- @evidence[[results/pairs.json#train_on_cut_vs_cut_after.synth_test.0]] -->
   +0.010 hit@10. <!-- @evidence[[results/pairs.json#train_on_cut_vs_cut_after.hit@10.0]] -->
   (Round 3)
6. **The bytes are the token table plus 3.4 MB a layer.** At q4 the 41,741-row table is
   13.69 MB, <!-- @evidence[[results/sizes.json#/l2_k2/embed_tokens/mb]] -->
   two layers 6.74 MB, <!-- @evidence[[results/sizes.json#/l2_k2/layers/mb]] -->
   four 13.48 MB. <!-- @evidence[[results/sizes.json#/l2_k4/layers/mb]] -->
   So k = 2 at 32K tokens is near the floor; a smaller vocabulary is the next lever, at
   #102's measured cost (16K cost full EG2 0.08 agreement). (`run/sizes.py`)
7. **q4 is smaller and closer than int8.** k = 2: q4 21.8 MB at minimum cosine 0.992 to the
   fp32 student; int8 33.7 MB <!-- @evidence[[results/final.json#/L2 k2_ep2_v32k int8/mb]] -->
   at 0.987. <!-- @evidence[[results/export_l2_k2_ep2_v32k.json#formats.int8.cos_to_pytorch.min]] -->
   The weights start on EG2's q4 grid, which q4 reproduces. (Round 3)
8. **The shipped file runs on WebGPU, not wasm.** onnxruntime-web 1.30 on SwiftShader WebGPU
   returns the CPU vectors (minimum cosine 0.9999999); <!-- @evidence[[results/smoke_web_l2_k4_ep2_v32k.json#webgpu.cos_to_cpu_min]] -->
   its wasm backend has no GatherBlockQuantized kernel, the op EG2's own q4 graph also uses
   for its token table. (Round 4, `run/smoke_web.py`)
9. **The bf16 release is not the teacher.** EG2's bf16 weights agree with the q4 teacher
   that built the index on 0.87 of the top 10; the student starts from the q4 weights,
   dequantized, which reproduce it exactly. (ERRORS 1)

## Method

**Fixture.** Unchanged from #102: the 61,686-post Bluesky index of
`embeddinggemma2-webgpu-tail` (EG2 q4, remex 2-bit codes), real_test (1,000 held-out
MS MARCO tech/AI queries) and synth_test (1,498 Haiku queries from held-out posts). Teacher
targets: embstore `bsky-ai-list/embeddinggemma-2-q4/query-targets-v1`. Training set: all
164,588 training targets (msmarco_train, synth_train, gpu_more, and gpu_posts snippets
rebuilt from re-hydrated post text; none dropped).

**Student.** `run/model.py`: transformers' `EmbeddingGemma2TextModel` with k layers, the
token table, the first k slices of the per-layer-input (PLE) projection, the final norm, the
512→768 projection, mean pooling over all tokens including the prompt, and L2
normalization. No new weights. Quantized weights come from the q4 ONNX, dequantized (fused
qkv and gate/up split, the folded embedding scale divided out); norms come from the bf16
release. k = 24 reproduces the teacher (overlap 1.000).

**Where the parameters live.** Text tower 271M: token table 134.2M (262,144 × 512),
24 layers 130.1M (5.25M each, 6.3M for the four full-attention layers), PLE projection 6.3M,
head 0.4M.

**Arms.** L0: the cut, untrained, k ∈ {1, 2, 3, 4, 6, 8, 12, 16, 20, 24}. L1: the cut trained on
1 − cos(student, EG2), everything except the token table (kept as the teacher's), AdamW,
lr 3e-3 with 100 warmup steps and linear decay, batch 64, length-bucketed, one epoch
(2,572 steps); continuation epochs at lr 1e-3 with a new batch order. L2: an L1 student with
#102's A4 vocabulary cut (top 32,000 tokens by frequency, closed under BPE: 41,741 rows,
pruned `tokenizer.json`), either cut after training or trained for the second epoch on the
cut tokenizer, with the cut table trainable.

**Shipping.** `run/export.py`: ONNX export (inputs `input_ids`, `attention_mask`; output the
unit embedding), q4 by onnxruntime `MatMulNBitsQuantizer` (4-bit, block 32, asymmetric,
MatMul and Gather, token table repacked to EG2's uint8 layout) and int8 by
`quantize_dynamic`. Every number in an L2 row is measured on the exported file in
onnxruntime. Download = ONNX graph + external data + gzip `tokenizer.json`.

**Metrics.** As #102: top-10 overlap with EG2 on the 2-bit codes, bootstrap 95% CIs (2,000
resamples); synth_test source-post hit@10 with the paired difference to EG2; paired CIs
between builds on the same queries (`run/pairs.py`).

## Log

### Round 1: setup and L0 (2026-10-20)

Fetched EG2 (bf16 safetensors and q4 ONNX), the 165K targets, and the post text (0 of 61,686
missing). The bf16 model matched the fp32 ONNX (cosine 0.9999998) but only 0.87 of the q4
teacher's top 10, so the student was rebuilt on dequantized q4 weights (ERRORS 1, 2). A
config rewrite of the last layer's attention type was caught and fixed before the L0 sweep
was rerun (ERRORS 3).

L0, real / synthetic overlap, hit@10:

| k | 1 | 2 | 4 | 8 | 12 | 16 | 20 | 24 |
|---|---|---|---|---|---|---|---|---|
| real | 0.003 | 0.000 | 0.003 | 0.014 | 0.075 | 0.202 | 0.411 | 1.000 |
| synthetic | 0.004 | 0.001 | 0.004 | 0.013 | 0.058 | 0.252 | 0.470 | 1.000 |
| hit@10 | 0.014 | 0.003 | 0.012 | 0.031 | 0.149 | 0.666 | 0.856 | 0.919 |

Cosine to EG2 on real queries ran 0.05 (k = 1) to 0.84 (k = 20).

### Round 2: L1 learning rate and depth (2026-10-20)

k = 4, 300 steps: lr 1e-4 0.100 real, 3e-4 0.237, 1e-3 0.330, 3e-3 0.336. #102's 5e-5 regime
is too slow for a cut model; 3e-3 for all first epochs. Steps took 5.0 s with random batches
and about 1.7 s length-bucketed (ERRORS 6).

One epoch each, fp32 PyTorch, full vocabulary (real / synthetic / hit@10):

| k | step 1,000 | step 2,000 | step 2,572 | CPU time |
|---|---|---|---|---|
| 2 | 0.455 / 0.565 / 0.859 | 0.508 / 0.623 / 0.872 | 0.518 / 0.643 / 0.875 | 0.7 h |
| 4 | 0.477 / 0.580 / 0.859 | 0.547 / 0.655 / 0.879 | 0.573 / 0.680 / 0.889 | 1.4 h |
| 8 | 0.502 / 0.589 / 0.863 | 0.571 / 0.675 / 0.889 | 0.600 / 0.703 / 0.897 | 2.4 h |

The k = 8 run was stopped by the 2-hour background cap at step 2,100 and resumed from its
checkpoint.

### Round 3: L2 and continuation epochs (2026-10-20)

Second epochs at lr 1e-3 from the k = 4 and k = 2 students, one on the full vocabulary and
the rest on the cut tokenizer. Final table, q4 ONNX unless marked:

| build | download | real | synthetic | hit@10 | vs EG2 |
|---|---|---|---|---|---|
| EG2 (#102) | 179.9 MB | 1 | 1 | 0.919 | — |
| pruned EG2, 32K (#102) | 102.9 MB | 0.932 | 0.947 | 0.917 | −0.002 |
| leaf-mt student, int8 (#102) | 24.4 MB | 0.482 | 0.513 | 0.794 | −0.125 |
| k = 2, 1 epoch, cut after | 21.8 MB | 0.498 | 0.617 | 0.870 | −0.049 |
| k = 2, 2nd epoch on cut | 21.8 MB | 0.546 | 0.667 | 0.889 | −0.030 |
| k = 2, 2nd epoch on cut, int8 | 33.7 MB | 0.537 | 0.662 | 0.881 | −0.038 |
| k = 4, 1 epoch, cut after | 29.0 MB | 0.544 | 0.651 | 0.878 | −0.041 |
| k = 4, 2 epochs, cut after | 29.0 MB | 0.580 | 0.681 | 0.891 | −0.029 |
| k = 4, 2nd epoch on cut | 29.0 MB | 0.588 | 0.697 | 0.901 | −0.019 |
| k = 8, 1 epoch, cut after | 44.1 MB | 0.576 | 0.674 | 0.887 | −0.032 |
| k = 4, 2 epochs, full vocabulary (fp32, not shipped) | — | 0.608 | 0.708 | 0.902 | −0.017 |

Midway through the runs, the prediction given to Oskar was 22–29 MB, hit@10 0.89–0.90,
and agreement plateauing around 0.60–0.65. Measured: 21.8–29.0 MB, 0.889–0.901, and 0.546–0.588
after two epochs, still rising.

### Round 4: browser check (2026-10-20)

`run/smoke_web.py` loads the shipped q4 file in onnxruntime-web 1.30 in headless Chromium.
WebGPU (SwiftShader): 32 queries, minimum cosine 0.9999999 to onnxruntime CPU, for both
the k = 4 and k = 2 builds. wasm: session creation fails, "Could not find an implementation
for GatherBlockQuantized(1)", before and after repacking the table to EG2's uint8 layout.
