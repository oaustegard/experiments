# RESULTS: a query encoder cut from EmbeddingGemma 2's own first layers

Issue: [#103](https://github.com/oaustegard/experiments/issues/103), following
[#102](../eg2-query-student/RESULTS.md). Run 2026-10-20 to 10-21: about 8 hours of CPU
training in a 4-vCPU CCotw container, then about 3 hours of training on Oskar's M5 GPU; no
API spend. What broke is in [`ERRORS.md`](ERRORS.md).

## Answer

Can EmbeddingGemma 2 (EG2) cut to its first k layers be the browser query encoder for
EG2-indexed posts? For retrieval, yes. Its first 4 layers, distilled for 10 epochs on a
32K-token vocabulary, with q4 layers and a 3-bit token table, download as 25.4 MB <!-- @evidence[[results/final.json#/L2 mac_k4 t3w/mb]] -->
and find a synthetic query's source post 90.7% of the time <!-- @evidence[[results/final.json#/L2 mac_k4 t3w/synth_test/hit@10/0]] -->
against EG2's 91.9%. <!-- @evidence[[results/final.json#/#102 EG2 (teacher)/synth_test/hit@10/0]] -->
On real queries it shares 0.623 of EG2's top 10. <!-- @evidence[[results/final.json#/L2 mac_k4 t3w/real_test/overlap@10/0]] -->
It runs in onnxruntime-web on wasm and WebGPU. Four layers is the elbow: eight add 15 MB
for +0.020 agreement <!-- @evidence[[results/pairs.json#mac_k8_vs_k4_10ep.real_test.0]] -->
and no measurable hit@10. The smallest build, 2 layers with a 2-bit table, is 15.50 MB <!-- @evidence[[results/final.json#/L2 mac_k2 t2w/mb]] -->
at hit@10 0.890. <!-- @evidence[[results/final.json#/L2 mac_k2 t2w/synth_test/hit@10/0]] -->

Full EG2 keeps more of its ranking but stays large: with 3-bit layers it is 75.2 MB at
0.780 agreement. <!-- @evidence[[results/quant_full.json#/gq:3:32/real_test/overlap@10/0]] -->

What would change this: a cut that keeps EG2's last layer, or a smaller vocabulary.

## Findings

1. **A 4-layer cut, trained long enough, is within about a point of EG2 at retrieval.**
   q4, 29.0 MB: hit@10 change against EG2 −0.012, <!-- @evidence[[results/final.json#/L2 mac_k4 q4/synth_test/hit@10_minus_eg2/0]] -->
   agreement 0.630. Two layers, 21.8 MB: hit@10 0.897, <!-- @evidence[[results/final.json#/L2 mac_k2 q4/synth_test/hit@10/0]] -->
   agreement 0.577. <!-- @evidence[[results/final.json#/L2 mac_k2 q4/real_test/overlap@10/0]] -->
   #102's best unrelated student, 24.4 MB: 0.794 and 0.482. (Round 6)
2. **The elbow is at k = 4.** Paired, 10 epochs each: k = 4 over k = 2 +0.053 real, <!-- @evidence[[results/pairs.json#mac_k4_vs_k2_10ep.real_test.0]] -->
   +0.011 hit@10; <!-- @evidence[[results/pairs.json#mac_k4_vs_k2_10ep.hit@10.0]] -->
   k = 8 over k = 4 +0.020 real for 15 MB more, hit@10 +0.004 [−0.003, +0.011]. <!-- @evidence[[results/pairs.json#mac_k8_vs_k4_10ep.hit@10.0]] -->
   (Round 6, `run/pairs.py`)
3. **Untrained, the cut is useless.** Top-10 agreement is 0.014 or less for every k up to 8,
   0.411 at k = 20, <!-- @evidence[[results/l0.json#/20/real_test/overlap@10/0]] -->
   and 1.000 only at k = 24. EG2's output space forms in its last layers; the training
   does the work. (Round 1)
4. **Epochs matter more than the first runs showed.** At k = 4, 10 epochs beat 2 by
   +0.042 real, <!-- @evidence[[results/pairs.json#mac_k4_10ep_vs_cpu_2ep.real_test.0]] -->
   and the curve is flat by epoch 8 (0.631, then 0.632). <!-- @evidence[[results/l1_mac_k4_v32k.json#/evals/step20576/real_test/overlap@10/0]] -->
   After one epoch each, two epochs at k = 4 had tied one at k = 8; with 10 each, depth
   pays again (finding 2). (Round 3, Round 6)
5. **Full EG2 holds to 3 bits only with a rotated codebook and calibration.** All 24 layers,
   32K vocabulary: GPTQ rounding onto a randomized-Hadamard + Lloyd-Max grid, 75.2 MB,
   hit@10 change −0.004; <!-- @evidence[[results/quant_full.json#/gq:3:32/synth_test/hit@10_minus_eg2/0]] -->
   plain 3-bit GPTQ 0.742 agreement at 85.3 MB; <!-- @evidence[[results/quant_full.json#/gptq:3:32/real_test/overlap@10/0]] -->
   2-bit round-to-nearest collapses to 0.041. <!-- @evidence[[results/quant_full.json#/rtn:2:32/real_test/overlap@10/0]] -->
   At 2 bits the best is 0.591 at 58.1 MB, <!-- @evidence[[results/quant_full.json#/gq:2:32/real_test/overlap@10/0]] -->
   no better than the 29 MB cut. Simulated; the load-time decoder is not built. (Round 5,
   `run/quant_full.py`)
6. **The remaining error is structured.** The k = 4 cut after two epochs in fp32 reaches
   cosine 0.948 to EG2 <!-- @evidence[[results/final.json#/L1 k4_ep2 step2572/real_test/cos_to_eg2]] -->
   and 0.608 agreement; <!-- @evidence[[results/final.json#/L1 k4_ep2 step2572/real_test/overlap@10/0]] -->
   in #102, isotropic noise at cosine 0.958 kept 0.81. (Round 3)
7. **Training on the cut vocabulary wins back part of what the cut costs.** Cutting after
   training costs 0.028 real (0.608 to 0.580); <!-- @evidence[[results/final.json#/L2 k4_ep2 q4/real_test/overlap@10/0]] -->
   training the second epoch on the cut tokenizer: +0.008 [+0.002, +0.015]. <!-- @evidence[[results/pairs.json#train_on_cut_vs_cut_after.real_test.0]] -->
   (Round 3)
8. **The bytes are the token table plus 3.4 MB a layer, and q4 beats int8.** At q4 the
   41,741-row table is 13.69 MB, <!-- @evidence[[results/sizes.json#/l2_k2/embed_tokens/mb]] -->
   four layers 13.48 MB, <!-- @evidence[[results/sizes.json#/l2_k4/layers/mb]] -->
   eight 27.64 MB. <!-- @evidence[[results/sizes.json#/l2_mac_k8/layers/mb]] -->
   For a student started on EG2's q4 grid, int8 is larger and further from fp32
   (k = 2: 33.7 MB <!-- @evidence[[results/final.json#/L2 k2_ep2_v32k int8/mb]] -->
   at minimum cosine 0.987 <!-- @evidence[[results/export_l2_k2_ep2_v32k.json#formats.int8.cos_to_pytorch.min]] -->
   against q4's 0.992). (`run/sizes.py`)
9. **The token table goes to 3 bits for free and 2 bits for a few tenths of a point.** k = 4:
   a 3-bit table saves 3.7 MB at hit@10 0.9065 against q4's 0.9072; <!-- @evidence[[results/final.json#/L2 mac_k4 t3w/synth_test/hit@10/0]] -->
   2-bit, 22.7 MB, <!-- @evidence[[results/final.json#/L2 mac_k4 t2w/mb]] -->
   hit@10 0.904 and agreement 0.606. <!-- @evidence[[results/final.json#/L2 mac_k4 t2w/real_test/overlap@10/0]] -->
   The `embeddinggemma-vocab-prune` session measured the same (0.906 and 0.903) and found
   training with the quantized table no better than quantizing after. (Round 8)
10. **A standard-op table unpack makes the file run on wasm.** onnxruntime-web 1.30's wasm
   backend has no GatherBlockQuantized kernel, the op EG2's own q4 graph uses. The same
   bytes unpacked with Gather/BitwiseAnd/BitShift/Cast/Sub/Mul (`model_q4w.onnx`) run on
   wasm with minimum cosine 0.9999999 to CPU, <!-- @evidence[[results/smoke_web_l2_mac_k4_model_q4w.json#wasm.cos_to_cpu_min]] -->
   and on WebGPU. The 2- and 3-bit tables use the same approach. (Rounds 7–8, `run/export.py`)
11. **The bf16 release is not the teacher.** EG2's bf16 weights agree with the q4 teacher
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

**Mac training (Round 6).** `mac_run.sh` on Oskar's 128 GB M5 (PyTorch MPS): k = 2, 4, 8,
each 10 epochs (25,720 steps) from the teacher's weights, trained on the 32K cut tokenizer
from step 0 with the cut table trainable, lr 3e-3 with linear decay, eval every epoch. A k = 4
step took 0.12 s against about 1.7 s on 4 vCPUs. Checkpoints came back as release
`eg2-layer-mac-20261020` and were exported and scored here.

**Full-EG2 layer quantization (Round 5).** `run/quant_full.py`: all 24 layers, the 32K cut,
the table kept at the teacher's q4. Every Linear in the layers, the PLE projection and the
output projection is re-quantized blockwise along its input dimension and simulated in fp32:
round-to-nearest (rtn), GPTQ (Hessians from 2,048 training queries, unquantized
activations), HIGGS-style (seeded randomized Hadamard rotation, Lloyd-Max levels of N(0,1),
per-block fp16 scale chosen by MSE) and gq (GPTQ rounding onto the higgs grid in the rotated
basis). Bytes: ORT MatMulNBits layout for rtn/gptq; b-bit codes plus fp16 scales for
higgs/gq, a download format that needs a load-time decoder.

**wasm variant (Round 7).** `run/export.py --wasm` rewrites the q4 file's GatherBlockQuantized
into standard ops on the same bytes (`model_q4w.onnx`); `run/smoke_web.py --file` runs it in
onnxruntime-web 1.30 (headless Chromium) on wasm and SwiftShader WebGPU against
onnxruntime CPU on 32 queries.

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

### Round 5: full EG2 below 4 bits (2026-10-20)

Asked by Oskar mid-run: could EG2 itself be quantized further instead of cut? Real overlap,
hit@10 and download, 32K vocabulary, table at q4:

| method | 4-bit | 3-bit, block 32 | 3-bit, block 64 | 2-bit, block 32 | 2-bit, block 16 |
|---|---|---|---|---|---|
| none (q4 teacher) | 0.932 / 0.917 / 103.0 MB | | | | |
| rtn | 0.928 / 0.916 | 0.642 / 0.903 / 85.3 | | 0.041 / 0.163 / 67.7 | 0.163 / 0.593 / 85.9 |
| gptq | | 0.742 / 0.910 / 85.3 | | 0.390 / 0.792 / 67.7 | 0.509 / 0.872 / 85.9 |
| higgs | | 0.690 / 0.908 / 75.2 | 0.679 / 0.910 / 70.9 | 0.176 / 0.658 / 58.1 | 0.340 / 0.784 / 66.7 |
| gq | | 0.780 / 0.915 / 75.2 | 0.770 / 0.909 / 70.9 | 0.591 / 0.886 / 58.1 | 0.610 / 0.886 / 66.7 |

The vector-quantization result from `remex-vs-higgs-ablation` (a Gaussian-optimal codebook
matters most at 2–3 bits) carries to weights; calibration matters more, and the two
combine. On a synthetic layer before the run, output error at 2 bits was 0.155 (GPTQ),
0.103 (higgs) and 0.083 (gq). A second session (`embeddinggemma-vocab-prune`) measured the
token table separately: it holds to 4 bits and loses agreement below.

### Round 6: 10 epochs on the Mac (2026-10-20 to 10-21)

Real overlap by epoch (fp32 PyTorch, cut vocabulary):

| k | 2 | 4 | 6 | 8 | 10 | hit@10 at 10 |
|---|---|---|---|---|---|---|
| 2 | 0.505 | 0.548 | 0.574 | 0.576 | 0.581 | 0.897 |
| 4 | 0.563 | 0.594 | 0.619 | 0.631 | 0.632 | 0.907 |
| 8 | 0.569 | 0.609 | 0.641 | 0.653 | 0.656 | 0.911 |

Shipped q4 (`run/export.py`):

| build | download | real | synthetic | hit@10 | vs EG2 |
|---|---|---|---|---|---|
| k = 2 | 21.8 MB | 0.577 | 0.688 | 0.897 | −0.023 |
| k = 4 | 29.0 MB | 0.630 | 0.721 | 0.907 | −0.012 |
| k = 8 | 44.1 MB | 0.650 | 0.746 | 0.911 | −0.008 |

Prediction given to Oskar after epoch 2 at k = 4: 0.62–0.66 final; measured 0.632. At epoch
2 the k = 4 run was behind the 2-epoch CPU run (0.563 against 0.591) because its learning
rate was still near peak; it passed it by epoch 4.

### Round 7: wasm (2026-10-21)

The embed-coord board (#104) pointed out that only GatherBlockQuantized lacks a wasm kernel.
`model_q4w.onnx` unpacks the table with standard ops: same size, cosine to the q4 file
≥ 0.9999998 on the test sets, and on onnxruntime-web 1.30 wasm and WebGPU both k = 4 and
k = 8 match CPU at minimum cosine 0.9999999. Per-query hit arrays for the board are in
`results/*_perquery.json`.

### Round 8: 3- and 2-bit token tables (2026-10-21)

The vocab session found a 3-bit table free on hit@10 for these students. `run/export.py
--table-bits B` quantizes each Mac student's trained table (asymmetric, groups of 32, fp16
scales, uint8 zero points), packs 8 values into B bytes, and unpacks with Gather, Cast,
Mul, ReduceSum, BitShift, BitwiseAnd, Sub and Mul. Layers stay MatMulNBits q4.

| build | download | real | synthetic | hit@10 |
|---|---|---|---|---|
| k = 2, 3-bit table | 18.17 MB | 0.574 | 0.682 | 0.898 |
| k = 2, 2-bit table | 15.50 MB | 0.552 | 0.661 | 0.890 |
| k = 4, 3-bit table | 25.35 MB | 0.623 | 0.717 | 0.907 |
| k = 4, 2-bit table | 22.68 MB | 0.606 | 0.699 | 0.904 |
| k = 8, 3-bit table | 40.39 MB | 0.647 | 0.744 | 0.911 |
| k = 8, 2-bit table | 37.72 MB | 0.643 | 0.727 | 0.909 |

Predicted from fp16 scales before the run: k = 4 at 3 bits about 25.3 MB; measured 25.35.
The first WebGPU run failed: ORT's WebGPU Gather on an fp16 tensor needs shader-f16, which
SwiftShader lacks (ERRORS 10). Casting the scale table to fp32 before the Gather fixed it;
every build tested (k = 2 t2/t3, k = 4 t2/t3, k = 8 t3) then matched CPU on wasm and WebGPU at
minimum cosine 0.9999999.

