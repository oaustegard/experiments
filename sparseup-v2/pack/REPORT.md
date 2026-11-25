# SparseUp v2 in the browser: a hand-written WebGPU encoder

`Linkup-Platform/linkup-sparseup-embed-v2` is a learned sparse retriever: ModernBERT-base reads a
text, its masked-language-model head scores all 50,370 vocabulary entries at every position, and
the model keeps each position's 12 strongest terms, max-pools them over the text and merges case
and spacing variants of a word into one term. A query or document comes out as a few dozen to a
few hundred weighted terms, and relevance is a dot product. This directory runs that whole forward
pass in a browser tab with WebGPU compute shaders, from an 8-bit or 4-bit weight pack, and checks it
against PyTorch.

## Answer

The browser encoder reproduces the int8 model exactly. On 47 texts (20 queries, 27 documents, 7 to
512 tokens), the WGSL output matches PyTorch run on the same int8 weights at cosine 1.000000
(worst 0.99999988), with the largest weight difference 2.4e-5. <!-- @evidence[[results/summary.json#int8.kernel_vs_dequant_max_abs]] -->
All remaining difference from fp32 is quantization: int8 vs fp32 cosine min 0.9931, median 0.9992, <!-- @evidence[[results/summary.json#int8.vs_fp32_cos_min]] -->
top-32 term overlap 0.971, about the gap between the model's own bf16 default and fp32 (min 0.9968).

| pack | raw | brotli | SciFact nDCG@10 vs fp32 0.7476 | Mac GPU, 128 / 512 tokens |
|---|---|---|---|---|
| int8 | 154.5 MB | 137.5 MB | −0.0027 [−0.0068, +0.0015] | 8.6 / 34.4 ms |
| q4 (attention + MLP-up 4-bit) | 121.4 MB | 105.0 MB | +0.0024 [−0.0073, +0.0123] | 9.3 / 36.1 ms |

Neither SciFact change is distinguishable from zero. q4 drifts further from fp32 (cosine
median 0.982), saves 33 MB and is no faster. Under SwiftShader (WebGPU emulated on 4
CPU cores, not a GPU number) a text costs about 0.5 s per token: 72 s at 137 tokens, 141 s at
279. Use int8.

## Findings

1. **The WGSL kernels add no measurable error.** Engine vs PyTorch on the same dequantized weights,
   47 texts: cosine min 0.99999988 for int8 and q4, max |difference| 2.4e-5 (int8) and 1.9e-5 (q4)
   (`results/parity_int8_mac.json`, `results/parity_q4_mac.json`, Mac Chrome 157, Apple Metal).
2. **int8 costs as much fidelity as bf16 does.** int8 vs fp32: cosine min 0.9931, median 0.9992,
   top-32 Jaccard mean 0.971. bf16 (the model card's default dtype) vs fp32: min 0.9968, median
   0.9998, Jaccard 0.975. <!-- @evidence[[results/summary.json#bf16.vs_fp32_cos_median]] --> Weight differences up to 1.6 come from terms crossing a position's
   top-12 cutoff, which is a step function.
3. **Retrieval does not move at int8 or q4.** SciFact (300 test queries, 5,183 documents, 512-token
   cap): fp32 0.7476, int8 −0.0027 [−0.0068, +0.0015], q4 +0.0024 [−0.0073, +0.0123], bf16 −0.0014
   (`results/scifact.json`, paired bootstrap, 10k resamples). q4 keeps 90% of fp32's top-10 and int8 97%. <!-- @evidence[[results/summary.json#q4.scifact_top10_overlap]] -->
   The sibling compression study measured the same int8 delta independently (−0.00267,
   `../remex/results/weightquant.json`) and an all-4-bit variant at −0.0009.
4. **Speed on a laptop GPU: 34 ms for a full 512-token document.** Mac, Chrome 157, Apple Metal,
   median of 20: 8.6 ms at 128 tokens, 34.4 ms at 512, 13.3 ms for 16 queries (221 tokens) in one
   batch. Engine build 2.7 s after the 151 MB download. q4 is 5% slower: the 4-bit matmul decodes
   more per weight and this GPU is not bandwidth-bound at these sizes.
5. **The reference is the model's own code.** `modeling_splade.py`, run sandboxed, agrees with
   `ref.py` on all 47 texts: token ids and pooling masks identical, 38 vectors bit-identical, <!-- @evidence[[results/summary.json#official.identical_texts]] --> max
   |difference| 1.1e-5, identical active terms (`results/official_vs_ref.json`).
6. **Compression barely helps the download.** int8 codes are close to incompressible: brotli
   saves 11% on int8 and 13% on q4 (`results/sizes.json`).

## Method

- **Pack** (`pack.py`): the safetensors converted into `weights.bin` + `manifest.json`. int8:
  every linear layer, the token table and the tied decoder symmetric per output row (scale =
  max|row| / 127). q4: attention and MLP-up layers in onnxruntime's MatMulNBits layout, symmetric,
  group 32; MLP-down (K = 1152 does not fit the shared q4 kernel's K % 256), head and token table
  stay int8. Norms, decoder bias and the fold index are f32 / u32. Two regions keep each GPU
  binding under the default 128 MiB limit.
- **Engine** (`web/engine.js`, `web/sparseup.js`): a `SeqEngine` subclass on the shared kernels in
  `_lib/wgsl/`. Token-table gather, LayerNorm, int8 / q4 matmuls, RoPE with both thetas (160,000
  global, 10,000 local), `attentionLong` with the ±64 sliding window on two of every three layers,
  GELU GLU, the MLM head (dense + GELU fused into the matmul epilogue, LayerNorm), then the
  decoder: an int8 matmul over the token table's own bytes into a [tokens × 50,400] logit buffer.
  A new `splade` kernel runs one workgroup per token: it computes w = log1p(relu(logit − 18)) on
  the fly, keeps each thread's top 12 in registers, finds the row's exact 12th-largest w by radix
  select over the candidates' float bits (ties kept, as in the reference), and atomically maxes every
  w at or above it into the output at its folded vocabulary slot. Max-pooling and folding are both
  maxima, so one atomicMax does both; the prefix token ("[Q]" / "[D]") is attended but skipped.
- **Reference** (`ref.py`): transformers' built-in `ModernBertForMaskedLM` on the checkpoint plus
  the SPLADE head; arms fp32, bf16, and fp32 compute on weights read back from each pack
  (`fixture.py`). PyTorch emulation, not NumPy: same arithmetic, faster on 4 cores.
- **Fixture**: 8 hand-written queries, 12 SciFact test queries, 20 SciFact abstracts sampled across
  four length bands, 6 three-abstract concatenations truncated to 512 tokens, one title.
- **Metrics** (`check.py`): per text, cosine of the full folded vectors, Jaccard of the top-32
  term sets, max |weight difference|.
- **Where it ran**: parity and timing on Oskar's Mac through the GPU Bridge (`bridge/job.js`,
  jobs `sparseup-int8-1` and `sparseup-q4-1`); the same page code under SwiftShader in headless
  Chromium here (`check.py`, `web/check.html`, which also checks the JavaScript tokenizer against
  the Python one); SciFact on Kaggle's two T4s (`kaggle_job.py`), with the CPU and T4 fp32
  encodings agreeing at cosine ≥ 0.9999998 (`results/scifact_cpu_vs_t4_fp32.json`).

### Reused and new files

Reused unchanged: `_lib/wgsl/runtime.js` (`SeqEngine`, uploads, batching) and every kernel in
`_lib/wgsl/kernels.js` used here (embed t8, norm, matmul i8 / q4, rope, attentionLong, gated);
`embeddinggemma2-webgpu-tail/run/run_demo.py` (Chromium flags, CDN cache); the GPU Bridge.
Changed: `_lib/wgsl/kernels.js` norm accepts D up to 1,024 (was 512), with a new LayerNorm-768 op
case (rel. error 1.2e-7). New: everything in this directory: the `clear` and `splade` kernels,
the two-region pack format, the decoder-as-int8-matmul wiring, the pooling flag in the token
buffer, the tokenizer prefix rule, the Node/Dawn runner and the bridge job.

### Run it

```bash
python3 pack.py --hf hf --out packs/int8 --quant int8       # hf/ = the Hub snapshot
python3 -m http.server                                       # at the experiments repo root
# open /sparseup-v2/pack/web/ (demo) or /sparseup-v2/pack/web/check.html?pack=int8&reps=10 (timing)
```

`bridge/job.js` is the same timing as a GPU Bridge job. Weights are not hosted anywhere; the
page loads a local pack.

### What did not work

- **No real-GPU WebGPU on Kaggle.** Its containers get NVIDIA compute and utility capabilities,
  no graphics, so no Vulkan driver loads: headless Chromium offers only SwiftShader, and Dawn's Node
  bindings (`run_node.mjs`) find no adapter. Kaggle ran the PyTorch SciFact encodes instead.
- **SwiftShader is too slow for a full fixture in one job.** About 0.5 s per token, so 47 texts and
  8,309 tokens take close to two hours; `check.py` runs in resumable chunks (`--start`, `--only`).
- **The model's code first could not run.** Loading it with `trust_remote_code` was refused in this
  session's automatic permission mode. Oskar asked for it explicitly afterwards, and it ran in the
  workspace sandbox (`run_official.py`: no network, unprivileged user, no secrets).
- **q4 is not worth it here.** It trades 33 MB for noticeably noisier vectors and no speed.

### Delegation

Three jobs went to cheaper models (`DELEGATION.md`): the converter and the SciFact script to
gemini-4-hydrogen ($0.061 and $0.046, both accepted with trivial edits) and the demo page to a
Claude Sonnet subagent (accepted, wording revised). Kernel design, the reference and every check
stayed in this session.

## Log

### 2026-11-25

Built the pack, engine, reference and harness; parity first on SwiftShader (3 texts, exact), then
all 47 texts on the Mac through the GPU Bridge in 4.6 s (int8) and 2.1 s (q4) of job time. SciFact
on Kaggle T4s in 11 minutes for four variants. The local SwiftShader run of all 47 texts and its
128 / 512-token timing was still running when this was written.
