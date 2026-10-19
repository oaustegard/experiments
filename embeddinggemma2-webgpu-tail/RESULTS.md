# RESULTS: EmbeddingGemma 2 tail on WebGPU

Issue: [experiments#100](https://github.com/oaustegard/experiments/issues/100).
Contract in [`SPEC.md`](SPEC.md); regenerate-and-check commands in
[`README.md`](README.md). Result files are in `results/`.

## Answer

Can everything after EmbeddingGemma 2's transformer (rotation, 2-bit remex
codes, packing, the ADC scan, top-k) run as WebGPU kernels on
onnxruntime-web's own device, so an embedding never leaves the GPU before it is
a packed code? Yes, and it is cheap next to the model. On a Mac's Metal GPU in
Chrome, embedding 32 documents with the real q4 model takes 549.5 ms; <!-- @evidence[[results/demo_bench_apple_metal3.json#embed_only_ms.median]] -->
encoding them on the GPU adds 5.0 ms, <!-- @evidence[[results/demo_bench_apple_metal3.json#derived.gpu_encode_added_ms]] -->
and reading them back for the JS reference encode adds 45.6 ms. <!-- @evidence[[results/demo_bench_apple_metal3.json#derived.readback_encodeRef_added_ms]] -->
The scan of 100k codes × 20 queries takes 13.1 ms, <!-- @evidence[[results/bench_tail_apple_metal3.json#scan_ms.median]] -->
and encoding 100k vectors with remex's Hadamard rotation 38.3 ms. <!-- @evidence[[results/bench_tail_apple_metal3_rht.json#encode_full_ms]] -->
The codes agree with remex: 2 of 3.1M flip on real EG2 vectors, all at a boundary.

The JS reference is slow on purpose (float64, dense), so the 45.6 ms
overstates what readback costs; a native-speed CPU encode of 32 vectors would
take under 1 ms. What the GPU path buys is the scan, about 580× remex's
codes-only ADC search on a 4-core Xeon, and no embedding ever crossing to the
host. A second GPU would change these numbers most.

## Findings

1. Real GPU, real model (Oskar's Mac, Metal 3, Chrome 154; `demo.html`
   bench): the three arms take 549.5, 554.5 <!-- @evidence[[results/demo_bench_apple_metal3.json#embed_gpu_encode_ms.median]] -->
   and 595.1 ms <!-- @evidence[[results/demo_bench_apple_metal3.json#embed_readback_encodeRef_ms.median]] -->
   per 32 documents (embed only, + GPU encode, + readback and JS encode). The
   scan agreement holds on Metal: 1 <!-- @evidence[[results/demo_bench_apple_metal3.json#scan_encode_vs_cpu.code_flips]] -->
   of 7.68M codes flips against the float64 CPU encode, and the scan agrees with
   `searchRef` on the GPU's own codes to 0.00000077. <!-- @evidence[[results/demo_bench_apple_metal3.json#scan_agreement_max_abs_err]] -->
2. The tail on Metal against native CPU code (different machines: the Mac
   against this 4-core Xeon container, `run/native_bench.py`): encode 100k in
   203.4 ms against remex's 1,919 ms; <!-- @evidence[[results/native_bench.json#encode_ms.median]] -->
   scan 100k × 20 in 13.1 ms against remex `search_adc` at 7,544 ms, <!-- @evidence[[results/native_bench.json#scan_search_adc_ms.median]] -->
   remex `search_batch` at 43 ms (which scores a cached 300 MB dequantized
   float matrix) <!-- @evidence[[results/native_bench.json#scan_search_batch_ms.median]] -->
   and an fp32 matrix product over the raw vectors at 35 ms. <!-- @evidence[[results/native_bench.json#scan_fp32_matmul_ms.median]] -->
   Native onnxruntime embeds 32 documents in 817 ms on the Xeon. <!-- @evidence[[results/native_bench.json#embed_ms_all_cores.median]] -->
3. Encode tiling (4 vectors per workgroup in `vec4` lanes) is bit-identical to
   the one-vector kernel on both fixture sets and 3.2× faster on Metal: 16,384
   vectors in 33 ms <!-- @evidence[[results/bench_tail_apple_metal3.json#encode_tiled_ms.median]] -->
   against 104.3 ms. <!-- @evidence[[results/bench_tail_apple_metal3.json#encode_untiled_ms.median]] -->
   The 16 KB default workgroup-memory limit caps the tile at 4, and ORT creates
   the device, so a larger limit cannot be requested.
4. GPU top-k (bitonic sort of 1024-score chunks, repeated passes) beats readback
   plus host select on Metal: search plus top-10 takes 18.8 ms <!-- @evidence[[results/bench_tail_apple_metal3.json#topk_gpu_ms.median]] -->
   against 23.5 ms. <!-- @evidence[[results/bench_tail_apple_metal3.json#topk_host_ms.median]] -->
   It matches the host top-k on 10 edge cases with 0 mismatches each, <!-- @evidence[[results/topk_edge.json#cases.9.mismatches]] -->
   including duplicated rows, n < k and a three-pass n, and goes red on exactly
   the four duplicate cases when the tie rule is flipped.
5. The real EG2 q4 graph matches every guess in `demo.js`: inputs `input_ids`
   and `attention_mask` (int64), `image/video/audio_features` that accept an
   empty `[0, 512]` float32, outputs `last_hidden_state` and a unit
   `sentence_embedding`, pad id 0. Browser embeddings sit at cosine 1.0 to CPU
   onnxruntime <!-- @evidence[[results/demo_real_run1.json#reference.cos_to_cpu.0]] -->
   and reproduce the model card's score of 0.8542. <!-- @evidence[[results/demo_real_run1.json#reference.browser_scores.1]] -->
   (`run/ref_real.py`)
6. Real EG2 vectors pass the same checker as the synthetic fixtures: 6 flips,
   <!-- @evidence[[results/check_gpu_eg2.json#codes.excused]] -->
   all excused. The real corpus is less anisotropic than the synthetic stand-in,
   ‖mean‖ ratio 0.788 <!-- @evidence[[results/fixtures_eg2_params.json#stats.mean_norm_ratio]] -->
   and pair cosine 0.622 <!-- @evidence[[results/fixtures_eg2_params.json#stats.pair_cos]] -->
   against the synthetic set's 0.852 and 0.724. (`make_fixtures.py --vectors`)
7. The scan bench's 0.005 disagreement came from the encode. Scored on the
   GPU's own codes, the scan agrees with `searchRef` to 0.00000072. <!-- @evidence[[results/demo_dummy.json#bench.scan_agreement_max_abs_err]] -->
   Against a float64 CPU encode, 3 <!-- @evidence[[results/demo_dummy.json#bench.scan_encode_vs_cpu.code_flips]] -->
   of 7.68M codes flip; rows holding a flip differ by up to 0.0050 <!-- @evidence[[results/demo_dummy.json#bench.scan_encode_vs_cpu.score_max_abs_diff_rows_with_flip]] -->
   and the other rows by at most 0.00000083. <!-- @evidence[[results/demo_dummy.json#bench.scan_encode_vs_cpu.score_max_abs_diff_rows_without_flip]] -->
   One flip moves a score by `scale · 0.033 · qr_j`, as the handoff predicted.
8. On the 150-document demo corpus, the GPU 2-bit ranking finds 90% <!-- @evidence[[results/demo_real_run1.json#search.recall_at_10]] -->
   of the fp32-cosine top 10 for the test query.
9. The randomized Hadamard rotation (remex's own transform, in WGSL) encodes
   16,384 vectors on Metal in 9.1 ms, <!-- @evidence[[results/bench_tail_apple_metal3_rht.json#encode_tiled_ms.median]] -->
   against 33 ms for the dense tiled kernel <!-- @evidence[[results/bench_tail_apple_metal3_rht.json#encode_tiled_dense_ms.median]] -->
   and 95.7 ms untiled; <!-- @evidence[[results/bench_tail_apple_metal3_rht.json#encode_untiled_ms.median]] -->
   100k take 38.3 ms, 50× remex's native encode on the Xeon. It also passes
   the checker with fewer flips than the dense product: 2 <!-- @evidence[[results/check_gpu_eg2_rht.json#codes.mismatch]] -->
   of 3.1M codes on real EG2 vectors (dense: 6) and 1 <!-- @evidence[[results/check_gpu_synthetic_rht.json#codes.mismatch]] -->
   on the synthetic set (dense: 2), because in float32 it rotates bit-identically
   to remex's operator. A one-round mutant fails 2.27M codes.
10. Truncating to 512 dims is not free for nearest-neighbour recall, though it
    was close to free for nDCG in `embeddinggemma2-remex-quant`. On 100k Quora
    vectors and 1,000 queries, recall@10 against exact 768-dim search is 0.915 <!-- @evidence[[results/dim512.json#recall_at_10_vs_fp32_768.rx2_768]] -->
    for 2-bit codes at d=768 (192 B) and 0.874 <!-- @evidence[[results/dim512.json#recall_at_10_vs_fp32_768.rx2_512]] -->
    at d=512 (128 B), a difference of −0.041 <!-- @evidence[[results/dim512.json#rx2_512_minus_rx2_768_vs_fp32_768.mean]] -->
    (95% CI −0.048 to −0.035). Truncation alone, in fp32, keeps 0.926 <!-- @evidence[[results/dim512.json#recall_at_10_vs_fp32_768.fp32_512]] -->
    of the neighbours; the two code widths share 0.855 of their top 10. <!-- @evidence[[results/dim512.json#overlap_rx2_512_with_rx2_768]] -->
    (`run/eval_512.py`)
11. SwiftShader is no stand-in for a GPU: on it the real model takes 137,828 ms <!-- @evidence[[results/demo_real_run1.json#bench.embed_only_ms.median]] -->
   per 32 documents (250× Metal), and the GPU top-k is slower than the host
   path (8,814 ms against 7,031 ms <!-- @evidence[[results/bench_tail_swiftshader.json#topk_gpu_ms.median]] -->
   for search plus top-10), the reverse of Metal.

## Method

The reference is remex 2.1.0, `Quantizer(d=768, bits=2, seed=42, mean=mu)`.
`oracle/make_fixtures.py` writes fixtures from it, either synthetic EG2-shaped
vectors or, with `--vectors`, the first 4,096 docs and 32 queries of the
embstore set `quora/embeddinggemma-2/fp32-768-first100k` with the mean over all
100k docs. `oracle/check_tail.py` compares an implementation's output with
remex (codes, with flips excused within 1e-5 of a boundary; scale, bias, scores,
top-10, pool) and with a float64 recomputation from the implementation's own
codes. The arms are the float32 numpy model, the CPU reference
(`web/tail_ref.js`), the WGSL kernels (`web/tail.js`), and the ORT 1.30.0
gpu-buffer handoff (`interop/`).

`web/demo.html` runs the full pipeline on 150 short documents, with a startup
gate that encodes the first 32 again on the CPU and hides results unless they
agree. `run/run_demo.py` drives it in Chromium with SwiftShader WebGPU, either
with a dummy ONNX graph (`?local=1&model=dummy`) or, with `--real`, the hosted
page and the real `model_q4.onnx`. `run/ref_real.py` runs the same graph on
onnxruntime's CPU provider for the model card's text-search example; `--real`
compares the browser's embeddings with it.

## Log

### 2026-10-18 — round 1: kernels, oracle, demo (Claude app session)

Built `SPEC.md`, the oracle, the WGSL kernels, the CPU reference, the ORT
interop test and the demo page. The checker passed for every arm on synthetic
fixtures (kernels: 2 of 3.1M codes flip, scores within 3.5e-7), and the dummy
smoke run passed. The session could not reach Hugging Face's CDN or the R2
store, so the real model and real vectors were left to the handoff issue, along
with the 0.005 scan-bench disagreement.

### 2026-10-18 — round 2: real model, real vectors, scan check (Claude Code)

Items 1-4 and 7 of the handoff, plus two of the three performance gaps in item 6.

- **Pages assets.** Committed `web/assets/R.f32.txt` (2.36 MB) and listed
  `embeddinggemma2-webgpu-tail/web` in `pages.json`; `web/index.html` redirects
  to `demo.html` because the builder requires an index for a directory.
- **Real graph.** Read the graph's inputs and outputs with `onnx` before
  running it: all as guessed. `@huggingface/tokenizers` 0.2.0 gives the same ids
  as Python `tokenizers`, `<bos>` … `<eos>` included. CPU q4 reproduces the
  card's four scores to 3 places (0.6845 / 0.8542 / 0.7517 / 0.7833). The first
  `--real` attempt served the model through Playwright's `route.fulfill`; with
  `path=` Playwright typed the extensionless cache file as octet-stream and the
  ES module import failed, and with `body=` the 174 MB weights killed the
  browser. The runner now serves the model files from a local mirror via
  `&hf=`, and only jsDelivr goes through the route. The real run passed every
  check except zero console errors: ORT prints its node-placement warnings
  through `console.error`. Setting `ort.env.logLevel = 'error'` did not stop
  them (a second 30-min run); the session option `logSeverityLevel: 3` targets
  the native logger they come from, and with it a `--real --no-bench` run passed
  every check, zero console errors included, in 12 min (`results/demo_real_nobench.json`).
  Wall clock 39 min on SwiftShader.
- **Real vectors.** `make_fixtures.py --vectors DIR`; `test.html?fx=` and
  `run_gpu.py --fixtures/--out` select the fixture set. All arms pass on it.
- **Scan check.** `scanBench` now scores `searchRef` on the GPU's codes, scale
  and bias from `readCorpus`, and reports the encode flips apart. The
  hypothesis in the issue held (finding 3).
- **Performance.** Per-dispatch uniform buffers are destroyed after their
  submit. `topk` does a partial select instead of sorting all n scores; a
  node check against `topkRef` (5 cases, ties included) and the checker's top-10
  both agree. Not done: tiling several vectors per `encode` workgroup so R is
  read once per tile, and a GPU top-k.
- **Not done: real-GPU numbers.** This container has no GPU.

### 2026-10-19 — round 3: encode tiling, GPU top-k, native comparison

Oskar: "do the encode tiling and GPU top-k, also, how does performance compare to
native?"

- **Encode tiling.** `ENCODE` now takes 4 vectors per workgroup in `vec4`
  lanes: R is read once per tile, reductions are `vec4` tree reductions, and
  after the rotation the 12 KB activation buffer is reused for the codes. The
  outputs on both fixture sets are byte-identical to the one-vector kernel
  (`cmp` on every output file). SwiftShader, 16,384 vectors: 39,479 ms → 13,705 ms.
- **GPU top-k.** New `TOPK` kernel and `tail.topk` passes; the old host path is
  `tail.topkHost`. The first build failed to compile: WGSL constant-evaluates
  `bitcast<f32>(0xff800000u)` to -inf and rejects it, and rejects a
  `-3.40282347e38` literal as unrepresentable; `bitcast<f32>(0xff7fffffu)`
  (-FLT_MAX) works. `run/run_topk_edge.py` checks 10 edge cases against
  `topkHost`; flipping the tie rule fails the four duplicate-row cases. The
  checker's top-10 on both fixture sets still matches remex.
- **Native comparison.** `run/native_bench.py` (remex 2.1.0 numpy, onnxruntime
  1.30.0 CPU) and `run/run_bench.py` (`web/bench_tail.html`, SwiftShader) at
  the same sizes. Finding 5 has the numbers. The browser's embed is 169× slower
  than native onnxruntime here; that measures SwiftShader, not WebGPU.
- **Not done:** an RHT rotation kernel to match remex's work per vector, and any
  real-GPU number.

### 2026-10-19 — round 4: real GPU (Oskar's Mac)

Oskar ran both pages from Pages on his Mac (Chrome 154, adapter `apple` /
`metal-3`) and pasted the JSON of a second run each:
`results/bench_tail_apple_metal3.json` and `results/demo_bench_apple_metal3.json`.

| | Metal 3 | SwiftShader | native CPU (Xeon, 4 cores) |
|---|---|---|---|
| embed 32 docs | 549.5 ms | 137,828 ms | 817 ms (onnxruntime) |
| + GPU encode of those 32 | +5.0 ms | +<1 s (noise) | — |
| + readback and JS `encodeRef` | +45.6 ms | — | — |
| encode 16,384, tiled / untiled | 33 / 104.3 ms | 13,705 / 39,479 ms | — |
| encode 100k | 203.4 ms | 81,406 ms | 1,919 ms (remex, RHT) |
| scan 100k × 20 | 13.1 ms | 7,145 ms | 7,544 ms `search_adc`, 43 ms `search_batch` |
| search + top-10, GPU / host | 18.8 / 23.5 ms | 8,814 / 7,031 ms | — |

Predictions from round 3 against these: the tail "adds nothing measurable" to
the embed held (+0.9%); the GPU top-k reversing on real bandwidth held (18.8 vs
23.5 ms); the tiling gain grew slightly (2.9× on SwiftShader, 3.2× on Metal).
The demo's startup gate and the scan agreement passed on Metal. Wall clock: a
few minutes of Oskar's time against about 40 min per SwiftShader run of the
real model.

### 2026-10-19 — round 5: RHT rotation kernel

Oskar: "build the RHT rotation kernel".

remex rotates with `RHTOperator`: for d = 768, block size 256 and 2 rounds; for
r = 1, 0, an unnormalized Walsh-Hadamard transform over each 256-wide block,
then `z[perm_r[i]] = y[i] · sign_r[i] / 16`. A float32 numpy re-implementation
of that reading matched remex's operator bit for bit, and a float64 one matched
the dense R to 2e-15, before any WGSL.

- `tail.js` takes `rotation: 'rht'` (the default when `params.rht` is given)
  or `'dense'`. The RHT variants of ENCODE and PREP run the transform in
  workgroup memory (8 barrier-separated butterfly stages per round, then a
  register-staged signed scatter); the 2.36 MB R is no longer uploaded.
  `setMean` computes Rmu with a host RHT in float64.
- The plan (two permutations, two sign vectors) travels in `params.json`
  (fixtures and `web/assets/`), 25 KB with the rest of the parameters.
- Checks: both fixture sets pass with 1 and 2 flips; `--rot dense` output is
  byte-identical to round 3; interop still makes 0 `getData` calls; the top-k
  edge cases and the dummy demo pass. Mutant (one round instead of two): 2.27M
  unexcused flips, red.
- Timing: no SwiftShader bench this round. Oskar runs `bench_tail.html`, which
  now times the RHT encode, the dense tiled encode and the untiled dense
  kernel side by side.

Oskar's Metal run of the round-5 bench (`results/bench_tail_apple_metal3_rht.json`):

| encode, Metal 3 | 16,384 vectors | 100k vectors |
|---|---|---|
| RHT, tiled | 9.1 ms | 38.3 ms |
| dense, tiled | 33 ms | 203.4 ms (round 4) |
| dense, untiled | 95.7 ms | — |

The 590k → 12k drop in arithmetic per vector gave 3.6×, not 50×. Probably,
not measured: at 9 ms the RHT encode is bound by its 16 barrier-separated
butterfly stages and the scatters rather than by arithmetic. Scan (13.4 ms) and GPU top-k (18.1 ms against
26.7 ms) are unchanged, as expected. Encoding 100k now costs about 3× one scan
of them, against 15× before.

### 2026-10-19 — round 6: d=512

A sibling session relayed Oskar's suggestion to encode at 512 dims (EG2 is
Matryoshka-trained); Oskar: "test 512". Since the RHT encode is now ~0.4 µs per
vector on Metal, 512 would buy storage (128 B against 192 B per vector) and a
scan reading two thirds as much, not visible speed, so it was tested offline
before any kernel work. remex accepts d=512 (one 512-wide block, 2 rounds).

`run/eval_512.py`: 100k EG2 Quora docs, 1,000 test queries, top-10. Docs and
queries truncated to 512 and renormalized; the d=512 quantizer takes the mean of
the truncated docs. remex `search_adc` (codes only, as the GPU scans).

| arm | bytes | recall@10 vs fp32 768 | vs fp32 512 |
|---|---|---|---|
| fp32, d=512 | 2048 | 0.926 | 1 |
| remex 2-bit, d=768 | 192 | 0.915 | — |
| remex 2-bit, d=512 | 128 | 0.874 | 0.893 |

rx2_512 − rx2_768: −0.041 (paired bootstrap 95% CI −0.048 to −0.035); the two
code widths share 0.855 of their top 10. Native encode of 100k: 2.86 s at 768,
1.49 s at 512.

Reading: the earlier "512 is free" result measured nDCG@10 against human
relevance labels on SciFact and NFCorpus, where swapping a few near-tied
neighbours costs nothing. Against the 768-dim model's own neighbours, truncation
moves 7.4% of the top 10 before quantization does anything, and the 2-bit code
on 512 dims adds more error than on 768. Whether that matters for search
quality needs a labelled set, which the 100k prefix lacks (most Quora qrels fall
outside it). No GPU d=512 path built: a third off storage for 4 points of
recall is a trade to make for a specific corpus, not a default.

