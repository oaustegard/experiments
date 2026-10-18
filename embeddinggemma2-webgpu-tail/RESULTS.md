# RESULTS: EmbeddingGemma 2 tail on WebGPU

Issue: [experiments#100](https://github.com/oaustegard/experiments/issues/100).
Contract in [`SPEC.md`](SPEC.md); regenerate-and-check commands in
[`README.md`](README.md). Result files are in `results/`.

## Answer

Can everything after EmbeddingGemma 2's transformer (normalize, centre, rotate,
2-bit remex codes, packing, the ADC scan) run as WebGPU kernels on
onnxruntime-web's own device, so an embedding never leaves the GPU before it is
a packed code? Yes, and the codes agree with remex. With the real
`model_q4.onnx` in headless Chromium, browser embeddings match a CPU
onnxruntime run of the same graph at cosine 1.0 <!-- @evidence[[results/demo_real_run1.json#reference.cos_to_cpu.0]] -->
and reproduce the model card's example score of 0.8542 <!-- @evidence[[results/demo_real_run1.json#reference.browser_scores.1]] -->
(card: 0.854). The startup gate saw 0 code mismatches <!-- @evidence[[results/demo_real_run1.json#startup.gate.codes.mismatch]] -->
against the CPU reference over 32 real documents. On 4,096 real EG2 Quora
vectors the kernels flip 6 <!-- @evidence[[results/check_gpu_eg2.json#codes.mismatch]] -->
of 3.1M codes, each within 1e-5 of a boundary, and scores agree with a float64
recomputation to 0.00000034. <!-- @evidence[[results/check_gpu_eg2.json#scores_vs_own_codes.max_abs_err]] -->

Every timing here is SwiftShader, a software rasterizer, so no number says
whether the GPU tail is faster than a readback and a CPU encode. A run on a real
GPU would answer that; the bench in `demo.html` is ready for one.

## Findings

1. The real EG2 q4 graph matches every guess in `demo.js`: inputs `input_ids`
   and `attention_mask` (int64), `image/video/audio_features` that accept an
   empty `[0, 512]` float32, outputs `last_hidden_state` and a unit
   `sentence_embedding`, pad id 0. Browser embeddings sit at cosine 1.0 to the
   CPU run. <!-- @evidence[[results/demo_real_run1.json#reference.cos_to_cpu.4]] --> (`out/demo/real.json`, `run/ref_real.py`)
2. Real EG2 vectors pass the same checker as the synthetic fixtures: 6 flips,
   <!-- @evidence[[results/check_gpu_eg2.json#codes.excused]] -->
   all excused. The real corpus is less anisotropic than the synthetic stand-in,
   ‖mean‖ ratio 0.788 <!-- @evidence[[results/fixtures_eg2_params.json#stats.mean_norm_ratio]] -->
   and pair cosine 0.622 <!-- @evidence[[results/fixtures_eg2_params.json#stats.pair_cos]] -->
   against the synthetic set's 0.852 and 0.724. (`make_fixtures.py --vectors`)
3. The scan bench's 0.005 disagreement came from the encode. Scored on the
   GPU's own codes, the scan agrees with `searchRef` to 0.00000072. <!-- @evidence[[results/demo_dummy.json#bench.scan_agreement_max_abs_err]] -->
   Against a float64 CPU encode, 3 <!-- @evidence[[results/demo_dummy.json#bench.scan_encode_vs_cpu.code_flips]] -->
   of 7.68M codes flip; rows holding a flip differ by up to 0.0050 <!-- @evidence[[results/demo_dummy.json#bench.scan_encode_vs_cpu.score_max_abs_diff_rows_with_flip]] -->
   and the other rows by at most 0.00000083. <!-- @evidence[[results/demo_dummy.json#bench.scan_encode_vs_cpu.score_max_abs_diff_rows_without_flip]] -->
   One flip moves a score by `scale · 0.033 · qr_j`, as the handoff predicted.
4. On the 150-document demo corpus, the GPU 2-bit ranking finds 90% <!-- @evidence[[results/demo_real_run1.json#search.recall_at_10]] -->
   of the fp32-cosine top 10 for the test query.
5. SwiftShader cost of the real model: one batch of 32 documents takes a median
   137,828 ms to embed. <!-- @evidence[[results/demo_real_run1.json#bench.embed_only_ms.median]] -->
   The tail adds nothing measurable on top; the three bench arms' medians sit
   within 1 s of each other.
   A software-rendered scan of 100k × 20 queries is slower than the CPU
   reference (speedup 0.66). <!-- @evidence[[results/demo_real_run1.json#bench.scan.gpu_over_cpu_scaled_speedup]] -->
   Neither number says anything about a real GPU.

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
  through `console.error`. `demo.js` now sets `ort.env.logLevel = 'error'`.
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
