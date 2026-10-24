# Refractometer: WebGPU performance cliffs, measured per device

A prism's refractive index is read at its angle of minimum deviation. Refractometer does the
same for GPU kernels: it sweeps the knobs that the EmbeddingGemma 2 kernels in this repo hard-code
(workgroup size, tile shape, workgroup-memory use), finds the minimum-latency point on *this*
device, and marks the cliffs around it. The winners are saved as a JSON profile keyed by the
adapter, and `web/tuned.js` hands them to a kernel at startup.

Page: **https://austegard.com/experiments/refractometer/web/** — results in [`RESULTS.md`](RESULTS.md).

## Why

`eg2-wgsl-engine/web/engine.js` runs its q4 matmul as 32 columns × 8 k-splits × 8 rows
(`WG_MM = 256`), and `embeddinggemma2-webgpu-tail/web/tail.js` scans with 256-thread workgroups and
the 12 KB lookup table in workgroup memory. Those were chosen on one Mac. A laptop GPU and an iPhone
have different subgroup widths, cache sizes and occupancy limits, and the best value moves with
them. METHODS.md already records that SwiftShader timings can reverse a GPU conclusion, so the only
useful measurement is one taken on the device itself.

## Files

| file | role |
|---|---|
| `web/index.html` | the page: adapter report, sweep controls, one chart per sweep, profile copy/download |
| `web/refract.js` | WGSL generators, verification, timing, sweeps, profile format (no DOM; also runs in Node) |
| `web/tuned.js` | `getTuned(adapter, kernelName, fallback, opts)`: the saved config for this GPU |
| `web/profiles.json` | bundled profiles, keyed `vendor/architecture/device` (empty until a run is pasted in) |
| `run/check.mjs` | Node + Dawn (`npm install webgpu`): compile every variant, verify every config, negative controls, both timing paths |
| `run/test_tuned.mjs` | `tuned.js` lookup tests against the lavapipe check's winners |
| `run/summarize.py` | folds the check outputs into `results/summary.json`, which RESULTS.md cites |
| `root-docs.patch` | the root README row and METHODS entries, not yet applied (see RESULTS log) |
| `run/page_check.py` | headless Chromium (SwiftShader) run of the page: autorun, console errors, screenshots |
| `results/` | `lavapipe_check.json`, `swiftshader_page.json`, `summary.json` (profiles and screenshots are regenerated, not committed) |

## Sweeps

| sweep | kernel | knobs | metric |
|---|---|---|---|
| a) bandwidth | vec4 copy; vec4 read + workgroup reduce | buffer size 16 KiB → 128 MiB (up to 256) | GB/s |
| b) reduction | sum of 4M floats, tree in workgroup memory | workgroup size 32–1024 × items per thread 1–16 | GB/s |
| b) q4 GEMV | `eg2-wgsl-engine`'s MatMulNBits matmul, generalized | columns C, k-splits S, rows R, at the encoder's shapes: gate/up 2048×512, q 1024×512, down 512×2048 (M=16 tokens), embedding projection 768×512 (M=1) | GB/s of weights + activations |
| b) scan | the tail's 2-bit `SCAN` | workgroup size, LUT in workgroup memory vs storage, vectors per thread; 65,536 × 8 queries | GB/s of codes |
| c) tiled matmul | f32 128×2048×512 | tile TM×TN, k-step TK, workgroup WX×WY, with and without workgroup memory | GFLOP/s |
| d) subgroups | the reduction | `subgroupAdd` + one slot per subgroup vs the workgroup-memory tree | GB/s |
| e) f16 | the five best matmul tiles from (c) | f32; f16 data + f32 accumulate; f16 accumulate | GFLOP/s |

Every configuration is compiled (`getCompilationInfo` + a validation error scope), dispatched once
on seeded data, and compared element-wise with a JS reference before it is timed. A configuration
that fails is drawn nowhere and can never win. Timing: each sample is one compute pass of `inner`
back-to-back dispatches, calibrated to about 2 ms, timed by `timestamp-query` pass timestamps when the
adapter has them, else submit → `onSubmittedWorkDone`; median and p10/p90 of 25 samples after warmup.

## Prior art (four searches, 2026-10-24)

- **Device ceilings.** [clpeak](https://github.com/espes/clpeak) and vkpeak measure peak bandwidth and
  FLOPS natively; [Web3D Survey's WebGPU benchmark](https://web3dsurvey.com/webgpu/benchmark-report)
  (Ben Houston's webgpu-bench) does it in the browser — bandwidth and fp32/fp16/int8 FLOPS, crowdsourced.
  They report one ceiling per device; they do not sweep a kernel's configuration or keep a winner.
- **In-browser tuning.** [nnJIT](https://arxiv.org/abs/2309.08978) (MobiSys '24) tunes tile sizes and
  thread binding for WebGPU and Wasm on the client, prunes the space to dozens of candidates with
  microbenchmarks and heuristics, and caches results server-side keyed by kernel, shape and device.
  That is this idea at scale, inside a compiler. TVM/Ansor and WebLLM tune offline per target;
  [Burn's autotune](https://burn.dev/blog/autotune-for-gpu-kernels) benchmarks kernel variants at
  runtime on the device and caches the winner — the same loop, for its own kernels.
- **Timing practice.** Chrome quantizes `timestamp-query` results (developer flag to disable;
  [WebGPU developer features](https://developer.chrome.com/docs/web-platform/webgpu/developer-features)),
  and subgroups shipped behind `subgroups` from Chrome 128–134
  ([new in WebGPU 128](https://developer.chrome.com/blog/new-in-webgpu-128)).

**Components exist, this composition did not turn up in four searches**: a page that sweeps *these*
hand-written kernels (the EG2 q4 matmul and the tail's scan) at their real shapes, refuses any
configuration whose output is wrong, marks the cliffs as well as the winner, separates winners that
need raised limits from ones that fit a default device, and hands the result back to the kernels
through a small module. Unlike nnJIT it has no compiler and no search heuristics: the grids are
small enough (656 configurations at lavapipe's limits) to measure exhaustively in a few minutes.

## How eg2-wgsl-engine would consume a profile

`engine.js` builds its matmul shader from the constant `WG_MM = 256` and the literals 32 (columns),
8 (k-splits) and 8 (rows). Make those three numbers template parameters of `SHADERS.matmul` —
`refract.js`'s `WGSL.gemvQ4` is that generalization, with identical arithmetic — and key the
compiled pipeline by `(C, S, R)`. In `Eg2Engine.create`, before compiling, call
``await getTuned(adapter, `gemv_q4:${N}x${K}:m16`, { C: 32, S: 8, R: 8 }, { defaultLimits: true })``
once per weight shape (`defaultLimits: true` because the engine may share onnxruntime-web's device,
which cannot raise limits; with its own device it can drop that and take the overall winner), and
dispatch `[N / C, ceil(rows / R)]` instead of `[N / 32, ceil(rows / 8)]`. With no profile for the
adapter, `getTuned` returns the fallback, which is today's configuration, so behaviour only changes
on a GPU someone has measured. The tail's scan takes `scan_2bit:65536x8` the same way (`wg`, `lut`,
`v`). This experiment does not modify either engine.
