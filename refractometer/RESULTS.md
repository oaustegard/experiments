# RESULTS: Refractometer — per-device WebGPU kernel sweeps

Built 2026-10-24 from a brainstorm idea ("Refractometer", #3). Page:
https://austegard.com/experiments/refractometer/web/

## Answer

Can a page measure, on the device itself, where the EG2 kernels' hard-coded workgroup and tile
choices sit relative to the best and to the cliffs, and hand the winners back to the kernels?
The tool is built and checked, but no GPU numbers exist yet. They come from Oskar running the page
on his laptop and his iPhone.

What was checked here: all 910 WGSL variants the full sweeps can generate compile under Dawn
on lavapipe with no errors. <!-- @evidence[[results/summary.json#lavapipe.kernels_compiled]] -->
All 656 configurations of the full grid were dispatched and matched their JS references.
<!-- @evidence[[results/summary.json#lavapipe.configs_verified]] -->
All 48 deliberately broken kernels were rejected. <!-- @evidence[[results/summary.json#lavapipe.mutants_rejected]] -->
The page ran its quick sweeps end to end in headless Chromium on SwiftShader. 142 configurations were verified and timed there,
<!-- @evidence[[results/summary.json#swiftshader_page.configs_ok]] -->
with 0 console errors <!-- @evidence[[results/summary.json#swiftshader_page.console_errors]] -->
and no horizontal overflow at phone width.

SwiftShader does rank the engine's current GEMV config (C=32, S=8, R=8) 1.9–3.7× slower than the best one.
<!-- @evidence[[results/swiftshader_page.json#best.gemv_q4:2048x512:m16.speedup_vs_engine]] -->
That is a CPU rasterizer talking, and METHODS.md records SwiftShader reversing a GPU conclusion
before. A Metal or Apple-GPU run that puts C=32/S=8/R=8 near the top would settle it the other
way.

## Findings

1. **Every kernel compiles and every config is correct on a real Vulkan driver.** On lavapipe, 910 variants compiled with 0 failures. <!-- @evidence[[results/summary.json#lavapipe.kernels_failed]] -->
   All 656 full-grid configs matched the reference: 378 GEMV, 136 matmul, 36 scan, 36 subgroup, 30 reduce, 28 bandwidth and 12 f16. The check took 39.6 s on 2 cores. <!-- @evidence[[results/summary.json#lavapipe.verify_seconds]] -->
   (`run/check.mjs`, `results/lavapipe_check.json`)
2. **The verification gate catches wrong-but-fast kernels.** Five mutant families were tried: an off-by-one copy, a reduction with one lane poisoned, a GEMV zero-point shift of 1/16, a scan with reversed code order, and a matmul column swap. All 48 mutant configs came back `wrong` and none was timed. <!-- @evidence[[results/summary.json#lavapipe.mutants_rejected]] -->
   An earlier reduction mutant only permuted indices, and it passed. Sums do not care about order, so a mutant has to change values. (`run/check.mjs` negative controls)
3. **In this container, Dawn's queue submissions complete on lavapipe.** The claude.ai container never completes a submission under gVisor. Here a dispatch returned 42, and both the timestamp-query path and the wall-clock path produced timings, so the numerical checks are real here and not limited to compilation. (`results/lavapipe_check.json#timed`)
4. **The page works end to end in a browser.** Headless Chromium with SwiftShader WebGPU ran the 6 sweeps that the adapter supports. SwiftShader has no `shader-f16`, so f16 was skipped. It took 2280.6 s, all on a CPU. <!-- @evidence[[results/summary.json#swiftshader_page.seconds]] -->
   It saved the profile to localStorage, and the profile rendered again at 390 px with 0 px horizontal overflow. <!-- @evidence[[results/summary.json#swiftshader_page.phone_overflow_px]] -->
   (`run/page_check.py`; its screenshots are regenerable and not committed)
5. **`tuned.js` resolves profiles as designed.** The lookup tries an exact adapter key, then vendor/architecture, then the fallback. An unknown kernel returns the fallback unchanged. The lavapipe check's winners are the test fixture. (`run/test_tuned.mjs`)
6. **Not measured: any GPU number.** That covers bandwidth knees, the GEMV winner per shape, the occupancy cliff of the tiled matmul, the subgroup/tree ratio and the f16 speedup. Lavapipe and SwiftShader timings appear only in the profiles, as proof that the timing code runs. (Round 1)

## Method

- **Kernels** (`web/refract.js` `WGSL`): grid-stride vec4 copy; read-reduce; a 4M-float reduction (tree or
  `subgroupAdd`); the `eg2-wgsl-engine` MatMulNBits q4 matmul generalized to C columns × S k-splits ×
  R rows; the tail's 2-bit `SCAN` with workgroup size, LUT placement and vectors per thread as knobs;
  a tiled f32/f16 matmul (TM×TN×TK, WX×WY, with or without workgroup memory). All sizes are baked into
  the WGSL as constants, so there are no uniforms.
- **Shapes**: the 4-layer EG2 cut from `eg2-wgsl-engine/web/manifest.json`. These are q 1024×512, gate/up 2048×512 and down
  512×2048 at M=16 tokens, the 768×512 embedding projection at M=1, and the matmul at 128×2048×512.
  The scan is 65,536 vectors × 8 queries.
- **Verification**: data comes from a seeded mulberry32 generator. Reductions use integers 0–6, so the f32 sums are exact and the comparison
  has tolerance 0 in any summation order. The matmul uses multiples of 1/8, which are exact in f16 and exact in f32 for
  K=512. Its tolerance is 1e-6·Σ|ab| (2e-2·Σ|ab| for f16 accumulation). GEMV: 2e-5·Σ|wx|. Scan: 4e-6·aux·Σ|lut|.
  For copies above 32 MiB, four 1 MiB windows are compared. Every result buffer is cleared before the
  verification dispatch.
- **Timing**: a sample is one compute pass of `inner` identical dispatches. `inner` is calibrated from the
  slope between 1 and 8 dispatches, aiming at about 2 ms per pass (4 ms on wall clock). Then 3 warmup samples, 25 samples, and the
  median with p10/p90 are reported. Timestamps come from `timestampWrites` and are flagged when they arrive quantized to 100 µs. Without the feature, timing is
  `performance.now()` around submit → `onSubmittedWorkDone`.
- **Cliffs**: on the upper envelope (best config per x), a cliff is an x whose best falls below 75% of the best
  at smaller x (80% for bandwidth).
- **Limits**: the device requests the adapter's maximum limits. Configs above the WebGPU defaults are flagged, and
  `default_limits` keeps the best config that fits a default device.

## Log

### Round 1 — 2026-10-24: build and headless validation

Read `CLAUDE.md` and grepped METHODS.md for WebGPU, timing, workgroup and subgroup. `repo-index/ask.py`
needed `remex==0.6.0`, `tokenizers` and `onnxruntime`. It pointed at `embeddinggemma2-webgpu-tail` (bench
page, SwiftShader caveat), `eg2-wgsl-engine` and `py-wasm-bench/arms/wgsl`. Read `engine.js` (the
matmul, `WG_MM = 256`, 32×8×8) and `tail.js` (`SCAN`, 256 threads, 12 KB LUT in workgroup memory).
Prior art came from four searches; see README.

Container: `npm install webgpu` (0.6.2) found no Vulkan driver. After `apt-get install
mesa-vulkan-drivers`, `VK_ICD_FILENAMES=.../lvp_icd.json` gave `mesa/software/llvmpipe` with
timestamp-query, subgroups (size 8) and shader-f16. Dispatches complete, unlike the claude.ai
container.

The first `check.mjs` aborted with `std::system_error: Invalid argument` or "The futex facility
returned an unexpected error code" on its first pipeline. Bisection showed the cause was not
async pipelines and not the sandbox: `openDevice(create([]))` kept no reference to the
`GPU` object, so it was garbage-collected and the next Dawn call died. Any delay before that call
(a 3 s sleep, a busy loop) reproduced it. Fix: `ctx.gpu = gpu`. After that, the async pipeline path works too.

| check | result |
|---|---|
| compile, full grid at lavapipe limits | 910 / 910 |
| verify, full grid | 656 / 656 (bandwidth 28, reduce 30, gemv 378, scan 36, matmul 136, subgroup 36, f16 12) |
| negative controls | 48 / 48 rejected (copy 3, reduce 6, gemv 9, scan 12, matmul 18) |
| timed quick runs | timestamp path and wall path both produce samples; no quantization on Dawn |
| SwiftShader page, quick, autorun | 142 ok, 0 wrong, 0 failed; 0 console errors; 2280.6 s |

SwiftShader's quick-run winners, for the record only (CPU): GEMV best C=16, S=2, R=16 on gate/up
(3.72× the engine config), C=32, S=4, R=16 on down (1.9×). Scan best wg 256, LUT in workgroup memory,
V=4 (1.82× the tail config). Matmul best 64×64×32 on 8×8 threads. Subgroup/tree 1.67×. None of
this is a GPU result.

Not done: no GPU run, no Safari run, and no test that `tuned.js` changes `eg2-wgsl-engine`'s speed, since the
engine was deliberately left unmodified.

Publishing: this session's git proxy refused pushes to the repo, so the commit went through the GitHub
API. That API takes whole files, so the root `README.md` index row and the three `METHODS.md` entries
were left as `root-docs.patch` (`git apply --unidiff-zero refractometer/root-docs.patch`) rather than resending the
two 350 KB files.
