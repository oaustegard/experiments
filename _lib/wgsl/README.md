# `_lib/wgsl/`: the shared WebGPU kernel library

The hand-written WGSL models in this repo (text encoders, decision models, a decoder, LoRA and
full-weight trainers) began as copies of each other. This directory is where the copies are merged:
one shape-specialized generator per op, each with a parity case, and a small runtime the engines
build on. Engines importing it today: `eg2-wgsl-engine` (EmbeddingGemma 2), `ettin-native`
(Ettin-17m, leaf-ir), `browser-decider`, `adapter-library`, `moederncolbert-bench` and `lfm-wgsl`.
The migration plan for the rest is under "Kernel library" below.

| File | What |
|---|---|
| `runtime.js` | `SeqEngine` base class: uniform layout (`HEADER`), weight readers (`BYTE`), shader compilation, one static uniform and bind group per dispatch, packed variable-length batches, whole-file upload with binding windows, aligned per-tensor upload, `embed` / `embedTo` / `probe` / `destroy` |
| `kernels.js` | block templates and their recording helpers: `matmul` (weight formats q4 MatMulNBits, int8 per column, f32, each a WGSL snippet for one k-block inside a common loop and epilogue: alpha, column scale, bias, GELU, accumulate), `norm` (RMS or LayerNorm, residual before or after, layer scalar), `attention` (grouped-query, sliding window, separate or fused qkv, head dim up to 512), `pool`, `l2`, `attentionLong` and `gather` (sequences to 1,024 tokens), `embed` (t4 / t3 token tables), `ropeTable` and `rope`, `gated` (gelu, gelu_tanh, silu); the MoE ops `route` (top-2 of up to 8 experts into per-expert row lists), `expertMatmul` (grouped f32 matmul over the routed rows, 64 x 64 tiled or 8-row), `matmulTiled` (the same tiled kernel over all tokens), `combine` (gate-weighted sum into the residual) and `l2tok` |
| `tests/ops.js`, `tests/train_ops.js`, `tests/check_ops.mjs` | per-op parity cases against float64 references (finite differences for the backward kernels), and their headless runner; the same cases time each op on a real GPU through `gpu-bridge/jobs/wgsl-ops` |
| `tests/engine_ab.mjs` | an engine at a git ref against the working tree, same weights and inputs, max \|diff\| (0 expected when a move changes no shader); cases for browser-decider, adapter-library, ettin-native, eg2-cut, moedern-1.5M, lora-trainer and distill-trainer (weights not in git are downloaded once to a cache), add one per migrated engine |
| `train/spec.js`, `train/kernels.js`, `train/nce.js` | training kernels in the trainers' spec format (named uniforms and bindings, dynamic uniform offsets) and `SpecRunner`, the dispatch runner both trainers extend: LoRA and full-weight forward and backward passes, AdamW, the cosine and InfoNCE losses |
| `retrieval.js` | device-level retrieval ops with dispatch planners: `knnWGSL` (fused dot + top-K over a doc matrix, f32 or f16 docs, or a self-excluding kNN graph), `TOPK` + `planTopk` (multi-pass top-k over score rows), `SCAN` + `planScan` (exclusive prefix sum), `radixWGSL` + `planSort` (LSD radix sort, needs 17 KB of workgroup memory) |
| `tests/train_ops.js`, `tests/retrieval_ops.js` | the training kernels' and retrieval ops' parity cases, run by `check_ops.mjs` and the `wgsl-ops` bridge job after `ops.js` |
| `vendor.py` | copies an engine plus `runtime.js`, `kernels.js` and `retrieval.js` (as `wgsl-*.js`) into a site directory, rewriting the imports |

An engine subclasses `SeqEngine`, sets `this.shaders = { ...KERNELS, ...ownShaders }`, and records
its forward pass in `_build(bytes)` with the `kernels.js` helpers and `this.step(kind, params, bufs,
grid)` for its own shaders. Only the shaders its steps use get compiled. Engines import `../../_lib/wgsl/runtime.js` and `kernels.js`, so a page that uses one must also
publish both: `pages.json` lists them next to `eg2-wgsl-engine/web`, and `vendor.py` handles copies
made for the site repo.

## Checks

Per op: `node _lib/wgsl/tests/check_ops.mjs [filter]` runs every case in `tests/ops.js` (43 on
2026-11-09: each matmul format with bias, GELU, accumulate and alpha; RMS and LayerNorm in all three
residual modes; attention fused and separate, grouped-query, windowed, head dim 32 to 256; attentionLong to
1,024 tokens; gather; embed t4 / t3 / t8 with alpha and position rows; rope fused and plain; gated gelu,
gelu_tanh and silu; pool; l2; the MoE router, grouped matmuls in both kernels and row modes, tiled
matmul, combine, l2tok)
on random inputs and a packed batch of five sequences (1 to 128 tokens), against a float64 reference
in the same file. A case passes at max |gpu - ref| <= 1e-4 max |ref|; today's worst is 2.4e-7. It
takes 7 s. Injecting three bugs (swapped q4 nibbles, a window off by one, the pool divisor off by
one) failed the five cases that exercise them. On a real GPU the same cases are timed per op by the
bridge job `gpu-bridge/jobs/wgsl-ops` (inputs `{ sha, only, reps, rounds }`); that is the per-op
baseline a kernel change is compared against. First run, `wgsl-ops-1` on the Mac (apple metal-3,
Chrome 156, 2026-11-07): 19/19 pass, worst 2.4e-7; 0.006 to 0.040 ms per dispatch at these small
shapes, so the numbers mostly measure dispatch cost and catch regressions, not throughput
(`tests/results/wgsl-ops-1-mac.json`).

The same command then runs the training kernels in `tests/train_ops.js` (16 cases covering all 35 kernels): each forward
kernel against a float64 reference, and each backward kernel against central finite differences of
that reference in float64 (the 4-bit embed and copy; LayerNorm input, weight and bias grads, also through `ln_wpart` and with
accumulate; rope; attention global and windowed over a padded batch; GeGLU; bias + GELU and the bias
column sum; the head projection; segment cross-entropy; gather / scatter; mean pool; cosine loss; the
token-table gradient through `embed_f`; InfoNCE through `recordNce`; split-K `mm` with strided operands
and `sumz`; one clipped AdamW step). Same 1e-4 bar; the worst is 6.5e-6 (the AdamW step), the
gradients 5e-7 or better. Three injected bugs in `ln_bwd`, `attn_bkv` and `ln_pgrad` failed the four
cases that exercise them. The bridge job runs them after the timed ops (`train: false` skips them). On the Mac (apple metal-3, `wgsl-ops-step3`, 2026-11-07) all 47 cases pass; the training
cases' worst is 2.0e-5 (the AdamW step), the gradients 4.3e-7 or better (`tests/results/wgsl-ops-step3-mac.json`).

Per engine, in headless Chromium on SwiftShader:

```bash
python3 eg2-wgsl-engine/run/check.py            # EG2 cut (needs out/model/, see that RESULTS.md)
python3 eg2-wgsl-engine/run/check.py --full     # full EG2 (out/full/, from Hugging Face)
python3 ettin-native/run/check_wgsl.py          # Ettin-17m t4/t3, leaf-ir (web/models/*/model.onnx)
python3 eg2-vocab-codebook/run/engine_check.py DIR   # EG2 cut with an rhq token table
```

Run all four after touching `runtime.js`. Each passes at cosine >= 0.9999 against onnxruntime CPU or
the NumPy reference.

## Plan

### Kernel library (2026-11-07)

Oskar's go (2026-11-06) widens this directory from the three encoders' shared runtime into one
ingredient-level kernel library for every WGSL model in the repo, inference and training: one
generator per op, specialized to its shapes at compile time (that is where the 0.7 ms vs 4.9 ms of
ORT-web came from), a parity case per op in `tests/ops.js`, and engines that keep their
hand-written forward pass as the graph runner. It is not a general framework, so the model spec and
ONNX importer of phases 3 and 4 below are parked unless a model makes the hand-written pass the
bottleneck.

Inventory of the WGSL ML code outside this directory:

| where | kernels it owns | relation to `_lib/wgsl` |
|---|---|---|
| `eg2-wgsl-engine/web/engine.js` | embed q4 / rhq, prep, gelumul | imports runtime and kernels |
| `ettin-native/web/engine.js` | embed, rope, glu | imports runtime and kernels |
| `browser-decider/web/`, `adapter-library/web/` | engine: embed, rope, glu (ettin-native's); `wgsl/` is a vendored fork of this directory with `MAX_SEQ` 1024, `attnLongWGSL` (keys beyond one workgroup) and `GATHER`; adapter-library adds the LoRA side path | fork, two copies |
| `moederncolbert-bench/web/engine.js` | hash embed, rms, rope, attn, route, zero, gmm, mmt, glu, combine, l2tok | imports runtime and kernels, most ops its own |
| `lfm-wgsl/web/lfm.js`, `lfm3.js` | decoder: embed, gemv, short conv with state, rope into KV cache, cached attention, swiglu, last-token | imports runtime and the q4 matmul |
| `ettin-wgsl-train/web/train.js` | 26 forward and backward kernels: embed, ln, ln_bwd, ln_pgrad, tiled mm, sumz, rope, rope_bwd, attn, attn_bq, attn_bkv, geglu, geglu_bwd, gather, scatter, bias_gelu, gelu_bwd, colsum, head_out, head_out_bwd, loss, sqsum, norm, adamw | own spec format (named uniforms and bindings), own runtime with dynamic uniform offsets |
| `lifesci-ettin/web/distill.js` | full-weight gradients: embed_f, ln_wpart, pool, unpool, cosloss, tgrad | imports train.js |
| `leaf-wasm/web/leaf-gpu.js` | embed, matmul, attn, addln, pool | retired in step 6 (`leaf-engine.js` runs the same file on SmallEngine) |
| `embeddinggemma2-webgpu-tail/web/tail.js`, `gpu-bridge/jobs/{knn,knngraph,scan,sort}` | retrieval: encode, dot_row, scan, colsum, topk, sort | top-k, kNN, scan and sort in `retrieval.js` since step 6; tail.js keeps its remex-specific encode, prep, LUT scan and colsum |
| `trm-doom`, `image-to-wgsl`, `ettin-kernel-profile` | small one-off models and a profiling job | standalone |

Not WGSL, so nothing to fold: `hier-router` (PyTorch; its routers would run on adapter-library's
engine) and the live tagger (transformers.js). Graphics, audio and physics shaders
(camera-resonator, paper-drum, room-drum, navier-stokes-explainer and the like) are out of scope.

Migration order. Every step ends with that engine's own check at cosine >= 0.9999, its ops in
`tests/ops.js`, and a bridge A/B on the Mac within 5% of the old commit, with the device named:

0. **Per-op test spine** (done 2026-11-07): `tests/ops.js`, `tests/check_ops.mjs`, bridge job
   `wgsl-ops` for the per-op baseline.
1. **Fold the browser-decider / adapter-library fork back** (done 2026-11-07). `maxSeq` is a
   SeqEngine option (default 128, the two decider engines pass 1024), `attentionLong` and `gather`
   live in `kernels.js` with cases, both `web/wgsl/` copies are gone and the engines import
   `../../_lib/wgsl/`, which `pages.json` publishes with them. `tests/engine_ab.mjs` gives max |diff|
   0 against the forked versions for both engines (sequences of 1 to 1,024 tokens, router and a
   spike), and adapter-library's own page check passes its fixtures (6/6 argmax, logits within
   3.9e-5 of NumPy). On the Mac (bridge job `engine-ab`, run `engine-ab-1`, apple metal-3) outputs
   are identical and `score()` time is within 0.98x to 1.01x of the fork for 60, 829 and 1,500 tokens
   (`tests/results/engine-ab-1-mac.json`).
2. **Forward ops duplicated across engines** (done 2026-11-07).
   `embed` (token tables t4 and t3, optional alpha and BERT position rows), `ropeTable` + `rope`
   (fused q/k rows or a plain buffer, any number of theta tables) and `gated` (gelu, gelu_tanh or
   silu times a second operand, ModernBERT's [gate | up] row or in place against a strided buffer)
   replace each engine's copy. ettin-native, browser-decider and adapter-library now define no
   shaders of their own; eg2-wgsl-engine keeps only its q4 and rhq tables and the fused
   norm-plus-RoPE `prep`; moederncolbert-bench keeps its MoE, hash-embedding and ColBERT ops for
   step 4. `tests/engine_ab.mjs` gives max |diff| 0 against the previous commit for all five
   engines (decider, adapter-library with router and a spike, Ettin-17m, the EG2 cut, MoEdernColBERT
   1.5M with both matmul paths). On the Mac (apple metal-3) outputs are identical for the EG2 cut,
   Ettin-17m, the decider and adapter-library; EG2 cut 0.9 / 0.9 ms short and 1.9 / 1.9 ms at 118
   tokens, Ettin 0.7 / 0.7 ms (`wgsl-ab-step2`); decider and spike `score()` 0.98x to 1.03x, after a
   rerun with 12 rounds put the decider's one 1.07x reading at 1.004x (`engine-ab-step2`, `-step2b`,
   in `tests/results/`). MoEdernColBERT was checked on SwiftShader only. LFM's SwiGLU waits for step 5.
3. **Training ops** into `train/` (done 2026-11-07). Done: `train/spec.js` (the trainers' spec
   format, `wgsl()`, `grid()`, the erf GELU and its derivative, the workgroup reductions),
   `train/kernels.js` (`kernels(C)`, the LoRA trainer's 26 forward and backward kernels, and
   `fullKernels(C)`, which adds lifesci-ettin's full-weight gradients and pooling losses) and
   `train/nce.js` (the in-batch contrastive loss). ettin-wgsl-train's `train.js` re-exports them, so
   decider-lora-creator and the bridge jobs that import it are unchanged. All 60 shaders are
   byte-identical to before, and `tests/engine_ab.mjs` gives max |diff| 0 for both trainers (loss,
   every gradient and one AdamW step; the full-weight trainer under both the cosine and the
   contrastive loss). Both trainers now extend `SpecRunner` in `train/spec.js` (uniform slots, bind
   group cache, `op` / `buf` / `clear` / `flush` / `read`) instead of carrying a copy each; also max
   |diff| 0. Every training kernel has a parity case (`tests/train_ops.js`, finite differences for
   the backward ones). The trainer's forward ops (LayerNorm, rope, attention, GeGLU) stay separate
   from the inference ones (Oskar, 2026-11-07): the trainer pads batches and saves values for the
   backward pass, inference packs variable-length sequences, and merging them would mean rewriting
   the backward kernels or slowing inference to remove about four kernels.
4. **MoE ops** from moederncolbert-bench (done 2026-11-08): `route`, `expertMatmul`, `matmulTiled`,
   `combine` and `l2tok` in `kernels.js`, their six shaders byte-identical to the engine's, which now
   defines only its hash embedding, RMSNorm and 512-token attention. `engine_ab.mjs moedern-1.5M`
   (both matmul paths) gives max |diff| 0. Eleven per-op cases; the router is compared as its gates
   plus each slot's expert rebuilt from the lists, since the atomics leave list order free.
   Injected bugs in the router's lists, the 8-row grouped matmul and the combine failed their cases.
   On the Mac (apple metal-3, `wgsl-ops-step4`) all 58 cases pass, the MoE ones at 4.3e-7 or better,
   0.005 to 0.08 ms each (`tests/results/wgsl-ops-step4-mac.json`). No engine timing run: the
   shaders and dispatch sequence are unchanged.
5. **Decoder ops** from lfm-wgsl: gemv, short conv with state, KV-cache rope and attention. Skipped
   (Oskar, 2026-11-08).
6. **Retrieval ops** (done 2026-11-09) into `retrieval.js`: tail.js's top-k (shader byte-identical,
   tail.js imports it; its own edge-case page `test_topk.html` gives 0 mismatches over 10 cases), and
   the bridge jobs' kNN, scan and sort (scan and sort byte-identical; knn and knngraph merged into one
   template with `excludeSelf`). The four jobs import the module at `inputs.sha` (default `main`).
   Five exact-reference cases in `tests/retrieval_ops.js`. tail.js's encode, query prep, LUT scan and
   colsum stay in tail.js: they are remex-format specific and have one user.
   `leaf-gpu.js` is retired. `leaf-wasm/web/leaf-engine.js` maps leaf.js's int8 file onto ettin-native's
   SmallEngine (q/k/v split by column, a zero type row) with one new kernel option, the `t8` int8
   token table with an f32 scale per row. Against leaf-gpu.js on 24 sequences of 1 to 127 tokens the
   minimum cosine is 0.9999999999988, against leaf.js 0.9999999999986 (SwiftShader). The int8 file is
   not hosted anywhere a bridge job can fetch it, so the new path has no Mac timing of its own; the
   same SmallEngine runs leaf-ir's t4 build at 3.8 ms on the Mac (ettin-native RESULTS) against
   leaf-gpu.js's 5.0 ms (leaf-wasm RESULTS). The other five engines stay bit-identical (`engine_ab.mjs`).

trm-doom and image-to-wgsl stay where they are until a second experiment needs one of their ops.

### Generic engine (2026-10-26)

Option B of the generic-engine design (2026-10-26): a block-level model spec, compile-time-specialized
kernel templates with pluggable weight formats, and this shared runtime. Each phase ends with every
ported model at parity and no Metal slowdown beyond 5% (EG2 cut 0.7 ms per short query, Ettin-17m 2.4 ms).

1. **Shared runtime** (done): the common JS out of both engines; shaders and forward passes unchanged.
   The EG2 cut, full EG2, Ettin and leaf-ir checks give the same cosines as before the move, to the
   last digit; the rhq check passes.
2. **Kernel templates and weight-format plugins**: one matmul with format snippets (MatMulNBits q4,
   int8 per column, f32), one attention template (GQA, sliding window, fused-qkv strides, head dim
   64 to 512), one norm (RMS and LayerNorm), shapes as WGSL constants instead of uniforms.
3. **Model spec and buffer planner**: `model.json` per model replaces each hand-wired `_build`;
   activation buffers reused by liveness.
4. **ONNX importer and one verifier**: specs generated from the ONNX files plus a rule file per
   model family; a NumPy interpreter of the spec replaces each `ref.py`, and parity reports the
   first diverging named tensor.
5. **Acceptance test**: a model none of the engines has run (TIPS S/14, or MobileCLIP-S0's image
   tower for the live tagger), counted in lines of new code.
