# `_lib/wgsl/`: shared code for the hand-written WebGPU encoders

Three experiments run text encoders on WGSL kernels written here rather than on onnxruntime-web:
`eg2-wgsl-engine` (EmbeddingGemma 2, the 4-layer cut and the full 24 layers), `ettin-native`
(Ettin-17m and leaf-ir) and the older `leaf-wasm/web/leaf-gpu.js`. They began as copies of each other.
This directory is where the copies are merged, so that the next model costs a model description
instead of another engine.

| File | What |
|---|---|
| `runtime.js` | `SeqEngine` base class: uniform layout (`HEADER`), weight readers (`BYTE`), shader compilation, one static uniform and bind group per dispatch, packed variable-length batches, whole-file upload with binding windows, aligned per-tensor upload, `embed` / `embedTo` / `probe` / `destroy` |
| `kernels.js` | block templates and their recording helpers: `matmul` (weight formats q4 MatMulNBits, int8 per column, f32, each a WGSL snippet for one k-block inside a common loop and epilogue: alpha, column scale, bias, GELU, accumulate), `norm` (RMS or LayerNorm, residual before or after, layer scalar), `attention` (grouped-query, sliding window, separate or fused qkv, head dim up to 512), `pool`, `l2` |
| `vendor.py` | copies an engine plus `runtime.js` (as `wgsl-runtime.js`) into a site directory, rewriting the import |

An engine subclasses `SeqEngine`, sets `this.shaders = { ...KERNELS, ...ownShaders }`, and records
its forward pass in `_build(bytes)` with the `kernels.js` helpers and `this.step(kind, params, bufs,
grid)` for its own shaders. Only the shaders its steps use get compiled. Engines import `../../_lib/wgsl/runtime.js` and `kernels.js`, so a page that uses one must also
publish both: `pages.json` lists them next to `eg2-wgsl-engine/web`, and `vendor.py` handles copies
made for the site repo.

## Checks

The kernels' parity tests are per experiment and run in headless Chromium on SwiftShader:

```bash
python3 eg2-wgsl-engine/run/check.py            # EG2 cut (needs out/model/, see that RESULTS.md)
python3 eg2-wgsl-engine/run/check.py --full     # full EG2 (out/full/, from Hugging Face)
python3 ettin-native/run/check_wgsl.py          # Ettin-17m t4/t3, leaf-ir (web/models/*/model.onnx)
python3 eg2-vocab-codebook/run/engine_check.py DIR   # EG2 cut with an rhq token table
```

Run all four after touching `runtime.js`. Each passes at cosine >= 0.9999 against onnxruntime CPU or
the NumPy reference.

## Plan

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
   activation buffers reused by liveness. Retire `leaf-gpu.js`.
4. **ONNX importer and one verifier**: specs generated from the ONNX files plus a rule file per
   model family; a NumPy interpreter of the spec replaces each `ref.py`, and parity reports the
   first diverging named tensor.
5. **Acceptance test**: a model none of the engines has run (TIPS S/14, or MobileCLIP-S0's image
   tower for the live tagger), counted in lines of new code.
