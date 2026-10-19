# embeddinggemma2-webgpu-tail

WebGPU kernels for everything after EmbeddingGemma 2's transformer: pooling,
normalization, centring, rotation, 2-bit remex codes, packing, and the ADC scan
over the packed corpus. They run on onnxruntime-web's own `GPUDevice`, so an
embedding goes from ORT's output buffer to a packed code with no readback.
`SPEC.md` is the contract; `RESULTS.md` has the answer.

## Status (2026-10-19, round 4)

| piece | state |
|---|---|
| `SPEC.md` | done |
| `oracle/` — fixtures from remex 2.1.0, checker, float32 numpy model | done; checker goes red on wrong bit order, missing kappa, dropped score offset, top-k swap; `--vectors` builds fixtures from real EG2 Quora vectors |
| `web/tail.js` — WGSL kernels + host | passes the checker on SwiftShader: synthetic, 2 of 3.1M codes flip; real EG2 vectors, 6 of 3.1M; all within 1e-5 of a boundary; scores within 3.5e-7. Encode tiles 4 vectors per workgroup (bit-identical to one, 2.9x faster on SwiftShader); top-k runs on the GPU (`run/run_topk_edge.py`: 10 edge cases) |
| `web/tail_ref.js` — CPU reference | passes the checker |
| `interop/` — ORT 1.30.0 gpu-buffer output fed straight to the tail | passes the checker; 0 `getData` calls on the encode path |
| `web/demo.html` — real EG2 q4 page with startup gate, search, bench | passes with the real model in headless Chromium (embeddings at cosine 1.0 to CPU onnxruntime, model-card scores reproduced, gate 0/24,576) and with the dummy graph; scan check 7.2e-7 on the GPU's own codes; all timings SwiftShader |
| native comparison | `run/native_bench.py` vs `run/run_bench.py` on one 4-core CPU: see `RESULTS.md` finding 5 |
| real-GPU numbers | Metal 3 (Oskar's Mac, Chrome 154): embed 32 docs 549.5 ms, + GPU encode 5.0 ms; scan 100k × 20 in 13.1 ms; see `RESULTS.md` |

The default fixtures are synthetic EG2-shaped vectors (‖mean‖ ratio 0.852, pair
cosine 0.724); `fixtures_eg2/` holds 4,096 real EG2 Quora vectors (0.788, 0.622).

## Regenerate and check

```bash
pip install remex==2.1.0 onnx playwright
python3 -I oracle/make_fixtures.py fixtures          # 2 s, deterministic
python3 -I oracle/sim_tail.py fixtures out/sim && python3 -I oracle/check_tail.py fixtures out/sim
node run/run_ref.mjs && python3 -I oracle/check_tail.py fixtures out/ref
python3 -I run/run_gpu.py && python3 -I oracle/check_tail.py fixtures out/gpu
(cd interop && npm install --ignore-scripts) && python3 -I interop/make_dummy_onnx.py
python3 -I run/run_interop.py && python3 -I oracle/check_tail.py fixtures out/interop
python3 -I run/prep_assets.py                        # writes web/assets/R.f32.txt (binary, 2.36 MB)
python3 -I run/run_demo.py                           # ?local=1&model=dummy smoke run

# real EG2 vectors (R2 embstore, public read) and the real model
python3 <claude-workspace>/scripts/embstore.py get quora/embeddinggemma-2/fp32-768-first100k out/eg2q
python3 -I oracle/make_fixtures.py fixtures_eg2 --vectors out/eg2q
python3 -I run/run_gpu.py --fixtures fixtures_eg2 --out out/gpu_eg2 && python3 -I oracle/check_tail.py fixtures_eg2 out/gpu_eg2
pip install onnxruntime tokenizers
python3 -I run/ref_real.py                           # CPU q4 reference; downloads 175 MB into out/hf/
python3 -I run/run_demo.py --real                    # hosted page, real model: ~40 min on SwiftShader
python3 -I run/run_demo.py --real --no-bench         # startup, gate, search only: ~12 min

# top-k edge cases, and the bench against native CPU
python3 -I run/run_topk_edge.py
git show a6fa2213:embeddinggemma2-webgpu-tail/web/tail.js > out/tail_untiled.js   # the one-vector encode
python3 -I run/run_bench.py --untiled out/tail_untiled.js   # ~10 min on SwiftShader
python3 -I run/native_bench.py
```

The Playwright scripts use Chromium at
`/opt/pw-browsers/chromium-1194/chrome-linux/chrome` with SwiftShader WebGPU.
SwiftShader timings are not GPU timings.
