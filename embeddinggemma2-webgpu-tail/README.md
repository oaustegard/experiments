# embeddinggemma2-webgpu-tail

WebGPU kernels for everything after EmbeddingGemma 2's transformer: pooling,
normalization, centring, rotation, 2-bit remex codes, packing, and the ADC scan
over the packed corpus. They run on onnxruntime-web's own `GPUDevice`, so an
embedding goes from ORT's output buffer to a packed code with no readback.
`SPEC.md` is the contract.

## Status (2026-10-18)

| piece | state |
|---|---|
| `SPEC.md` | done |
| `oracle/` — fixtures from remex 2.1.0, checker, float32 numpy model | done; checker goes red on wrong bit order, missing kappa, dropped score offset, top-k swap |
| `web/tail.js` — WGSL kernels + host | passes the checker on SwiftShader: 2 of 3.1M codes flip, both within 1e-5 of a boundary; scores within 3.5e-7 |
| `web/tail_ref.js` — CPU reference | passes the checker |
| `interop/` — ORT 1.30.0 gpu-buffer output fed straight to the tail | passes the checker; 0 `getData` calls on the encode path |
| `web/demo.html` — real EG2 q4 page with startup gate, search, bench | `?local=1&model=dummy` smoke run passes (gate 0/24,576 code mismatches, 10 hits, bench numbers, no page errors); the real-model path has never run; the scan bench's agreement check reads 0.005 max abs error (see the handoff issue) |

Fixtures are synthetic EG2-shaped vectors (‖mean‖ ratio 0.852, pair cosine 0.724;
EG2 on SciFact measured 0.845 and 0.72). The session that built this could not
reach Hugging Face's file CDN or the R2 embedding store.

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
```

The Playwright scripts use Chromium at
`/opt/pw-browsers/chromium-1194/chrome-linux/chrome` with SwiftShader WebGPU.
SwiftShader timings are not GPU timings.
