# RESULTS: mdbr-leaf-ir in the browser with no inference runtime

Work in progress, 2026-10-20, on Oskar's MacBook Air (M1, 8 GB). Team slot `wasm` on the
embed-coord board (`../embed-coord`).

## What this is

leaf-ir (MongoDB/mdbr-leaf-ir, BERT/MiniLM-L6 shape: 6 layers, hidden 384, 30,522-token
uncased WordPiece, mean pool, Dense 384->768) running in the browser with:

- `web/tokenizer.js`: about 90 lines of BERT WordPiece (BertNormalizer, BertPreTokenizer,
  WordPiece, [CLS]/[SEP], truncation 128);
- `web/leaf.js`: the forward pass in JS, with every matmul in one WASM SIMD kernel;
- `web/kernel.wat` -> `web/kernel.wasm` (394 bytes): Y = X W (+ b), 4x4 blocks of f32x4;
- `run/export_weights.py`: one weights file. Choices: f32; int8 per output column for
  matmuls and per row for the token table; or the k16000 pruned vocab (from ettin-native)
  with an int8 or int4 table. All dequantized to f32 at load.

No onnxruntime-web. ettin measured that runtime at 3.07 MB brotli plus ~205 ms session
create for leaf-ir, so the saving is about that, not more.

## Gates passed

| Gate | Result | Evidence |
|---|---|---|
| Tokenizer vs HF tokenizers, all 64,244 eval texts (posts, prompted queries, 60 edge cases) | 0 mismatches | `results/parity.json` |
| JS f32 vs MongoDB's ORT fp32 export, 460 fixtures | min cos 0.99999993 | `results/parity.json` |
| JS int8 (23.76 MB) vs ORT fp32 | min 0.99897, mean 0.99974 | `results/parity.json` |
| Patched ONNX (int8 weights dequantized into the fp32 graph) vs JS int8 | min cos 0.99999993 | `run/patch_onnx.py`; lets ORT score the engine's int8 on the full corpus |
| Browser page: 24 fixtures vs ORT fp32 at load | min cos 0.99952 | `web/index.html` |
| k16000 tokenizer vs HF with ettin's pruned tokenizer.json | 0 mismatches on 64,244 texts | `results/pruned.json` |
| k16000 t8 vs full-vocab int8 where tokenization agrees | identical (max diff 0, 380/460) | `results/pruned.json` |

## Finding: in-domain vocab pruning hurts out-of-domain queries

Cosine of k16000 (t8 / t4) to full-vocab ORT fp32 (`results/pruned.json`):

| Group | median | p05 | min |
|---|---|---|---|
| posts (250) | 0.99977 / 0.99836 | 0.956 / 0.960 | 0.868 |
| queries (150, mostly MS MARCO real_test) | 0.99971 / 0.99788 | 0.825 / 0.819 | 0.604 / 0.597 |

The 16k vocabulary was chosen by frequency over the AI-list posts and training queries.
Ordinary English outside that domain loses its token, for example:

- sidekick -> side ##ki ##ck (cos 0.604)
- tennessee -> ten ##ness ##ee (0.606)
- bleed -> b ##lee ##d (0.680)
- crest -> cr ##est (0.704)

hit@10 on synth_test will understate this, because those queries were written from the same
posts the vocabulary was mined from.

## Retrieval quality (SYM: each arm on its own 2-bit codes, 1,498 synth_test, hit@10)

From `../eg2-native-small/results/native.json` and the `*_perquery.json` files, run 2026-10-20
22:17 under the cpu.lock:

| Arm | hit@10 rx2 | paired vs leaf-ir fp32 |
|---|---|---|
| EG2 q4 (committed codes, anchor) | 0.918 | — |
| leaf-ir fp32 (MongoDB onnx/model.onnx) | 0.893 | — |
| **leaf-wasm int8** (this engine's weights; scored via the patched fp32 graph) | **0.889** | -0.004 [-0.010, +0.001] |
| MongoDB ORT int8 (model_quantized.onnx) | 0.889 | -0.005 [-0.013, +0.003] |

Retrieval cost of this engine's int8 vs fp32 is within noise.

## Latency (uncontended, cpu.lock held, M1 MacBook Air, one thread)

Browser: Chrome 152 (Claude desktop's browser), `web/index.html?n=30`, int8 full vocab.

| Kernel | warm query, median (15 tokens) | 16 / 32 / 64 / 128 tokens |
|---|---|---|
| relaxed-SIMD FMA, 4x16 tile | 11.0 ms (p90 12.9) | 11.1 / 19.3 / 36.6 / 74.4 ms |
| plain mul+add, 4x16 tile | 16.6 ms | 16.5 / 29.1 / 55.6 / 114.3 ms |

Node 25 on the same machine (`TIMING=1 node test/parity.mjs`, `results/timing.json`):
FMA 12.4 / 21.2 / 39.2 / 79.1 ms.

Cold start, served from localhost so fetch is not network time:
- init (parse + dequantize into wasm memory): 180-460 ms, noisy run to run;
- first query: 59-67 ms (JIT warm-up).

Dequantization runs in a scalar JS loop and is the next thing to move into WASM.

## Download size: raw is not what the wire carries

brotli -q 11, which is what a static host serves:

| Route | Contents | brotli MB |
|---|---|---|
| leaf-wasm int8 | weights 21.57 + vocab.txt 0.11 + engine and kernels ~0.01 | ~21.7 |
| ORT-web + MongoDB int8 | onnx 15.85 + runtime 3.07 + tokenizer.json ~0.25 | ~19.2 |

Per-column scales use the full int8 range: 7.2-7.4 bits of entropy per weight, against
5.1-6.3 for MongoDB's per-tensor codes, so they barely compress. Next: fewer levels per
column (`run/export_weights.py --levels`, `test/levels.mjs`) to trade precision for wire size.
