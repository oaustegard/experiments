# latent-calculator in the browser

The phase-2 `stream-left` port of `demo.py`, running client-side: SmolLM2-135M
frozen and split after decoder layer 16, four ONNX graphs, onnxruntime-web, no
build step and no framework.

Three routes to the same arithmetic answer:

| route | what crosses the model boundary |
|---|---|
| frozen model | nothing; the model answers on its own |
| tool as text | the result spliced into the prompt as `" [<result>]"` |
| latent port | one 576-d encoder vector added to the residual stream after layer 16 at every answer position. Zero tokens, either direction. |

The latent route's query comes either from the trained attention head reading
layer 16 (the experiment) or from a regex over the prompt text (the null model,
`regex_query.py` ported to JS). The page has a toggle and says which one
produced the query.

## Serving it locally

```
cd web
python3 -m http.server 8000
# http://localhost:8000/index.html
```

`index.html` pulls onnxruntime-web 1.24.2 from cdnjs. `?ort=local` switches to
the vendored copy in `vendor/` (used by the offline Playwright test).
`?model=<url>` overrides `MODEL_BASE`, which defaults to `./model/`.

## Hosting the weights

GitHub *release assets* are not fetchable cross-origin from a browser, so
`MODEL_BASE` has to point at a Hugging Face `resolve` URL or at
raw.githubusercontent (both send `access-control-allow-origin: *` and honour
range requests). raw.githubusercontent caps a file at 100 MB, so the export
writes every graph as a small `.onnx` plus external-data files of at most
90 MB, and cuts anything still over the cap into `<file>.000`, `.001`, ...
pieces. `meta.json` carries the manifest; `LatentModel.loadGraph` fetches the
pieces, concatenates them, and hands ORT an `externalData` entry per location.

SmolLM2 ties `lm_head` to the input embeddings, so the 49152x576 table is
written once as `embed[.variant].data` and both halves' graphs point at it.
The loader caches fetched buffers by name, so it is downloaded once too.

## Where the weights go

`model/` is gitignored; regenerate it with

```
python3 export_onnx.py            # writes model/*.onnx, meta.json, tokenizer.json
```

The page fetches, from `MODEL_BASE`:

| file | purpose |
|---|---|
| `meta.json` | layer counts, kv geometry, eos id |
| `tokenizer.json` | vocab + merges for the JS BPE |
| `lower{,.fp16,.int8}.onnx` | embeddings + layers 0–16 → hidden16, present kv |
| `upper{,.fp16,.int8}.onnx` | layers 17–29 + norm + lm_head → logits, present kv |
| `query_head.onnx` | `AttnQueryHead`: hidden16 + mask → op logits, 12×11 slot logits |
| `encoder.onnx` | `ResultEncoder`: symbols[1,14] + step[1] → vec[1,576] |

## Files

| | |
|---|---|
| `llama_min.py` | pure-torch Llama with explicit past k/v; `python3 llama_min.py` runs the verification |
| `export_onnx.py` | the four exports, fp16 and int8 variants, ORT validation |
| `pipeline.py` | python-ORT reference of the exact browser algorithm |
| `bpe.js` | byte-level BPE from `tokenizer.json` (Digits + ByteLevel pre-tokenizers) |
| `latent.js` | the three routes, the calculator, the regex query, and the UI |
| `index.html` | the page |
| `verify_tokenizer.py` / `.mjs` | JS tokenizer against the python one |
| `test_web.py` | Playwright: page vs `pipeline.py`, plus a WebGPU attempt |

## What was checked, and against what

| check | result |
|---|---|
| `llama_min` logits vs the HF model, 5 prompts | max abs diff **0.0** |
| `llama_min` two-half pipeline vs `eval.py`'s stream arm, 5 dataset rows | **5/5** token for token |
| ONNX vs torch, random inputs (`export_onnx.py`) | lower 6.1e-5, upper 1.3e-4, query head 1.9e-5, encoder 7.6e-6 |
| `pipeline.py` latent route vs `demo.py` stream arm, 20 dataset rows | **20/20** text, **20/20** decoded query |
| `bpe.js` vs the python tokenizer, 30 prompts | **30/30** encode and decode |
| browser latent route vs `pipeline.py`, 3 prompts, fp32 and fp16 | **3/3** each |


## Sizes and validation

Weight variants, as emitted by `export_onnx.py` (download totals count the
shared table once):

| variant | download | lower | upper | largest file | table shared | latent exact-match, 50 rows |
|---|---|---|---|---|---|---|
| fp32 | 538.8 MB | 15.6 MB graph + 225.6 MB weights | 11.9 MB + 172.6 MB | 90.0 MB | yes | 0.860 |
| fp16 | 270.2 MB | 30.8 MB + 90.3 MB | 23.5 MB + 69.0 MB | 88.5 MB | yes | not measured* |
| int8 | 249.2 MB | 61.0 MB | 46.7 MB + 28.3 MB | 90.0 MB | no | **0.640** |

plus `embed[.variant].data` at 113.2 MB (fp32/int8) or 56.6 MB (fp16),
`query_head.onnx` 1.0 MB, `encoder.onnx` 3.1 MB, `tokenizer.json` 2.1 MB.

\* fp16 on the CPU execution provider has no native kernels and runs far too
slowly to score 50 rows here; what was checked instead is that its argmax
agrees with fp32 on the export probe, that python ORT fp16 gives the same
three answers as fp32, and that the browser reproduces those three exactly.

int8 (dynamic, MatMul only) costs 22 points against fp32, far outside the
2-point budget, so **it is exported but should not be shipped**. It also
barely helps: the embedding table is a `Gather`, which dynamic quantization
leaves alone, so it stays fp32 and dominates the download.

## Backends

`navigator.gpu` present -> WebGPU is tried first, WASM otherwise; the page shows
which one it got. GitHub Pages does not send the COOP/COEP headers, so
`crossOriginIsolated` is false there and `ort.env.wasm.numThreads` falls back to
1 -- the page reads `self.crossOriginIsolated` and sets threads accordingly
rather than failing.

onnxruntime-web 1.24's extended graph optimizers (SimplifiedLayerNormFusion)
reject the fp16 graph that python ORT 1.29 accepts, so `init` steps
`graphOptimizationLevel` down `all -> basic -> disabled` and reports the level
it settled on. Buffers are cached, so a retry costs no download.

The query head and the result encoder always run on WASM: they are 1 MB and
3 MB, and keeping them off the GPU keeps the injected vector bit-identical
across backends.

Measured in headless Chromium 141 in this container (single WASM thread, no
cross-origin isolation, weights served from localhost), `test_web.py`:

| variant | model load | ms per latent answer | generated tok/s | matches `pipeline.py` |
|---|---|---|---|---|
| fp32 | 8.5 s | 497 | 8.04 | 3/3 |
| fp16 | 6.2 s | 1350 | 2.96 | 3/3 |

`test_web.py --skip-webgpu` avoids the redundant second browser run in an
environment like this one, where the WebGPU EP cannot exist.

fp16 is slower on WASM because that backend has no native fp16 kernels; it is
the variant for WebGPU, where the halved weights are the point. **WebGPU could
not be exercised here at all**: `navigator.gpu` is undefined in this headless
Chromium both with and without `--enable-unsafe-webgpu
--enable-features=Vulkan,WebGPU --use-angle=swiftshader`, so the WebGPU path in
`latent.js` is unverified and the page falls back to WASM. Before trusting it,
compare its output token for token against the WASM path -- an int8 WebGPU
pipeline silently collapsed an embedding space on a sibling demo in this repo.

Shipped weights: https://huggingface.co/austegard/latent-calculator-web (fp32 and fp16, 815 MB, uploaded 2026-09-07).
