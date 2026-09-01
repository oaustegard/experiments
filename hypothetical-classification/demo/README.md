# In-browser demo — gte-small over the WANDS taxonomy

`index.html` is standalone. It pulls `@huggingface/transformers` from jsDelivr and
`Xenova/gte-small` (33 MB int8 ONNX) from Hugging Face, embeds the 860 WANDS
`product_class` labels client-side, and snaps whatever you type onto the nearest one.
No API key, no server component. The browser caches the model after the first load.

```bash
python3 -m http.server 8801   # from this directory, then open localhost:8801
```

A `file://` open will not work — the ES-module import needs an http(s) origin.

## What it does

- **Load** downloads the model and indexes all 860 labels, reporting how long the
  indexing took and the embedding dimension.
- **Try it** ranks your text against every label, live, with cosine scores.
- **Run the benchmark** encodes all 468 WANDS queries in the page and scores two arms
  against their gold `product_class`: the raw query snapped directly, and a
  `gemini-3.5-flash-lite` register-prompt label snapped. The offline numbers (0.455 /
  0.594 and 0.571) sit in the last column so a drift is visible rather than silent.

`demo_data.json` (42 KB) carries the labels, the queries, the gold indices and the
pre-written labels; it is inlined into `index.html`, so the file is the whole app.

## Status

**The live version is deployed at austegard.com/ai-tools/taxonomy-snap.html.** This copy is the pre-deployment draft; the deployed one carries the WASM pin and the load-time smoke test described below.

**Originally written unverified, and it was broken.** The container this was authored in could not
run its Playwright/Chromium pair, so the load path, the WebGPU/WASM branch and the
benchmark timings are unexercised. Numbers the page computes should match `../RESULTS.md`;
if they do not, the page is wrong, not the results — those are reproduced by
`../recheck.py` with no browser involved.

## The WebGPU bug this shipped with

The first deployed version ran on `device: "webgpu"` and returned a collapsed embedding
space — every pair of labels ~0.995 apart, `Pillow` answering *Wedding, Drains, Fabric,
Flags, Candles*. transformers.js int8 on the WebGPU backend does this silently. The
weights and the calling code are both fine: fp32 PyTorch, `model_quantized.onnx` under
onnxruntime, and this page's own `embed()`/`rank()` on a CPU backend all rank *Standard
Bed Pillows* first at ~0.88, and that CPU path reproduces 0.434/0.588 over all 468
queries. Fixed in oaustegard.github.io#345 by pinning to WASM and adding a load-time
smoke test over 24 held-out queries.
