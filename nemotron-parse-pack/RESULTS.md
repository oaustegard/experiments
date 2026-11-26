# nemotron-parse-pack: NVIDIA's Nemotron Parse in a browser tab

NVIDIA's [Nemotron 3.5 Parse](https://huggingface.co/nvidia/Nemotron-3.5-Parse) reads a document page as one image and
writes it back block by block: a box, a class (title, text, table, picture, page header, ...) and Markdown text,
tables included, in reading order. This experiment asked whether a 631M-parameter document parser can run in a
browser tab on the GPU, how small its weights can be packed without losing accuracy, and where it lands against the
five smaller OCR models already on [anything-to-text](https://austegard.com/web-utilities/anything-to-text.html).
Try it at [austegard.com/experiments/nemotron-parse-pack/web/](https://austegard.com/experiments/nemotron-parse-pack/web/).

## Answer

Nemotron 3.5 Parse runs in the browser on hand-written WebGPU kernels from an all-8-bit pack of 561.1 MB. <!-- @evidence[[results/summary.json#pack_mb.v35i8]] -->
On 250 Omni Parse Bench pages it scores 79.27 <!-- @evidence[[results/summary.json#opb.np35_D.headline]] -->
against 80.00 at full precision, <!-- @evidence[[results/summary.json#opb.np35_fp32.headline]] -->
a gap of −0.72 (interval −2.71 to +1.08). <!-- @evidence[[results/summary.json#gaps.v35i8_vs_fp32.gap]] -->
That is 8.57 points above Indra OCR, <!-- @evidence[[results/summary.json#gaps.v35i8_vs_indra.gap]] -->
the best small model on the page, for about 100 times the download. On Oskar's Mac it reads a full page in 8.6 s <!-- @evidence[[results/summary.json#mac.v35i8_encode_s]] -->
and then writes 57.4 tokens a second. <!-- @evidence[[results/summary.json#mac.v35i8_decode_tok_s]] -->

Rounding the image encoder to 4 bits instead (403 MB) costs −7.86 points, <!-- @evidence[[results/summary.json#gaps.v35_vs_fp32.gap]] -->
most of it in tables, so that pack is an option and not the default. A pack of Nemotron Parse 2.0, built first by
mistake, scores 74.40 at full precision, <!-- @evidence[[results/summary.json#opb.np20_fp32.headline]] -->
5.6 points under 3.5 with 270M more parameters. A 6-bit encoder might sit between the two packs; it was not tried.

## Findings

1. **8-bit weights are free for Nemotron 3.5 Parse; a 4-bit image encoder is not.** All-int8 scores 79.27 against
   80.00 for fp32 on the same 250 pages; the int4 encoder (groups of 32) scores 72.14, with tables falling from 75.1 to
   59.4 and text from 74.8 to 67.5 (`results/opb_all.json`, `results/opb_np35_D.json`). The same 4-bit scheme cost
   2.0 only −2.88 (interval −7.29 to +1.32); 3.5's smaller encoder (width 1152 against 1280) takes it worse.
2. **3.5 is the most accurate model on anything-to-text by a clear margin.** Text 75.0 and overall 79.27 against
   Indra 63.3 / 70.70, OCRia-nano-1 66.7 / 68.18, PP-OCRv6 small 34.5 / 11.49, PP-OCRv6 tiny 32.9 / 10.95 and
   fine-tuned Tesseract 21.6 / 7.21 (`results/pareto.json`). The plain-text models score nothing on tables and
   layout, so their overall score is about a third of their text score.
3. **The harness reproduces NVIDIA's published score.** Our fp32 greedy run scores 80.00 against the benchmark's
   published Nemotron 3.5 Parse row's 81.89 on the same pages (gap −1.89, interval −4.20 to +0.16;
   `results/summary.json`); NVIDIA's run used vLLM with temperature 0.2.
4. **On a GPU, encoder attention is the whole cost.** For 2.0, the first engine's encoder took 684 s a page on the
   Mac, 99% of it in attention over 13,312 patches; tiled kernels brought it to 14.2 s (`results/np_perf_mac.json`).
   Gemini (gemini-4-argon) drafted the tiled matmul and flash-attention kernels; both passed the parity check
   unchanged, the matmul ships largely as written, and the attention ships as Fable 5.2's rewrite of Gemini's
   structure. For 3.5 the encoder takes 8.6 s, of which attention is about 80%.
5. **3.5 is smaller, faster and better than 2.0.** 631M against 903M parameters, 8.6 s against 14.2 s to encode on
   the Mac, and +5.60 points at fp32 (interval +1.30 to +9.93; `results/summary.json`). Its decoder reads the image
   features as a causal prefix instead of cross-attending to them, and it reads a page at its own shape instead of
   padding it to 1664 × 2048.
6. **A phone runs it at half size.** An iPhone (Chrome 158, iOS 27) encodes a 832 × 1024 page in 19.5 s and decodes at
   40 tokens a second with the 3.5 pack; a full-size 2.0 run stalled and a second one closed the tab
   (relay logs, `gpu-bridge-comms` `logs/nemotron-parse-pack/2026-11-26/`). The page starts phones at half size.
7. **Two traps would have corrupted the numbers.** `from_pretrained` casts both models to bfloat16 in `__init__`, so
   `.float()` afterwards holds bf16-rounded weights; `run/ref.py` and the Kaggle job reload `model.safetensors` over
   the model. And 2.0 writes `{}` in table cells covered by a `\multirow`, which NVIDIA's own `latex2html` keeps as a
   cell and shifts the row: table tests on the pilot went from 0 to 100 once `{}` reads as empty
   (`run/np_pages.py`, `web/engine/postprocess.js`).

## Method

**Benchmark.** Omni Parse Bench (datalab-to/omni_parse_bench), scorer omni-parse-bench 1.8.0 run inside the hub's
`scripts/jail.sh`, on a seeded sample of 250 of its 2,937 pages (`run/sample250.txt`, seed 20261126). Every model
reads the same PNGs, rendered by the benchmark's own `llm.image` (long edge 2048 px). Scores are the share of tests
passed: text (present, order, repeat), tables, layout, and the headline as their mean; intervals are the scorer's
paired bootstrap over documents (95%). `ocria-pack/run/grade.py` grades every run and the published Indra and
Nemotron 3.5 rows together.

**Nemotron runs.** `run/kaggle_eval.py` (2.0) and `run/kaggle_eval35.py` (3.5) on Kaggle T4 x2, fp32 activations,
greedy decoding with repetition penalty 1.1, the model card's prompt (3.5: text in pictures on; 2.0: off), at most
8,192 (2.0) or 8,744 (3.5) new tokens. Packed arms run the same model with weights replaced by exactly the dequantized
values the browser sees (`fq` in the Kaggle job, `run/export.py` and `run/ref.py` share the arithmetic). Outputs
become benchmark predictions through `run/np_pages.py`: for 3.5, NVIDIA's own `nemotron_parse.postprocess_document`
and the benchmark's 3.5 adapter mapping; for 2.0, the model card's `postprocessing.py` with the same mapping.

**Other models.** PP-OCRv6 tiny and small (fp32 ONNX, `ocria-pack/run/pp_pages.py`), Tesseract 5.3.4 with the
fine-tuned `eng-ft` fast model anything-to-text ships, OCRia-nano-1 in LiteParse 2.21.2's WebAssembly build with
LayoutNet, and Indra OCR from the benchmark's published rows. Download sizes are what anything-to-text fetches on a
first run for an English page, compressed: PP packs 1.57 / 6.26 MB, Indra mix8 4.83 MB, Tesseract 6.0 MB (layout core,
stock layout model, fine-tuned model), OCRia 14.0 MB (runtime plus packs).

**Packing.** `run/export.py`: int8 per output row with an f16 scale (max/127), or int4 per row in groups of 32 with
f16 scales (max/7); norms and biases f16; gzip shards of at most 47 MiB on R2. Schemes: `run/scheme_v1.json` (2.0),
`run/scheme_v35.json` (3.5, int4 encoder), `run/scheme_v35i8.json` (3.5, all int8, the default).

**Engine and timing.** `web/engine/nemotron.js` (hand-written WGSL, f32 activations), checked against `run/ref.py`
on SwiftShader at small sizes and on the Mac through the GPU bridge (`run/bridge/`). Timings are from the bridge on
Oskar's Mac (Chrome 157, Metal 3) and from the page's relay logs on his iPhone.

## Log

### 2026-11-26: Nemotron Parse 2.0, engine, pages

Asked (Oskar): pack NVIDIA-Nemotron-Parse-2.0 (the link given), build a test page with anything-to-text's PDF
flattener, add it to anything-to-text, and write a post with a Pareto chart. The 2.0 card's `new_version` pointed at
3.5, which was noted and not acted on. Fable 5.2 wrote the engine (SwiftShader parity 40/40 tokens at 512 × 416);
the first full-size Mac run read correctly but encoded in 684 s; tiled kernels drafted by Gemini brought it to
14.2 s. 8-page Kaggle pilot of four schemes (text similarity to fp32): all-int8 0.996, int4 encoder 0.961, int4
decoder too 0.949, int4 everything including the embedding 0.879 with a repetition loop. Shipped the int4-encoder pack
v1 (558 MB). 250-page OPB: fp32 74.40, v1 71.51.

### 2026-11-26: Nemotron 3.5 Parse

Oskar asked for 3.5 instead. Fable ported the engine (C-RADIOv4-SO400M encoder, NaViT input, decoder-only prefix with
grouped-query attention): 300 of 300 tokens equal to PyTorch at 1024 × 832 on the Mac; full page 8.7 s encode,
56 tokens a second. 8-page pilot: int4 encoder 0.964, all-int8 0.998. 250-page OPB: fp32 80.00, int4 encoder (v35,
403 MB) 72.14, all-int8 (v35i8, 561 MB) 79.27. Switched the test page and anything-to-text to v35i8; Mac timing for
it matched v35 (8.6 s, 57.4 tokens a second).
