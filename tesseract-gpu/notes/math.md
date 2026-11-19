# Math coverage for a GPU Tesseract (server and WebGPU)

Survey, 2026-11-19. Tesseract's LSTM line models emit characters, not LaTeX, and the legacy
`equ.traineddata` runs only on the legacy engine, so math needs a separate recogniser that runs
beside the line OCR. Numbers marked **unverified** could not be sourced from a primary document
during this survey. Sizes are from the Hugging Face file tree unless noted.

## 1. What issue #114 will provide

Source: oaustegard/experiments#114 (open, opened 2026-11-19, no comments yet).

- Port Indra's math path to WebGPU in `indra-browser/web/`: the math classifier `mathcls.onnx`
  and the encoder-decoder `math_enc.onnx`, `math_dec_init.onnx`, `math_dec_step.onnx` with
  `math_meta.json`.
- Pack those weights and load them lazily in anything-to-text, only when the classifier flags a
  math line or the user turns math on.
- Target: Indra's `--math` score on OmniParseBench (71.4 vs 66.7 without math, per the Indra
  README).

Reuse for a Tesseract pipeline, rather than a second port:

- The #114 deliverable is a self-contained WGSL module (`math_enc`, `math_dec_*`, the classifier)
  with a JS pipeline. A Tesseract page can call it as a function from a line-level hook: crop each
  line that Tesseract flags, run the classifier, and send only flagged crops to the decoder.
- Reuse the classifier as the detector and the Indra decoder as the recogniser. Do not port a
  second math model unless #114 shows a gap.
- Keep the math pack as a separate fetched artifact, so the Tesseract page carries it only when
  needed, as #114 already plans.
- Licence follows the weights: Indra's weights are under a modified OpenRAIL-M licence, which
  limits commercial use. Tesseract itself is Apache-2.0.

Caveat: #114 is a single-owner port with no interface contract yet. The Tesseract side needs an
agreed function signature (crop in, LaTeX string plus confidence out) before either side builds
on it.

## 2. Math recognisers

| Model | Params | Architecture (encoder / decoder / AR?) | Weights | Licence | Benchmark number (source) | ONNX export |
|---|---|---|---|---|---|---|
| Indra math (`datalab-to/indra-ocr`, math models) | unverified (default pack is 7.3M total for the non-math models per the card; math extra) | CNN or ViT encoder / small decoder, autoregressive (`math_dec_step` is a per-token step) | ONNX, `math_*.onnx` (enc 9.9 MB, dec_init 1.4 MB, dec_step 8.4 MB, cls 0.19 MB) | OpenRAIL-M (modified); code Apache-2.0 | OmniParseBench math tests **55.8** with `--math`; headline 71.4 vs 66.7 off ([indra-ocr README](https://huggingface.co/datalab-to/indra-ocr)) | Yes, native |
| pix2tex / LaTeX-OCR (`lukas-blecher/LaTeX-OCR`) | unverified (~ConvNeXt/ResNet + ViT, roughly 0.1B; **unverified**) | ViT encoder with ResNet backbone / Transformer decoder, autoregressive | PyTorch checkpoint | MIT | Texify's comparison table: BLEU **0.382659** on Texify's arXiv + im2latex benchmark, a biased comparison that the Texify README itself flags ([texify README](https://github.com/VikParuchuri/texify)). No im2latex-100k number in the README (unverified) | No official export found |
| Texify (`VikParuchuri/texify`) | unverified (Donut base; **unverified** count) | Donut (Swin encoder / BART-style decoder), autoregressive | PyTorch | Code GPL-3 (unverified); weights CC BY-SA 4.0 per README (line 173) | Same table: BLEU **0.842349**, METEOR 0.885731, edit distance 0.0651534 ([texify README](https://github.com/VikParuchuri/texify)). Deprecated by its author, functionality moved to surya | No |
| Pix2Text-MFR (`breezedeus/pix2text-mfr`) | unverified (TrOCR-based; encoder ONNX 87.5 MB, decoder 30.1 MB, so ~118 MB fp32) | ViT encoder / Transformer decoder, autoregressive | ONNX (`encoder_model.onnx`, `decoder_model.onnx`) + tokenizer | MIT | No benchmark number on the model card (**unverified**); the Pix2Text article is linked from the card | **Yes**, shipped as ONNX |
| UniMERNet tiny / small / base (`opendatalab/UniMERNet`, `wanderkid/unimernet_*`) | tiny and small are smaller than base; exact counts **unverified**. File sizes: tiny 441 MB, small 773 MB, base 1.3 GB (PyTorch `.pth`, from the repo README) | Hybrid ViT-style encoder / Transformer decoder, autoregressive | PyTorch `.pth` | Model cards say apache-2.0 (HF metadata); repo licence **unverified** | BLEU and CDM figures are in repo images only (`fig1_bleu.jpg`, `fig2_cdm.jpg`), numeric values **unverified**. CDM is the metric the authors recommend ([UniMERNet README](https://github.com/opendatalab/UniMERNet)) | No official export found |
| PP-FormulaNet-S / PP-FormulaNet_plus-S (PaddleOCR; `PaddlePaddle/PP-FormulaNet_plus-S`) | unverified; the "S" variants are the small ones | Hybrid encoder / Transformer decoder, autoregressive | Paddle inference (`inference.pdiparams`) | Apache-2.0 (HF metadata) | En-BLEU table exists on the model card (lines 22 to 32); the exact values were not captured in this survey (**unverified**). Evaluated on UniMERNet's simple and complex sets | Not in repo; conversion from Paddle required (**unverified**) |
| PDFluent formula-recognition-onnx (`PDFluent/formula-recognition-onnx`) | unverified | Encoder / decoder ONNX, autoregressive | ONNX `encoder.onnx`, `decoder.onnx`, `vocab.json` | **AGPL-3.0**, with a separate commercial licence file | None on the card (**unverified**) | Yes, native |

Notes on the table:

- Only Indra and Texify give a number on a named benchmark that this survey could read directly.
  The Texify comparison is run on Texify's own corpus, not UniMER-Test or im2latex-100k.
- CDM is the metric the UniMERNet authors recommend over BLEU. None of the other rows report it
  in text.
- The PDFluent row shows that an ONNX export can carry a copyleft licence. It rules that file out
  for a page that is distributed.
- No 2025-2026 small formula recogniser beyond these was verified in this pass (search was limited).

## 3. Detection: how math lines get flagged

- **Indra `mathcls.onnx`**: a small line classifier, 191 KB in fp32. It runs on each line crop
  and flags math lines before the decoder runs. This is the cheapest option and it already exists.
- **Layout models**: PP-DocLayout-style detectors label "formula" regions. Their sizes are
  **unverified** in this survey.
- **Heuristic**: symbol-density and glyph-class check on the Tesseract output (e.g. share of
  characters in a math set such as `∑∫√=±^_{}`, or low word-dictionary hit rate). Zero bytes and
  no GPU, but it misses formulas that look like text.
- Recommendation for a Tesseract pipeline: run the heuristic first (free), then the Indra
  classifier on candidate lines only.

## 4. Recommendation

Ranking criteria: accuracy per megabyte, and new GPU kernel types beyond conv, LSTM and matmul.

**WebGPU page.** Load the Indra math path (the #114 port).

- It is the only candidate whose number on a named benchmark (OmniParseBench math 55.8) is in a
  source this survey could read, and it is already being ported.
- Size: about 19.8 MB of fp32 ONNX (enc 9.9 + dec_init 1.4 + dec_step 8.4 + cls 0.19). Packing
  it lowers this further; the packed size is **unverified** until #114 reports it.
- New kernels: the decoder needs a per-step attention or recurrent update and a softmax over the
  vocabulary, and the encoder needs whatever conv variant it uses. Exact count depends on the
  architecture, which **#114 must confirm**.
- Licence: OpenRAIL-M (modified). Free for research, personal use and start-ups under $20M
  (per the Indra README). It does **not** cover broad commercial use without a licence from
  datalab. Ship only under those terms, or ask Oskar before publishing the packed weights.

**Server.** Pix2Text-MFR (MIT, ONNX, ~118 MB fp32) is the simplest permissive option: it runs on
onnxruntime with no new GPU kernels in our stack, and MIT imposes no commercial limit. Its
accuracy is **unverified**. Texify has the best published BLEU in the table, but its licence is
CC BY-SA on weights and its code is deprecated; use it only if its accuracy is confirmed on our
own set. Avoid PDFluent (AGPL) for any shipped product.

**Ranking, accuracy per MB.** Only Indra has a sourced accuracy and a known size, so it ranks
first on that axis by default. Pix2Text-MFR and Texify cannot be ranked on accuracy per MB until
their sizes and scores are checked on the same benchmark.

**Next checks, in order:**
1. Run Pix2Text-MFR and Indra math on the same 50 formula crops from UniMER-Test (dataset linked in
   the UniMERNet README) with CDM. This fills the unverified cells.
2. Read the #114 port's architecture to count its new kernels.
3. Decide with Oskar whether the OpenRAIL-M limit is acceptable for the WebGPU page.
