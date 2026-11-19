# Math recognition for the GPU Tesseract (server and WebGPU)

This file surveys recognisers that turn formula lines into LaTeX, for the two
tesseract-gpu engines: the PyTorch server engine and the WebGPU engine. Tesseract's
LSTM line models emit characters, not LaTeX, and the legacy `equ.traineddata`
runs only on the legacy engine, so math needs a recogniser beside the line OCR.
The recommendation: on the WebGPU page, use Indra's math reader, which already
runs in `indra-browser`; on the server, use Pix2Text-MFR once its accuracy is
measured. Numbers marked **unverified** could not be sourced from a primary
document in this survey. Sizes come from the Hugging Face file tree unless noted.

## 1. Indra's math reader in indra-browser

Indra's math path runs on WebGPU in `indra-browser/web/`. The math classifier
flags each line that holds math, and the math reader writes that line as LaTeX.
`indra.js` takes a mode, `setMath("off" | "auto" | "on")`, and `readMath` runs
the reader on the flagged lines. The port is tracked in oaustegard/experiments#114.

The files, from the Indra OCR Hugging Face repository (`datalab-to/indra-ocr`):

| file | size |
|---|---:|
| `mathcls.onnx` (line classifier, fp32) | 191,005 bytes |
| `math_enc.onnx` (encoder) | 9.9 MB |
| `math_dec_init.onnx` (decoder, first step) | 1.4 MB |
| `math_dec_step.onnx` (decoder, per token) | 8.4 MB |
| `math_meta.json` | 7.8 KB |

The four ONNX files total 19.8 MB in fp32. The browser ships a packed copy: the
comment in `indra.js` gives the classifier as 90 KB and the math reader as 4 to
8 MB, and it fetches the math models only when the mode is `auto` and a line is
flagged, or the mode is `on`.

**Reuse in tesseract-gpu.** A Tesseract page needs one function: a line crop in,
a LaTeX string and a confidence out. Its crop-level call would run the classifier
on each line and send only the flagged crops to the decoder. `readMath` in
`indra.js` already does this for Indra's own line boxes, and tesseract-gpu does
not yet call it. The math pack stays a separate fetched artifact, so a page loads
it only when it needs it.

**Licence.** The weights are under the modified OpenRAIL-M licence in the
`LICENSE` file of the Hugging Face repository. Its Attachment A, paragraph 2,
bars use:

- "for any purpose if You (your employer, or the entity you are affiliated with)
  generated more than twenty million US Dollars ($20,000,000) in gross revenue
  in the prior year";
- "for any purpose if You (your employer, or the entity you are affiliated with)
  has raised more than twenty million US dollars ($20,000,000) in total equity or
  debt funding from any source";
- or any product or service "that competes with any product or service offered
  by or made available by Licensor or any of its affiliates".

Both funding and revenue limits lift for personal use or research. The code is
Apache 2.0, and Tesseract is Apache-2.0. The licence follows the weights, so a
package that ships the Indra math pack carries the OpenRAIL-M terms.

## 2. Math recognisers

| Model | Params | Architecture (encoder / decoder / AR?) | Weights | Licence | Benchmark number (source) | ONNX export |
|---|---|---|---|---|---|---|
| Indra math (`datalab-to/indra-ocr`, math models) | unverified (the default non-math models total 7.3M per the card; the math extra is not counted there) | encoder / small decoder, autoregressive (`math_dec_step` is a per-token step) | ONNX, `math_*.onnx` (enc 9.9 MB, dec_init 1.4 MB, dec_step 8.4 MB, cls 0.19 MB) | OpenRAIL-M (modified); code Apache-2.0 | OmniParseBench math tests **55.8** with `--math`; headline 71.4 vs 66.7 off ([indra-ocr README](https://huggingface.co/datalab-to/indra-ocr)) | Yes, native |
| pix2tex / LaTeX-OCR (`lukas-blecher/LaTeX-OCR`) | unverified (~ConvNeXt/ResNet + ViT, roughly 0.1B; **unverified**) | ViT encoder with ResNet backbone / Transformer decoder, autoregressive | PyTorch checkpoint | MIT | Texify's comparison table: BLEU **0.382659** on Texify's arXiv + im2latex benchmark, a biased comparison that the Texify README itself flags ([texify README](https://github.com/VikParuchuri/texify)). No im2latex-100k number in the README (**unverified**) | No official export found |
| Texify (`VikParuchuri/texify`) | unverified (Donut base; **unverified** count) | Donut (Swin encoder / BART-style decoder), autoregressive | PyTorch | Code GPL-3 (**unverified**); weights CC BY-SA 4.0 per the Texify README | Same table: BLEU **0.842349**, METEOR 0.885731, edit distance 0.0651534 ([texify README](https://github.com/VikParuchuri/texify)). Deprecated by its author; functionality moved to surya | No |
| Pix2Text-MFR (`breezedeus/pix2text-mfr`) | unverified (TrOCR-based; encoder ONNX 87.5 MB, decoder 30.1 MB, so ~118 MB fp32) | ViT encoder / Transformer decoder, autoregressive | ONNX (`encoder_model.onnx`, `decoder_model.onnx`) + tokenizer | MIT | No benchmark number on the model card (**unverified**); the Pix2Text article is linked from the card | **Yes**, shipped as ONNX |
| UniMERNet tiny / small / base (`opendatalab/UniMERNet`, `wanderkid/unimernet_*`) | tiny and small are smaller than base; exact counts **unverified**. File sizes: tiny 441 MB, small 773 MB, base 1.3 GB (PyTorch `.pth`, from the repo README) | Hybrid ViT-style encoder / Transformer decoder, autoregressive | PyTorch `.pth` | Model cards say apache-2.0 (HF metadata); repo licence **unverified** | BLEU and CDM figures are in repo images only (`fig1_bleu.jpg`, `fig2_cdm.jpg`), numeric values **unverified**. CDM is the metric the authors recommend ([UniMERNet README](https://github.com/opendatalab/UniMERNet)) | No official export found |
| PP-FormulaNet-S / PP-FormulaNet_plus-S (PaddleOCR; `PaddlePaddle/PP-FormulaNet_plus-S`) | unverified; the "S" variants are the small ones | Hybrid encoder / Transformer decoder, autoregressive | Paddle inference (`inference.pdiparams`) | Apache-2.0 (HF metadata) | En-BLEU table exists on the model card; the exact values were not captured in this survey (**unverified**). Evaluated on UniMERNet's simple and complex sets | Not in repo; conversion from Paddle required (**unverified**) |
| PDFluent formula-recognition-onnx (`PDFluent/formula-recognition-onnx`) | unverified | Encoder / decoder ONNX, autoregressive | ONNX `encoder.onnx`, `decoder.onnx`, `vocab.json` | **AGPL-3.0**, with a separate commercial licence file | None on the card (**unverified**) | Yes, native |

Notes on the table:

- Only Indra and Texify give a number on a named benchmark that this survey could
  read directly. The Texify comparison is run on Texify's own corpus, not on
  UniMER-Test or im2latex-100k.
- CDM is the metric the UniMERNet authors recommend over BLEU. None of the other
  rows report it in text.
- The PDFluent row shows that an ONNX export can carry a copyleft licence. That
  rules the file out for any distributed page.
- No 2025-2026 small formula recogniser beyond these was verified in this pass;
  the search was limited.

## 3. Detection: how math lines get flagged

- **Indra `mathcls.onnx`**: a small line classifier, 191 KB in fp32 and 90 KB in
  the browser pack. It runs on each line crop and flags math lines before the
  reader runs. It is the cheapest option, and it already exists.
- **Layout models**: PP-DocLayout-style detectors label "formula" regions. Their
  sizes are **unverified** in this survey.
- **Heuristic**: a symbol-density and glyph-class check on the Tesseract output,
  for example the share of characters in a math set such as `∑∫√=±^_{}`, or a low
  word-dictionary hit rate. It costs no bytes and no GPU, but it misses formulas
  that look like text.
- For a Tesseract pipeline, run the heuristic first, then the Indra classifier on
  the candidate lines only.

## 4. Recommendation

Ranking criteria: accuracy per megabyte, and new GPU kernel types beyond conv,
LSTM and matmul.

**WebGPU page: Indra's math path.**

- It is the only candidate whose number on a named benchmark (OmniParseBench math
  55.8) is in a source this survey could read, and it already runs in
  `indra-browser`.
- Size: 19.8 MB of fp32 ONNX (enc 9.9 + dec_init 1.4 + dec_step 8.4 + cls 0.19).
  The packed copy in `indra-browser` is 4 to 8 MB for the reader, plus 90 KB for
  the classifier, per the comment in `indra.js`.
- Licence: OpenRAIL-M, with the revenue and funding limits in section 1. A page
  that ships the pack must carry those terms. The repo owner, Oskar Austegard,
  decides whether the packed weights may be published under them.

**Server: Pix2Text-MFR.** It is MIT-licensed, ships as ONNX (~118 MB fp32), and
runs on onnxruntime with no new GPU kernels. Its accuracy is **unverified**. Texify
has the best published BLEU in the table, but its code is deprecated and its
number comes from its own corpus. Use it only if its accuracy holds on this
repository's own formula set. Do not ship PDFluent (AGPL-3.0) in a distributed
product.

**Ranking, accuracy per MB.** Only Indra has a sourced accuracy and a known size,
so it ranks first on that axis. Pix2Text-MFR and Texify cannot be ranked until
their sizes and scores are measured on the same benchmark.

**Next checks, in order (none run yet):**

1. Run Pix2Text-MFR and Indra math on the same 50 formula crops from UniMER-Test
   (the dataset linked in the UniMERNet README), scored with CDM. This fills the
   unverified cells.
2. Count the new kernel types in the `indra-browser` math path against the
   kernels tesseract-gpu already has.
3. Settle the OpenRAIL-M terms for the WebGPU page with the repo owner.
