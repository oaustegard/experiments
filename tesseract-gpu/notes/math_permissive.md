# Permissively licensed math reader for tesseract-gpu

**Question.** Indra OCR's math path (a line classifier that flags math lines,
and a reader that writes them as LaTeX) is the one `notes/math.md` recommends for
the WebGPU page. Its weights are OpenRAIL-M, so they cannot ship under an
Apache-2.0 or MIT stack. Which math path has both weights and code under Apache-2.0,
MIT or BSD, and what does it cost to port?

**Answer in one line.** The page should load Pix2Text-MFR-1.5 (ONNX, about 120 MB
fp32, MIT on its model card) as the reader, with PP-DocLayout-S (4.8 MB, Apache-2.0,
has a `formula` class) to flag lines. The server should run PP-FormulaNet_plus-S
(Apache-2.0, the best published permissive score in this survey, but no ONNX). Both
recommendations are provisional: no permissive reader in this survey has an accuracy
number measured on the same crops as Indra, so the choice is made on licence and size,
not on measured quality.

## How licences were read

"Code" means the `LICENSE` file in the code repository, read from GitHub raw. "Weights"
means the Hugging Face model card's `license:` field (the YAML front matter, or the
`cardData.license` value from the Hub API), plus a `LICENSE` file in the model repo
where one exists. Where a model repo has no `LICENSE` file, the card field is the only
statement of the weights licence, and the table says so. A licence not found in a
primary file is marked **not found**, not assumed.

Files read directly for this note:

- `datalab-to/indra-ocr/LICENSE` (AI PUBS Open RAIL-M, modified; Attachment A read)
- `PDFluent/formula-recognition-onnx/LICENSE` (GNU AGPL-3.0)
- `VikParuchuri/texify` `LICENSE` (GPL-3.0) and `README.md` (weights line: CC BY-SA 4.0)
- `breezedeus/Pix2Text` `LICENSE` (MIT)
- `lukas-blecher/LaTeX-OCR` `LICENSE` (MIT)
- `RapidAI/RapidLatexOCR` `LICENSE` (MIT)
- `PaddlePaddle/PaddleOCR` `LICENSE` (Apache-2.0); `PaddlePaddle/PaddleX` `LICENSE` on `master` (Apache-2.0)
- `opendatalab/UniMERNet` `LICENSE` (Apache-2.0)
- `OleehyO/TexTeller` `LICENSE` (Apache-2.0)
- Model-card front matter read from the Hub: `datalab-to/indra-ocr`, `PaddlePaddle/PP-FormulaNet_plus-S`,
  `PaddlePaddle/PP-FormulaNet-S`, `PaddlePaddle/PP-DocLayout-S`, `PaddlePaddle/PP-DocLayout_plus-L`,
  `breezedeus/pix2text-mfr`, `breezedeus/pix2text-mfr-1.5`, `breezedeus/pix2text-mfd`,
  `breezedeus/pix2text-mfd-1.5`, `wanderkid/unimernet_small`, `PDFluent/formula-recognition-onnx`,
  `inYourOwnBrowser/rapid-latex-ocr-onnx`, `onnx-community/TexTeller-ONNX`, `OleehyO/TexTeller`
- Model repo file listings and byte sizes from the Hub tree API (all sizes below are file sizes in bytes, decimal MB).

## Candidates

Verdict column: **usable** means licence is Apache-2.0, MIT or BSD on code and weights
with a file or card read; **rejected** names the clause that fails.

| # | Candidate (repo) | Code licence (file read) | Weights licence (file read) | Params | Download (fp32 unless said) | Architecture | ONNX | Accuracy, named benchmark and source | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Indra math, `datalab-to/indra-ocr` (`mathcls.onnx`, `math_enc.onnx`, `math_dec_init.onnx`, `math_dec_step.onnx`) | Apache-2.0 (per `notes/math.md`; not re-read here) | OpenRAIL-M (modified), `LICENSE` file read. Attachment A, clauses (a) to (c), bars use by any entity with over $20M revenue in the prior year, (b) over $20M raised in funding, (c) products that compete with Licensor's products. Limits lift only for personal or research use. | Not stated for the math extra. `math_meta.json`: conv blocks (6, 192 channels, kernel 9, squeeze-excite), 2 attention layers (8 heads, FFN 768), 3 decoder layers (8 heads, FFN 768), 652 tokens, max 320 | 19.87 MB total: enc 9.89, dec_init 1.36, dec_step 8.43, cls 0.19 | Conv-attention encoder, small autoregressive decoder, split into a first-step graph and a per-token step graph | Yes, native | OmniParseBench math tests **55.8** with `--math` (Indra README, "Math" table) | **Rejected** (OpenRAIL-M, Attachment A (a) and (b) bind a page or a server operator with that revenue or funding) |
| 2 | PP-FormulaNet_plus-S, `PaddlePaddle/PP-FormulaNet_plus-S` | PaddleOCR `LICENSE`, Apache-2.0, read. PaddleX `LICENSE`, Apache-2.0, read. | Model card `license: apache-2.0`. No `LICENSE` file in the model repo. | About 64M, estimated from `inference.pdiparams` = 256.8 MB assuming fp32 (not read from the model) | 256.8 MB `inference.pdiparams` (Paddle format), plus 3.95 MB `config.json` and 0.5 MB `inference.json` | Hybrid: PPHGNetV2-B4 CNN encoder, Transformer decoder, autoregressive, input 384x384 | **No.** Paddle inference format only. Community ONNX exists only for the L variant (`ningpp/PP-FormulaNet_plus-L-ONNX`, 728 MB, Apache card). None for S found in 3 Hub searches. | En-BLEU **88.71**, Zh-BLEU 53.32, GPU ms 191.69. UniMERNet 85.91 and PP-FormulaNet-S 87.00 in the same card table. Set: UniMERNet simple and complex formulas plus PaddleX's internal set. Source: model card table (vendor's own run, not independent) | **Usable** for the server. Not an ONNX candidate. |
| 3 | PP-FormulaNet-S, `PaddlePaddle/PP-FormulaNet-S` | As row 2 | Model card `license: apache-2.0`; no `LICENSE` file | About 58M, from 231.7 MB `pdiparams` (fp32 estimate) | 231.7 MB `pdiparams` | As row 2 | No | En-BLEU 87.00 (same card table as row 2) | Usable, but row 2 is the better model at about the same size |
| 4 | Pix2Text-MFR-1.5, `breezedeus/pix2text-mfr-1.5` | Pix2Text `LICENSE`, MIT, read | Model card `license: mit` (front matter read). No `LICENSE` file in the model repo. | Not read. The v1.0 card says TrOCR architecture; about 30M estimated from ONNX sizes (fp32 estimate) | `encoder_model.onnx` 87.5 MB + `decoder_model.onnx` 32.0 MB = **119.5 MB** fp32. Tokenizer 0.11 MB | TrOCR (`vision-encoder-decoder`): ViT encoder, Transformer decoder, autoregressive | **Yes**, shipped by the maintainer | **No benchmark number on the card.** v1.5 release notes link a Breezedeus article; not read. Unverified. | **Usable** (licence). Accuracy unknown: the page's first measured task. |
| 5 | Pix2Text-MFR v1.0, `breezedeus/pix2text-mfr` | As row 4 | MIT (card) | As row 4 | 117.6 MB (ONNX) | As row 4 | Yes | No number on card | Superseded by row 4 on the same repo family. |
| 6 | Pix2Text-MFD-1.5, `breezedeus/pix2text-mfd-1.5` (detector) | As row 4 | MIT (card) | Not read | `pix2text-mfd-1.5.onnx` 80.3 MB (the older `pix2text-mfd` is 44.6 MB) | Detector (card says "based on CnSTD"; architecture not read) | Yes | None on card | Usable licence, but 80 MB just to find formulas. Prefer row 7. |
| 7 | PP-DocLayout-S, `PaddlePaddle/PP-DocLayout-S` (layout detector with a `formula` class) | PaddleOCR `LICENSE`, Apache-2.0, read | Model card `license: apache-2.0`. No `LICENSE` file. | Not read | `inference.pdiparams` 4.80 MB, `inference.json` 0.34 MB | GFL detector, 480x480 input. `config.json` `label_list` includes `formula` (index 7) and `formula_number` (index 19) | No (Paddle format). Conversion needed. | Layout mAP not read in this survey (unverified) | **Usable** as the page's formula flagger. Caveat: it sees the page, not one line crop. |
| 8 | PP-DocLayout_plus-L, `PaddlePaddle/PP-DocLayout_plus-L` | As row 7 | Model card `license: apache-2.0` | Not read | `inference.pdiparams` 129.3 MB (RT-DETR-L) | RT-DETR-L | No | Card has an mAP table; values not captured | Usable licence, server option for flagging |
| 9 | pix2tex / LaTeX-OCR, `lukas-blecher/LaTeX-OCR` (no HF repo found; checkpoints on GitHub releases, not read) | `LICENSE`, MIT, read | **Not found.** No separate weights licence in the README. Checkpoints not opened. | Not read | Not read | ViT encoder with ResNet backbone, Transformer decoder, autoregressive (README "Model" section) | Not in the repo | README table, one row: BLEU 0.88, normed edit distance 0.10, token accuracy 0.60. Dataset and split not named in the README. | Code usable. Weights licence **not found**, so not usable for a stated permissive weights licence. |
| 10 | RapidLatexOCR ONNX, `inYourOwnBrowser/rapid-latex-ocr-onnx` (re-host of `RapidAI/RapidLatexOCR` weights) | `RapidAI/RapidLatexOCR` `LICENSE`, MIT, read. Repo says the reasoning code is modified from LaTeX-OCR. | Re-host card `license: mit`, which says the files are unmodified copies of RapidAI's ONNX weights. RapidAI's repo gives no separate weights licence. | Not read | `encoder.onnx` 89.0 MB + `decoder.onnx` 51.0 MB + `image_resizer.onnx` 39.0 MB = **179 MB** | pix2tex-derived: ResNet+ViT encoder, Transformer decoder, autoregressive, plus a size-bucket predictor | Yes (ONNX, converted by RapidAI) | None on card. Inherits pix2tex's own number (row 9), not re-measured | **Usable** on the stated licence chain (MIT code, MIT re-host card). Largest of the ONNX readers. |
| 11 | Texify, `VikParuchuri/texify` (Donut-based) | `LICENSE`, **GPL-3.0**, read | `README.md` line 173, read: "Model weights are licensed under the CC BY-SA 4.0 license." | Not read | Not read | Donut (Swin encoder, BART-style decoder), autoregressive | No | BLEU 0.842349, METEOR 0.885731, edit distance 0.0651534 on Texify's own arXiv and im2latex sample (README table; the README calls it biased). | **Rejected**: GPL-3.0 code and CC BY-SA weights. Deprecated by its author (README line 2). |
| 12 | UniMERNet small, `wanderkid/unimernet_small` (weights mirror; official repo is `opendatalab/UniMERNet`) | `opendatalab/UniMERNet` `LICENSE`, Apache-2.0, read | Model card `license: apache-2.0` (front matter read). No `LICENSE` in the mirror. | Not stated in the card. `config.json`: encoder 12 layers, decoder 8 layers, 16 heads, FFN 3072, vocab 50,000 | `unimernet_small.pth` **810.3 MB** (the README calls this 773 MB, which is MiB). Tiny 441 MB and base 1.3 GB per README. | Donut-derived Swin/ViT encoder, Transformer decoder, autoregressive | No | Charts only in README (`fig1_bleu.jpg`, `fig2_cdm.jpg`). Texo's paper (row 13) reports CDM on UniMER-Test for UniMERNet-T at 107M params: SPE 0.991, CPE 0.949, SCE 0.938, HWE 0.933. | Usable licence. Too large for a page. Reference point for accuracy. |
| 13 | Texo (arXiv 2602.17189, Feb 2026), 20M params | **Not found.** Paper gives no code or weights URL in its links. | **Not found.** | 20M (paper) | Not read | HGNetV2-B4 encoder (as PP-FormulaNet-S) with a distilled 687-token vocabulary, Transformer decoder, autoregressive. Paper: ONNX export and Transformers.js in browser. | Paper says yes; no file found | UniMER-Test CDM, paper Table II: SPE 0.958, CPE 0.825, SCE 0.882, HWE 0.902. PP-FormulaNet-S in the same table: 0.949, 0.678, 0.856, 0.818. Source: arXiv paper, authors' run. | **Not usable yet.** Best small-model number, but no licence file exists to read. Ask the authors or wait for a release. |
| 14 | TexTeller, `OleehyO/TexTeller` (code) and `onnx-community/TexTeller-ONNX` (ONNX) | `OleehyO/TexTeller` `LICENSE`, Apache-2.0, read | Model card `license: apache-2.0` on both repos (front matter read) | Not read | `decoder_model.onnx` 908.7 MB fp32; `q4` 198.6 MB; `q4f16` 152.2 MB; `int8` 228.4 MB. Encoder file not read. | ViT-based encoder, Transformer decoder (per Texo paper's description) | Yes (onnx-community) | Not read in this survey | Usable licence. Quantized decoder too big for the page; a server option. |
| 15 | PDFluent formula-recognition-onnx, `PDFluent/formula-recognition-onnx` | `PDFluent/formula-recognition-onnx` `LICENSE`: **AGPL-3.0**, read. A `LICENSE-COMMERCIAL` file also exists (not read). | Card `license: agpl-3.0`; `PROVENANCE.md` says trained from scratch on CC BY and CC0 papers | Not read | `encoder.onnx` 18.4 MB + `decoder.onnx` 18.2 MB = 36.6 MB | Encoder-decoder ONNX, autoregressive | Yes, native | Held-out evaluation described in `PROVENANCE.md`; numbers not extracted | **Rejected**: AGPL-3.0 (weights and code, per card). A commercial licence is a separate purchase. |

Notes on the table:

- Row 1 is the only candidate with a sourced math accuracy and a known size, and it
  is the one the permissive rule excludes. The number is kept so a later measurement
  has a target.
- The PP-FormulaNet numbers and the Texo numbers come from different metrics and sets
  (BLEU on UniMERNet plus an internal set, versus CDM on UniMER-Test). They cannot be
  ranked against each other.
- The `notes/math.md` line "Pix2Text-MFR ... ONNX (~118 MB)" refers to v1.0. Row 4 is the
  current MFR. Its size is 119.5 MB.
- Params for rows 2, 3 and 4 are estimates from file size, assuming fp32 weights. They
  are not read from the models.

## Detection: can math lines be flagged without a learned model?

For a first version, yes, with a stated risk. Two options:

1. **Heuristic on Tesseract output.** Mark a line as math when its characters are
   mostly outside the prose set, for example a high share of `^ _ \ { } = + - < > ∑ ∫ √ ± ≤ ≥`
   and Greek letters, or when digits and single letters alternate with no word
   dictionary hits. Cost: no bytes, no GPU. Limits: it misses formulas that read as
   prose (inline `x_i` in a sentence), and it fires on tables and code. The thresholds
   have not been measured on any page in this repo.
2. **Formula-class detector.** PP-DocLayout-S (4.8 MB, row 7) already has a `formula`
   label. It is trained for page layout, not line crops, so it flags regions. The
   page would map its boxes onto Tesseract lines. Its own mAP for that class was not
   read.

Recommended order: run the heuristic everywhere (free), and the detector only when the
page has a candidate line or the user turns math on. The heuristic has not been
measured, so the first step is a labelled sample of lines from this repo's data to set
its thresholds and report precision and recall. No number for either method is given
here.

## Kernels a port needs beyond conv, LSTM and matmul

The existing `web/kernels/` set covers conv, the xLSTM and yLSTM steps and the output
projection. Each reader below needs more:

| Model | Attention (self and cross) | LayerNorm | GELU or other activation | Softmax | KV cache | Beam search | Other |
|---|---|---|---|---|---|---|---|
| Pix2Text-MFR-1.5 (TrOCR, row 4) | Yes, ViT encoder and decoder self and cross | Yes | Likely GELU (not read from the graph) | Yes | Needed for speed (decoder is autoregressive; not measured) | `generation_config.json` read for the decode settings; beam width not read | Patch embedding (conv), positional embeddings, token-embedding lookup |
| RapidLatexOCR (row 10) | Yes, ViT encoder, decoder | Yes | Not read | Yes | As row 4 | Greedy or beam, not read | ResNet backbone (conv, batch norm) and the size-bucket predictor |
| PP-FormulaNet_plus-S (row 2) | Yes, Transformer decoder | Yes | Not read | Yes | As row 4 | Config's `UniMERNetDecode`; beam not read | HGNetV2-B4 backbone: conv, batch norm and pooling, no attention in the encoder (per Texo paper's description, not read in the Paddle graph) |
| Indra math (row 1, for comparison) | Yes, 2 encoder attention layers and decoder | Likely (not read) | Not read | Yes | Per-token step graph split, so a cache is likely (inferred, not read) | Greedy likely (not read) | Squeeze-excite gating in conv blocks |

The attention, LayerNorm and softmax kernels are the common cost. Every reader
above needs them, so one kernel set covers all of them.

## Recommendations

**(a) WebGPU page: Pix2Text-MFR-1.5 as the reader, PP-DocLayout-S as the flagger.**

- Reader: `breezedeus/pix2text-mfr-1.5` ONNX, encoder 87.5 MB plus decoder 32.0 MB,
  about 119.5 MB fp32 (about half that if converted to fp16, not tested). Licence:
  `license: mit` on the model card, and the Pix2Text code `LICENSE` (MIT) read
  directly. The model repo has no `LICENSE` file, so the card field is the weights
  statement.
- Flagger: `PaddlePaddle/PP-DocLayout-S`, 4.8 MB, `license: apache-2.0` on the card,
  and PaddleOCR's `LICENSE` (Apache-2.0). It needs a Paddle-to-ONNX conversion that
  has not been run.
- Heuristic (0 bytes) runs first as a fallback when the flagger is not loaded.
- Cost: the reader needs attention, LayerNorm, softmax and a KV cache in WGSL. That is
  more kernel work than the flagger.

**(b) Server: PP-FormulaNet_plus-S for the reader, PP-DocLayout-S or PP-DocLayout_plus-L for the flagger.**

- Reader: `PaddlePaddle/PP-FormulaNet_plus-S`, 256.8 MB `pdiparams`, `license: apache-2.0`
  on the card and PaddleOCR `LICENSE` (Apache-2.0). It has the best published
  permissive score in this survey (En-BLEU 88.71, card table), though that table is the
  vendor's own run.
- Runtime: PaddlePaddle or PaddleOCR in Python. No WGSL work. The server can take the
  size. It has no ONNX, so the ONNX route is not available without conversion.
- Flagger: `PaddlePaddle/PP-DocLayout_plus-L` (129 MB) for the server if the S model misses
  formula regions; otherwise the S model at 4.8 MB.

**Plan, in one paragraph.** Start with the server, because it needs no kernel work and
has the best published permissive score: run PP-FormulaNet_plus-S on the formula regions
that PP-DocLayout-S flags, and score it on the same 50 formula crops that Indra's
`--math` run used, reporting the Indra numbers as the target. Measure the heuristic's
precision and recall on a labelled sample of lines from this repo's data at the same
time. Next, convert Pix2Text-MFR-1.5 and run it through onnxruntime on the same crops
to fill the accuracy cell that is empty today; if it matches the server model on CDM,
it becomes the WebGPU reader. Only then write the WGSL attention, LayerNorm, softmax and
KV-cache kernels for the page. Until a measured number exists, the page ships with the
heuristic flag and no math reader, and the page says so. Do not ship Indra's math pack
(OpenRAIL-M), Texify (GPL-3.0 code, CC BY-SA weights) or PDFluent (AGPL-3.0).

## Open items

1. No accuracy number for Pix2Text-MFR-1.5 exists in a primary source. Measure it.
2. Texo has no licence file and no release found. Contact the authors before using it.
3. The pix2tex weights licence is not stated anywhere read here. Check the GitHub
   release page before using RapidLatexOCR's weights on a licence basis.
4. PP-FormulaNet_plus-S has no ONNX export found. Converting it is a Paddle task.
5. Parameter counts for rows 2 to 4 are estimates from file sizes (fp32 assumed).
6. `notes/math.md` still recommends Indra for the WebGPU page. That recommendation
   fails the permissive rule and should be replaced by the one above.
