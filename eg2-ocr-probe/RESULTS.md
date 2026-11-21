# EmbeddingGemma 2's vision encoder as an OCR base: a line-level CTC probe

Google's EmbeddingGemma 2 (`google/embeddinggemma-2`, Apache-2.0) turns text, images, audio and video into one
search vector. Its image side is the 167M-parameter Gemma 4 vision encoder. The question was whether that encoder,
trained for retrieval, still sees individual letters well enough to build an OCR model on, before spending effort
on one.

## Answer

Not worth building on, for printed text or handwriting. The encoder keeps letters, but a free model already reads
better than a reader on its frozen features in both cases.

A small reader, a BiLSTM trained with CTC loss (the standard way to read characters off image columns), was
trained on the frozen encoder's 16 px patch features.

- **Printed lines (synthetic):** character error rate (CER) 1.53% <!-- @evidence[[results/result_pretrained.json#seen_fonts.prepool.cer]] -->,
  against 73.87% <!-- @evidence[[results/result_random.json#seen_fonts.prepool.cer]] --> for random encoder weights,
  so the encoder does keep letters. Tesseract 5 reads the same lines at 0.26% <!-- @evidence[[results/baselines.json#tesseract.seen_fonts.cer]] -->
  with no training.
- **Handwriting (IAM test, 2,915 lines):** CER 14.70% <!-- @evidence[[results-iam/iam_result.json#test_cer_norm]] -->.
  Microsoft's TrOCR-small (62M, MIT licence, trained on IAM) gets 6.16% <!-- @evidence[[results/iam_baselines.json#trocr_small.cer_norm]] -->;
  Tesseract fails at 51.03% <!-- @evidence[[results/iam_baselines.json#tesseract.cer_norm]] -->.

Fine-tuning the 167M encoder end to end and adding a real decoder would narrow the handwriting gap, but the
likely outcome is a model about three times TrOCR-small's size that matches it, under a licence no more permissive.

What would change this: a use that needs OCR and EG2's shared search space from one model, or an input where both
TrOCR and Tesseract fail.

## Findings

1. On IAM handwriting the frozen-encoder reader reaches 14.70% CER (2.68% <!-- @evidence[[results-iam/iam_result.json#test_exact_norm]] --> of lines exact)
   against TrOCR-small's 6.16% (40.65% <!-- @evidence[[results/iam_baselines.json#trocr_small.exact_norm]] --> exact).
   The reader fits its training lines far better (7.63% <!-- @evidence[[results-iam/iam_result.json#train500_cer_raw]] --> CER
   on 500 of them) and its test curve was flat from step 1,500, so more steps on 6,482 lines would not close the
   gap. Many errors are doubled letters (`changgees`, `liveely`), so a better decoder would recover some.
   (`iam_probe.py`, `results-iam/`, `iam_baselines.py`)
2. Tesseract 5.3.4 (`--psm 7`, English LSTM model) reads the same 2,000 evaluation lines at 0.26% CER,
   93.65% of lines exactly right <!-- @evidence[[results/baselines.json#tesseract.seen_fonts.exact]] -->, and the held-out font at 0.13% <!-- @evidence[[results/baselines.json#tesseract.heldout_font.cer]] -->,
   about six times fewer errors than the best EG2 reader, at about 0.1 s a line per CPU core. PaddleOCR's
   PP-OCRv3 recognizer (its Chinese model, which drops many English spaces) scores 3.09% <!-- @evidence[[results/baselines.json#ppocr_v3.seen_fonts.cer]] -->.
   (`baselines.py`, `results/baselines.json`)
3. The frozen encoder's 16 px patch features read printed lines at 1.53% character error rate (CER) on unseen
   text in seen fonts, 65.05% of lines exactly right <!-- @evidence[[results/result_pretrained.json#seen_fonts.prepool.exact]] -->,
   and 2.17% CER in an unseen font. (`results/result_pretrained.json`)
4. The random-weight control reaches 73.87% CER and gets no line exactly right, so the head alone cannot read
   these features. (`results/result_random.json`)
5. The 3x3-pooled 48 px tokens cost a factor of about five: 8.07% CER in seen fonts and 11.17% <!-- @evidence[[results/result_pretrained.json#heldout_font.pooled.cer]] -->
   in the unseen font. (`results/result_pretrained.json`)
6. The pretrained error curve was still falling at 3,000 steps, from 6.13% <!-- @evidence[[results/result_pretrained.json#curve.0.prepool]] -->
   at step 500 to 1.42% <!-- @evidence[[results/result_pretrained.json#curve.4.prepool]] --> at step 2,500, so these
   are probably not this head's floor (the curve points use the first 400 evaluation lines; the final numbers use
   all 2,000, which is why step 3,000 reads 1.53% after 1.42% at 2,500). Several errors in the sample output are
   doubled letters (`modifiied`, `leastt`), which suggests some of the remaining error is in CTC decoding, not in
   the features; this was not counted.
7. A 300-step run is too short to say anything: both variants sat on CTC's all-blank plateau at 100% CER <!-- @evidence[[results-short/result_pretrained.json#curve.5.prepool]] -->.
   By step 500 the long run was already at 6.13% CER, so the plateau ends somewhere between 300 and 500 steps; a
   quick version of this test should run about 1,000 steps. (`results-short/`)
8. `Gemma4ImageProcessor` resizes every image up to fill its soft-token budget. A 768x48 line becomes 198x12
   patches at the 280-token default. To control resolution, patchify by hand; `probe.py` `patchify()` matches the
   processor's `do_resize=False` path. (`probe.py` `verify_patchify`)

## Method

- **Model:** the vision tower of `google/embeddinggemma-2` in transformers 5.23.0, frozen, fp32, last hidden state
  before the pooler.
- **Data:** synthetic printed lines, 48 x 1152 px (3 x 72 patches), grey text on a light background, 22–30 px
  font size, which is body text on a page at the 1,120-token budget. Text is wikitext-103 spans of 15–60
  characters (validation split for training, test split for evaluation), printable ASCII. Nine Google Fonts for
  training (Noto Serif, Source Code Pro, Merriweather, Lato, Open Sans, EB Garamond, Courier Prime, Roboto Slab,
  PT Sans); Crimson Text held out. Augmentation: rotation of up to 1°, Gaussian blur, pixel noise.
- **Reader:** LayerNorm, Linear to 512, 2-layer BiLSTM (256 per direction), Linear to CTC classes (95 characters
  plus blank). The prepool reader takes each 16 px patch column (3 rows concatenated, 2,304 dims) and emits 3 frames
  per column. The pooled reader takes the 24 averaged 48 px tokens and emits 9 frames per token. Both get 216
  frames per line.
- **Training:** 3,000 steps at batch 64, AdamW at 1e-3 with warmup and cosine decay, fresh lines every step. Both
  readers on the pretrained encoder trained in one run; the random-weight control trained the prepool reader only.
- **Metric:** greedy CTC decoding, character error rate (edit distance over reference characters) and exact-line
  rate, on 2,000 lines in seen fonts and 1,000 in the held-out font.
- **Compute:** Kaggle, 2x T4, the two variants in parallel, one per GPU: 84 min for the pretrained run and 118 min
  for the control, about 2 h of wall-clock including setup. The control is slower although it trains one reader,
  probably because both processes rendered their training lines on the same 4 CPUs; this was not profiled.

## Log

### 2026-11-21 — handwriting (IAM)

Asked: try handwriting, with the same goal of deciding whether more effort is worth it. Data: Teklia/IAM-line
(MIT; 6,482 train and 2,915 test lines, normalised to 128 px tall). `iam_probe.py` (Kaggle T4,
`austegard/eg2-ocr-iam`) resizes each line to 64 px tall keeping its aspect ratio (4 patch rows, median width
880 px), encodes every line once with the frozen tower and caches the features (312 s), then trains the same
BiLSTM CTC head for 2,000 steps at batch 32 on the cache (115 s). `iam_baselines.py` runs Tesseract on the
original 128 px images and TrOCR-small-handwritten with greedy decoding on CPU. CER is on IAM text with its
tokenised punctuation spacing closed up on both sides (`norm`), so engines that print `it's` are not charged for
IAM's `it 's`; raw CER differs by 0.4 points or less for every engine.

| engine | test CER | exact lines | where |
|---|---|---|---|
| TrOCR-small-handwritten (62M) | 6.16% | 40.65% | CPU, 473 s |
| EG2 encoder + BiLSTM CTC, frozen | 14.70% | 2.68% | T4, 7 min |
| Tesseract 5.3.4 `--psm 7` | 51.03% | 0.10% | CPU, 78 s |

Test CER on the first 500 lines by step: 250 28.3%, 500 19.84%, 1,000 15.97%, 1,500 14.56%, 2,000 14.40%.

Reading at the time: stop here too. The gap to TrOCR-small is 2.4x, against 6x to Tesseract on print, and closing
it needs end-to-end fine-tuning of a model three times TrOCR-small's size, to arrive about where TrOCR already is.

### 2026-11-20 — baselines on the same lines (`baselines.py`)

Asked: with the first goal of deciding whether more effort is worth it, take the least-effort next step. The
cheapest decider was whether existing free engines already beat the probe on its own evaluation lines; if they
do, an EG2-based OCR model has nothing to win on printed text. `baselines.py` regenerates the evaluation sets
(the first 12 texts match the Kaggle run's, 12 of 12) and runs each engine on CPU, 4 processes.

| engine | seen fonts CER | held-out font CER | exact lines (seen) | wall-clock, 2,000 lines |
|---|---|---|---|---|
| Tesseract 5.3.4, `--psm 7` | 0.26% | 0.13% | 93.65% | 58 s |
| PP-OCRv3 rec (rapidocr, Chinese model) | 3.09% | 4.31% | 60.4% | 159 s |
| EG2 encoder + BiLSTM CTC, 16 px patches | 1.53% | 2.17% | 65.05% | (GPU) |

Reading at the time: stop. The decision rule from the first round passed, but against the wrong comparison;
clean synthetic printed lines are Tesseract's best case, and the probe was the only remaining argument for EG2.

### 2026-11-20 — first run, 3,000 steps (`austegard/eg2-ocr-probe` v1)

Asked: could EmbeddingGemma 2 be turned into an OCR model; run the cheapest test that decides it. Decision rule set
before the run: pretrained prepool CER in the training fonts at or below 3% means the encoder keeps glyphs and building a small
image-to-text model on it is worth pursuing; a CER near the random control's means it does not.

| reader | seen fonts CER | held-out font CER | exact lines (seen) |
|---|---|---|---|
| pretrained, 16 px patches | 1.53% | 2.17% | 65.05% |
| pretrained, 48 px pooled | 8.07% | 11.17% | 18.6% |
| random weights, 16 px patches | 73.87% | 74.31% | 0% |

Learning curve on the first 400 evaluation lines (pretrained prepool / pooled): step 500 6.13% / 17.81%, 1,000 3.34% / 11.92%,
1,500 2.37% / 10.13%, 2,000 1.63% / 9.02%, 2,500 1.42% / 8.38%.

What went wrong. The run was sized to settle the question, not to answer it quickly: two encoders at 3,000
steps in one push, running in parallel for 84 and 118 minutes, with no visible progress, because the worker processes wrote their logs
to files and Kaggle's live log stream only carries the parent's stdout. `probe.py` now relays worker output, and
the hub's `scripts/kaggle_job.py logs` reads that stream. A 300-step rerun (`eg2-ocr-probe-short`: `probe.py` with defaults of 300 steps, 400 evaluation lines and a
progress line every 25 steps; 10 and 12 minutes) sat on the CTC
blank plateau at 100% CER, so the quick version of this test needs about 1,000 steps (half an hour on a T4), not 300.
