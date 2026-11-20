# EmbeddingGemma 2's vision encoder as an OCR base: a line-level CTC probe

Google's EmbeddingGemma 2 (`google/embeddinggemma-2`, Apache-2.0) turns text, images, audio and video into one
search vector. Its image side is the 167M-parameter Gemma 4 vision encoder. The question was whether that encoder,
trained for retrieval, still sees individual letters well enough to build an OCR model on, before spending effort
on one.

## Answer

Yes, at this resolution. The encoder keeps letter-level detail. Pooling its 16 px patches into the 48 px image
tokens a language model receives (each token averages 3 x 3 patches) loses much of it.

A small reader, a BiLSTM trained with CTC loss (the standard way to read characters off image columns), was
trained on the frozen encoder's output for synthetic printed text lines. On the 16 px patches its
character error rate (CER) was 1.53% on unseen text in the 9 training fonts <!-- @evidence[[results/result_pretrained.json#seen_fonts.prepool.cer]] -->,
and 2.17% in a font it never saw <!-- @evidence[[results/result_pretrained.json#heldout_font.prepool.cer]] -->.
The bar set before the run was 3% or less in the training fonts. With random encoder weights the same reader got
73.87% wrong <!-- @evidence[[results/result_random.json#seen_fonts.prepool.cer]] -->,
so the reading comes from the trained weights.

On the pooled 48 px tokens the error rate was 8.07% <!-- @evidence[[results/result_pretrained.json#seen_fonts.pooled.cer]] -->,
about five times worse. An OCR model on this encoder should feed its decoder unpooled patches; a larger token
budget might also work, untested.

What would change this: real scans, and pages at the default budget of 280 image tokens. These lines used 22–30
px fonts, roughly body text on a page at the 1,120-token maximum; at 280 tokens letters would be about half that.
One run, one seed.

## Findings

1. The frozen encoder's 16 px patch features read printed lines at 1.53% character error rate (CER) on unseen
   text in seen fonts, 65.05% of lines exactly right <!-- @evidence[[results/result_pretrained.json#seen_fonts.prepool.exact]] -->,
   and 2.17% CER in an unseen font. (`results/result_pretrained.json`)
2. The random-weight control reaches 73.87% CER and gets no line exactly right, so the head alone cannot read
   these features. (`results/result_random.json`)
3. The 3x3-pooled 48 px tokens cost a factor of about five: 8.07% CER in seen fonts and 11.17% <!-- @evidence[[results/result_pretrained.json#heldout_font.pooled.cer]] -->
   in the unseen font. (`results/result_pretrained.json`)
4. The pretrained error curve was still falling at 3,000 steps, from 6.13% <!-- @evidence[[results/result_pretrained.json#curve.0.prepool]] -->
   at step 500 to 1.42% <!-- @evidence[[results/result_pretrained.json#curve.4.prepool]] --> at step 2,500, so these
   are probably not this head's floor (the curve points use the first 400 evaluation lines; the final numbers use
   all 2,000, which is why step 3,000 reads 1.53% after 1.42% at 2,500). Several errors in the sample output are
   doubled letters (`modifiied`, `leastt`), which suggests some of the remaining error is in CTC decoding, not in
   the features; this was not counted.
5. A 300-step run is too short to say anything: both variants sat on CTC's all-blank plateau at 100% CER <!-- @evidence[[results-short/result_pretrained.json#curve.5.prepool]] -->.
   By step 500 the long run was already at 6.13% CER, so the plateau ends somewhere between 300 and 500 steps; a
   quick version of this test should run about 1,000 steps. (`results-short/`)
6. `Gemma4ImageProcessor` resizes every image up to fill its soft-token budget. A 768x48 line becomes 198x12
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
