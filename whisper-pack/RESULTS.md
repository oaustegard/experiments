# RESULTS: Whisper from quantized weight packs on hand-written WebGPU kernels

[Anything to Text](https://austegard.com/web-utilities/anything-to-text.html) transcribes audio in the browser
with OpenAI's Whisper, run by transformers.js from ONNX files. This experiment asks whether Whisper can be packed
the way `gliner-browser` and `indra-browser` packed their models: weights stored as small integers, run by our own
WebGPU kernels. It measures what each packing scheme costs in word error rate (WER), builds the engine, and
compares download size and speed with what the page used before. The demo is
[live](https://austegard.com/experiments/whisper-pack/web/). Whisper's weights are MIT-licensed.

## Answer

Yes; it is now the page's default on WebGPU. Whisper runs from "packs": weights stored as small integers, read
by hand-written WebGPU kernels. Each size has an English-optimized pack and an all-languages pack.

For large-v3-turbo the English pack is a 412 MB download <!-- @evidence[[results/headline.json#turbo_en_gzip_mb]] -->
and the all-languages pack 493 MB, where the ONNX files the page used are 759.1 MB. <!-- @evidence[[results/headline.json#page_turbo_mb]] -->

Against the same model at full precision, turbo's English pack changes English word error rate (WER) by +0.04 points (95% interval −0.13 to +0.22) <!-- @evidence[[results/headline.json#turbo_r15_en_diff]] -->
and costs +0.79 points on four other languages; the all-languages pack costs +0.15 there, an interval that
includes zero.

It is also faster than transformers.js. On one Mac, from raw audio to timestamped text with the page's settings on
both sides, turbo's English pack took 0.8 s on a 7 s clip <!-- @evidence[[results/headline.json#fair_en_7s_s]] -->
where transformers.js with the ONNX files took 1.29 s, <!-- @evidence[[results/headline.json#fair_tjs_7s_s]] -->
and 1.69 s on a 28 s clip against 2.69 s. <!-- @evidence[[results/headline.json#fair_en_28s_s]] -->

The saving is in the download, not in memory: on the GPU the turbo English pack occupies 815 MB, because the engine
keeps each weight in one byte.

Not yet measured: other machines, and the WER of the ONNX files.

## Findings

Notation: "codes in ±15" means each weight is stored as an integer from −15 to 15 (5 bits) times a scale shared by
its matrix row; ±31 is 6 bits and ±63 is 7. English is LibriSpeech; "other languages" are Spanish, German, French
and Norwegian from FLEURS. Changes are WER percentage points against full precision.

1. **Turbo keeps its English accuracy at codes in ±15 and loses accuracy on other languages.**
   English changes by +0.04 <!-- @evidence[[results/headline.json#turbo_r15_en_diff]] -->
   and the other languages by +0.79, with an interval from +0.47 to +1.12. <!-- @evidence[[results/headline.json#turbo_r15_fleurs_diff]] -->
   With a 4-bit token table and matrices at ±31, English changes by +0.02 <!-- @evidence[[results/headline.json#turbo_t4r31_en_diff]] -->
   and the other languages by +0.15. <!-- @evidence[[results/headline.json#turbo_t4r31_fleurs_diff]] -->
   (Round 1, `results/table-whisper-large-v3-turbo-*.json`)
2. **English test sets hide the cost on other languages.**
   Whisper-small at ±31 changes English by +0.01 <!-- @evidence[[results/headline.json#small_r31_en_diff]] -->
   and the other languages by +0.77. <!-- @evidence[[results/headline.json#small_r31_fleurs_diff]] -->
   At ±63 it changes both by +0.02. <!-- @evidence[[results/headline.json#small_r63_fleurs_diff]] -->
   (Round 1, `results/table-whisper-small-*.json`)
3. **Four bits per weight needs more than one scale per row.**
   With codes in ±7 and one scale per row, turbo's English WER is 142.73. <!-- @evidence[[results/headline.json#turbo_r7_en_wer]] -->
   With one scale per 32 weights, each chosen to minimise error, turbo loses 0.21 points on English <!-- @evidence[[results/headline.json#turbo_g32r7_en_diff]] -->
   and 0.53 on the other languages. <!-- @evidence[[results/headline.json#turbo_g32r7_fleurs_diff]] -->
   The ONNX q4 files use a scheme of that second kind; their own WER was not measured. (Round 1)
4. **A 4-bit token table costs Whisper's small models accuracy.**
   Combined with matrices at ±63, it costs base 0.8 points on English <!-- @evidence[[results/headline.json#base_rec_en_diff]] -->
   and tiny 1.47. <!-- @evidence[[results/headline.json#tiny_rec_en_diff]] -->
   The same table recipe cost the text encoders in `gliner-browser` and `browser-decider` nothing; in Whisper the
   table is also the output projection, so every predicted token passes through it. (Round 1)
5. **OpenAI's English-only checkpoints score better on English at ±63 and break sooner below it.**
   At ±63, `small.en` scores 4.68 WER <!-- @evidence[[results/headline.json#small_en_r63_wer]] -->
   and multilingual small 5.37; <!-- @evidence[[results/headline.json#small_r63_wer]] -->
   whether that gap exceeds sampling noise was not tested.
   At ±15, `small.en` loses 4.56 points <!-- @evidence[[results/headline.json#small_en_r15_diff]] -->
   where multilingual small loses 0.43. <!-- @evidence[[results/headline.json#small_r15_en_diff]] -->
   The English packs of tiny, base and small are these checkpoints at ±63; turbo has no English-only checkpoint,
   so its English pack is the multilingual model at ±15. (Round 2)
6. **The engine reproduces Hugging Face's output.**
   In headless Chromium, 4 of 4 test cases give the same tokens as a NumPy reference that itself equals Hugging Face's `generate`. <!-- @evidence[[results/headline.json#parity_cases]] -->
   With timestamps on, 3 of 3 clips give Hugging Face's tokens, timestamps included. <!-- @evidence[[results/headline.json#ts_cases]] -->
   On the Mac both turbo packs give the reference's tokens. (Round 3, `test/check.mjs`, job `whisper-pack-turbo-1`)
7. **Turbo's encoder takes 457 ms per 30 s window on the Mac, and a decoded token 7.9 ms.** <!-- @evidence[[results/headline.json#mac_turbo_encode_ms]] -->
   The matrix-multiply kernel reads each packed group of four weights once and applies it to 8 rows.
   The attention kernel keeps its 1,500 scores in a local array, so each audio frame is scored once.
   A version without those two choices took 1,932 ms. <!-- @evidence[[results/headline.json#mac_turbo_encode_before_ms]] -->
   (Round 3, jobs `whisper-pack-turbo-1`, `-turbo-2`)
8. **In the decoder, attention over the audio runs one GPU thread per frame.**
   A decoded token takes 5.1 ms on tiny that way; <!-- @evidence[[results/headline.json#mac_tiny_ms_per_token]] -->
   with one thread per attention head scanning all 1,500 frames it took 28 ms. (Round 3, jobs `whisper-pack-time-2`, `-time-3`)
9. **The spectrogram, computed on the CPU, was half of a short clip's time until it skipped work that adds nothing.**
   Each of the 128 mel filters was summed over all 201 frequency bins, though it is non-zero on a few, and the
   silence that pads audio to 30 s went through the Fourier transform frame by frame. Skipping both changes no output
   value and took the 7 s clip's spectrogram on the Mac from 731 ms to 103 ms. <!-- @evidence[[results/headline.json#mel_7s_after_ms]] -->
   (Round 5, `test/mel_ab.mjs`, job `whisper-pack-fair-4`)

## Method

**Packing schemes.** `run/quant.py` rewrites a Hugging Face Whisper model with the values a pack would decode to:
integer codes times an fp16 scale. A scheme names a code range per part (token table, encoder matrices, decoder
matrices, conv stem). `r63` means codes in ±63 with one scale per output row; `g32r7` means ±7 with one scale per
32 weights. Biases, layer norms and the decoder's position table stay fp16; the encoder's position table is
regenerated.

**Accuracy.** `run/sweep.py` transcribes with greedy decoding, no timestamps and the language given, and scores
WER after Whisper's own text normalizer. English is 300 utterances each from LibriSpeech test-clean and test-other,
pooled. Other languages are 100 utterances each of Spanish, German, French and Norwegian from FLEURS, pooled
(Japanese was run and scored by character error rate; it is in the result files and not in the pooled figure).
Each change is against the same model at full precision, with a 95% paired bootstrap over utterances
(`run/table.py`). Tiny, base and their English-only versions ran on 4 CPUs; small and turbo on Kaggle T4s.

**Pack and reference.** `run/pack.py` writes `weights.bin` and `manifest.json` and asserts that every tensor decodes
bit for bit to what `run/quant.py` produced. `run/ref.py` reads a pack with NumPy only and decodes greedily;
`run/check_ref.py` requires its tokens to equal Hugging Face's `generate`.

**Engine.** `web/whisper.js` holds matrices on the GPU as int8 with one scale per row, and the token table once,
for both the embedding and the output projection. `web/mel.js` computes the log-mel spectrogram, `web/tokenizer.js`
decodes tokens, and `web/transcribe.js` adds language detection, Whisper's timestamp rules, a repeated 3-gram ban
and the seek loop for audio longer than 30 s. `test/check.mjs` compares the engine with the reference in headless
Chromium on SwiftShader, a software GPU whose timings mean nothing.

**Speed.** Timings come from GPU Bridge jobs, which run a script in a tab on Oskar's Mac (Apple Metal, Chrome 157) and
return its output. The comparison with transformers.js (job `whisper-pack-fair-4`) runs both engines on the same
7.28 s and 28 s clips from raw 16 kHz audio with the page's settings: language given, timestamps on, a ban on
repeated 3-grams. Each figure is the median of three runs after a warm-up. Our modules load from a commit-pinned
jsDelivr address, because the bridge worker caches modules by URL for as long as the tab is open. Encoder and
per-token times are from jobs `whisper-pack-turbo-1`, `-turbo-2`, `-time-2` and `-time-3`.

**Sizes of what the page used.** Encoder plus merged decoder from the Hugging Face file listing: turbo q4 759.1 MB;
base, small and tiny with an fp32 encoder and fp16 decoder 187.2, 661.4 and 92.5 MB.

## Log

### Round 1, 2026-11-21: prior art and the accuracy sweep

Upstream check before building: the smallest shipped turbo found was MLX 4-bit at 463.7 MB (Apple only) and ONNX
q4f16 at 563.5 MB; no quantized turbo under 450 MB running on WebGPU in a browser turned up in about 8 searches.
A from-scratch TypeScript and WGSL Whisper exists (benderrodriguez/whisper-large-v3-turbo-webgpu), with fp16
weights of about 1.6 GB. Published 4-bit results (Edge-ASR, arXiv 2507.07877) say small models degrade more.

Pooled WER change against full precision, English then four FLEURS languages where run:

| scheme | tiny | base | small | turbo |
|---|---|---|---|---|
| ±63 everything | +1.05 | +0.11 | +0.02 / +0.02 | −0.04 / +0.05 |
| ±31 everything | +0.59 | +0.18 | +0.01 / +0.77 | −0.03 / +0.26 |
| ±15 everything | +7.32 | +2.58 | +0.43 / +1.98 | +0.04 / +0.79 |
| 4-bit table, ±63 matrices | +1.47 | +0.80 | +0.04 / +0.84 | +0.03 / +0.07 |
| 4-bit table, ±31 matrices | +3.47 | +1.04 | +0.12 / +1.34 | +0.02 / +0.15 |
| ±7, one scale per 32, clip search | +8.80 | +5.89 | +0.20 / +3.36 | +0.21 / +0.53 |
| ±7, one scale per row | +73.47 | +104.34 | +3.04 / +18.26 | +139.57 / +111.72 |

Full tables with intervals are `results/table-*.json`. FLEURS for tiny and base is noisy (full-precision WER 44%
and 29%) and does not separate ±63 from full precision. Kaggle GPU time for the whole experiment: 5.4 hours.
<!-- @evidence[[results/headline.json#kaggle_gpu_hours]] -->

Oskar, after the small results: "We should have two versions for the page, a default optimized one for English
and regular as option".

### Round 2, 2026-11-21: English-only checkpoints

`tiny.en`, `base.en` and `small.en` at ±63 score 10.11, 7.8 and 4.68 on English, against 12.5, 8.06 and 5.37 for the
multilingual checkpoints at ±63. Whether each gap exceeds sampling noise was not tested. They break sooner: at ±15
`base.en` loses 8.05 points and `small.en` 4.56.

### Round 3, 2026-11-21: pack, reference, engine, speed

Turbo loads as float16 in transformers 5 unless asked for float32, which made the pack differ from the quantizer;
the pack writer's exactness assertion caught it and `pack.py` now loads in float32. The sweep had forced float32
already.

Packs as hosted (gzip, MB): tiny 25 English and 28 all languages, base 47 and 55, small 171 and 183, turbo 412 and
493. The small English pack is `small.en` at ±63: a grouped 5-bit scheme measured 169 MB at the same accuracy, and
the pack writer does not support it yet.

Mac timings: tiny encodes a 30 s window in 59 ms and base in 107 ms. Turbo took 13.9 s to download, inflate and
upload on first load. End to end on the 7.28 s clip: `base.en` 0.9 s and tiny 0.82 s, which is slower per unit of
model than turbo because the spectrogram and the per-token readback do not shrink with the model. Counting only
encoding and decoding, turbo's English pack takes 0.68 s on that clip.

### Round 4, 2026-11-21: timestamps, long audio, the page

`web/worker.js` speaks the messages of Anything to Text's existing worker. The page lists the packed models first
when WebGPU is present, with a Build selector, and falls back to the transformers.js model if a pack fails to load.
The English builds of tiny, base and small are English-only checkpoints, so the page uses them only when English
is selected; with Auto-detect those sizes use the all-languages build. Turbo's English build is the multilingual
checkpoint and serves Auto-detect.

### Round 5, 2026-11-21: a fair speed comparison

Oskar asked whether the engine had been made faster. Round 4's end-to-end figure (1.17 s against 1.18 s) compared our
run with language detection and timestamps against a transformers.js run with the language given and timestamps off.

With the page's settings on both sides (job `whisper-pack-fair-2`): 7 s clip 1.44 s against 1.37 s, 28 s clip 2.16 s
against 2.85 s. Our spectrogram took 731 ms of the 1.44 s. It summed each mel filter over all 201 frequency bins and
transformed every silent padding frame; skipping both is exact (`test/mel_ab.mjs`: 0 differing values on 7 clips and
2 filterbanks) and took the 7 s spectrogram in Node from 282 to 38 ms.

The next two runs (`fair-3`, profiles 1 to 4) showed no gain. Profiling found the decoder at 7.7 ms a step with the
page's settings, and `transcribe()` spending 715 ms in the spectrogram while the same function timed alone took
100 ms. The bridge page runs jobs in one long-lived worker, and a worker keeps each ES module it has loaded under its
URL: the jobs loaded `transcribe.js` with a cache-busting query, but its own `import "./mel.js"` resolved to the plain
URL already loaded hours earlier, so every run used the old spectrogram. With all modules loaded from a commit-pinned
jsDelivr address (`fair-4`): 7 s clip 0.80 s against 1.29 s, 28 s clip 1.69 s against 2.69 s, 1.6 times faster on both.
Transformers.js itself moved 5 to 15% between identical runs.

