# RESULTS: Whisper from quantized weight packs on hand-written WebGPU kernels

[Anything to Text](https://austegard.com/web-utilities/anything-to-text.html) transcribes audio in the browser
with OpenAI's Whisper, run by transformers.js from ONNX files. This experiment asks whether Whisper can be packed
the way `gliner-browser` and `indra-browser` packed their models: weights stored as small integers, run by our own
WebGPU kernels. It measures what each packing scheme costs in word error rate (WER), builds the engine, and
compares download size and speed with what the page used before. The demo is
[live](https://austegard.com/experiments/whisper-pack/web/). Whisper's weights are MIT-licensed.

## Answer

Yes. Each Whisper size now has two packs: an English-optimized one and an all-languages one. For large-v3-turbo the English pack is a 412 MB download <!-- @evidence[[results/headline.json#turbo_en_gzip_mb]] -->
and the all-languages pack 493 MB, <!-- @evidence[[results/headline.json#turbo_all_gzip_mb]] -->
against 759.1 MB for the ONNX files the page used. <!-- @evidence[[results/headline.json#page_turbo_mb]] -->
The English pack changes English WER by +0.04 points (95% interval −0.13 to +0.22) from full precision. <!-- @evidence[[results/headline.json#turbo_r15_en_diff]] -->
Tiny, base and small shrink more: base is 47 MB <!-- @evidence[[results/headline.json#base_en_gzip_mb]] -->
against 187.2 MB. <!-- @evidence[[results/headline.json#page_base_mb]] -->

On a Mac (Apple Metal, Chrome 157) the engine transcribed a 7.28 s clip with turbo in 0.68 s, <!-- @evidence[[results/headline.json#mac_turbo_clip_s]] -->
where transformers.js with the ONNX q4 files took 1.18 s. <!-- @evidence[[results/headline.json#mac_tjs_clip_s]] -->
The engine's tokens equal the NumPy reference's, which equal Hugging Face's.

The turbo saving is in the download. Unpacked, and on the GPU, the turbo English pack is 815 MB, <!-- @evidence[[results/headline.json#turbo_en_raw_mb]] -->
because the engine holds each weight in one byte.

What would change this: a speed comparison on more than one clip and one machine, the WER of the ONNX q4 files themselves (not measured here), and bit-packed weights on the GPU, which would cut memory.

## Findings

1. **Turbo holds English accuracy at 5 bits per weight, and loses it on other languages.** Codes in ±15 for every matrix and the token table change English WER by +0.04 <!-- @evidence[[results/headline.json#turbo_r15_en_diff]] -->
   and WER on four other languages by +0.79 (interval +0.47 to +1.12). <!-- @evidence[[results/headline.json#turbo_r15_fleurs_diff]] -->
   That is why there are two packs. The all-languages turbo pack uses a 4-bit token table and ±31 matrices: +0.02 on English <!-- @evidence[[results/headline.json#turbo_t4r31_en_diff]] -->
   and +0.15 on the other languages, an interval that spans zero. <!-- @evidence[[results/headline.json#turbo_t4r31_fleurs_diff]] -->
   (sweep, turbo) 2. **English test sets hide the damage on other languages.** For whisper-small, ±31 everywhere costs +0.01 on English <!-- @evidence[[results/headline.json#small_r31_en_diff]] -->
   and +0.77 on the other four languages. <!-- @evidence[[results/headline.json#small_r31_fleurs_diff]] -->
   Only ±63 was clean on both: +0.02 and +0.02. <!-- @evidence[[results/headline.json#small_r63_fleurs_diff]] -->
   (sweep, small) 3. **One scale per row does not survive 4 bits at any size.** With codes in ±7 and one scale per output row, turbo's English WER is 142.73. <!-- @evidence[[results/headline.json#turbo_r7_en_wer]] -->
   With one scale per 32 weights and a clip search, the scheme closest to what the ONNX q4 files use, turbo loses 0.21 on English <!-- @evidence[[results/headline.json#turbo_g32r7_en_diff]] -->
   and 0.53 on the other languages. <!-- @evidence[[results/headline.json#turbo_g32r7_fleurs_diff]] -->
   (sweep, turbo) 4. **The 4-bit token table that was free for our text encoders costs Whisper's small models accuracy.** With ±63 matrices it costs base 0.8 points on English <!-- @evidence[[results/headline.json#base_rec_en_diff]] -->
   and tiny 1.47. <!-- @evidence[[results/headline.json#tiny_rec_en_diff]] -->
   In Whisper the table is also the output projection, so every predicted token passes through it. (sweep, tiny and base) 5. **The English-only checkpoints are the better English default for tiny, base and small, and they need gentler packing.** At ±63, `small.en` scores 4.68 WER <!-- @evidence[[results/headline.json#small_en_r63_wer]] -->
   where multilingual small scores 5.37. <!-- @evidence[[results/headline.json#small_r63_wer]] -->
   At ±15, `small.en` loses 4.56 points <!-- @evidence[[results/headline.json#small_en_r15_diff]] -->
   and multilingual small 0.43. <!-- @evidence[[results/headline.json#small_r15_en_diff]] -->
   (English-only sweep) 6. **The engine reproduces the reference.** In headless Chromium, 4 of 4 cases give the NumPy reference's tokens for an 8-bit and a 4-bit token table, <!-- @evidence[[results/headline.json#parity_cases]] -->
   and with timestamps on, 3 of 3 clips give Hugging Face's tokens. <!-- @evidence[[results/headline.json#ts_cases]] -->
   On the Mac both turbo packs gave the reference tokens. (`test/check.mjs`, bridge job `whisper-pack-turbo-1`) 7. **Two kernel changes took turbo's encoder from 1,932 ms to 457 ms per 30 s window on the Mac.** <!-- @evidence[[results/headline.json#mac_turbo_encode_ms]] -->
   The matrix multiply decodes each weight word once for 8 rows, and attention keeps its scores in a local array so each key is scored once. A decoded token takes 7.9 ms. <!-- @evidence[[results/headline.json#mac_turbo_ms_per_token]] -->
   (bridge jobs `whisper-pack-turbo-1`, `-2`) 8. **A decoder step with one query must not walk the keys in one thread.** With one GPU thread per head scanning 1,500 audio frames, a token took 28 ms on tiny; with one thread per key and a second kernel for the weighted sum it takes 5.1 ms. <!-- @evidence[[results/headline.json#mac_tiny_ms_per_token]] -->
   (bridge jobs `whisper-pack-time-2`, `-3`)



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

**Speed.** Timings come from GPU Bridge jobs on Oskar's Mac (Apple Metal, Chrome 157): the engine loading the
hosted packs, and transformers.js 4.2.0 with `onnx-community/whisper-large-v3-turbo` q4 on WebGPU, on the same
7.28 s clip. The engine's figure is encode time plus tokens times the time per token, from the spectrogram on;
the transformers.js figure is the whole pipeline call including its own spectrogram.

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
upload on first load.

### Round 4, 2026-11-21: timestamps, long audio, the page

`web/worker.js` speaks the messages of Anything to Text's existing worker. The page lists the packed models first
when WebGPU is present, with a Build selector, and falls back to the transformers.js model if a pack fails to load.
The English builds of tiny, base and small are English-only checkpoints, so the page uses them only when English
is selected; with Auto-detect those sizes use the all-languages build. Turbo's English build is the multilingual
checkpoint and serves Auto-detect.
