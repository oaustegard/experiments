# RESULTS: Kokoro text-to-speech in the browser, on onnxruntime-web

Kokoro-82M is the text-to-speech model behind Kinescript's narration. It reports how long each phoneme lasts, so
every word gets a start and end time, which drive captions and cues. This experiment puts it in a web page with
onnxruntime-web and asks how small and light it can be made while every word keeps the time the PyTorch model gives
it and the voice sounds the same. The page speaks a fixed set of lines: their text was turned into phonemes ahead of
time in Python by misaki, Kokoro's own phonemizer, so the browser runs only the model. Typing your own text would
need misaki's dictionaries in the page (Finding 2). The page is [live](https://austegard.com/experiments/kokoro-web/web/).
Kokoro's weights and voices are Apache-2.0, kokoro-onnx's export is MIT, and misaki is Apache-2.0.

## Answer

The model is 95% of what the page downloads, so it is the only part worth shrinking now. A file that stores the
vocoder, Kokoro's acoustic text encoder and its energy predictor as int8, and the rest at fp16, is 102 MB unpacked.
Brotli-encoded it is 88.1 MB, against 325 MB for the fp32 export. <!-- @evidence[[results/headline.json#model_br_mb]] -->
In headless Chromium all 337 word times matched PyTorch exactly. <!-- @evidence[[results/headline.json#words_exact]] -->
Every frequency band was within 0.05 dB of PyTorch, and the pitch error (4.1 cents median, 17.1 at the 90th
percentile) was near that between two PyTorch runs (3.8 and 14.2). On 4 wasm threads on this container's 4 vCPUs
it spoke 125 s of narration in 93.0 s, <!-- @evidence[[results/headline.json#browser_synth_s]] --> about 3× slower than
native onnxruntime on the same CPUs (29 s); on one thread it took 261 s. A small service worker gives GitHub Pages the
cross-origin isolation those threads need. After loading, the page held about 500 MB (with garbage collected; an
empty page is 86 MB), and speaking the longest lines took it to about 800 MB.

What would change this: speed on real hardware (a laptop CPU, WebGPU on a real GPU), which nothing here measured,
and a listening test, since every audio comparison is a measurement.

## Findings

1. **The parts that set durations and pitch must stay fp16; the rest packs to int8 for free.** Kokoro predicts
   durations, then pitch (F0) and energy (N), then runs a vocoder. int8 on the duration part moved word times by
   up to 100 ms (`kokoro-onnx-kinescript` round 5). int8 on the acoustic text encoder and the N predictor left pitch
   at the floor: a 90th-percentile error of 16.5 cents (hundredths of a semitone). <!-- @evidence[[results/headline.json#p90_te_n]] -->
   int8 on the F0 predictor raised it to 30.9 cents, and on the LSTM that feeds F0 to 26.7, without moving a word.
   Packing everything outside the duration part (`mixed2`, 86 MB gzip) put pitch at 37.9 cents, <!-- @evidence[[results/headline.json#p90_mixed2]] -->
   against 14.2 between two PyTorch runs. (build round 2, `pitch.py`)
2. **The model is 95% of the download, the runtime 4%.** As served: model 88.1 MB (brotli), onnxruntime-web's wasm
   3.8 MB (gzip), each voice 0.19 MB, phonemes and page about 10 KB. The model is 95% of it. <!-- @evidence[[results/headline.json#model_share]] -->
   Phonemizing typed text in the page would add misaki's dictionaries, about 1.4 MB gzip for American or British
   English. (sizes)
3. **The browser matches Python onnxruntime.** For the shipped file, all 337 word times are exact, the waveforms
   correlate at 0.967 with Python's run of the same file (two runs of one engine reach 0.985, because the vocoder adds
   random noise), and bands agree within 0.02 dB. The WebGPU backend also gave line 1's 29 word times exactly, on
   SwiftShader, a software GPU whose speed means nothing. (browser round)
4. **Releasing the model bytes after the session starts saves 117 MB.** Resident memory after load (forced
   garbage collection) was 615 MB with the page holding the bytes and 498 MB without. <!-- @evidence[[results/headline.json#mem_release_mb]] -->
   Speaking adds about 310 MB for the longest lines. onnxruntime's arena and memory-pattern options made no
   difference. (memory round)
5. **Threads matter 2.8×.** The same model on one wasm thread took 260.8 s for 125 s of speech, against 93.0 s on
   four. <!-- @evidence[[results/headline.json#browser_1thread_synth_s]] --> GitHub Pages sends no isolation headers,
   so `web/coi.js` registers a service worker that adds them and reloads the page once. Served with no headers,
   the page came up isolated and used four threads. Where isolation fails, it falls back to one thread. A machine
   with fewer cores uses as many as it has. (service-worker round)
6. **onnxruntime turns fp16 weights back into fp32 at load but keeps int8 ones packed.** fp16 and fp32 files used
   the same memory (1,328 and 1,299 MB after load) and the same time; the int8 files used 750–800 MB. So int8 saves
   memory as well as download, and fp16 saves only download. (browser round)
7. **Brotli beats gzip by 3%, and Cloudflare R2 serves it pre-compressed.** 88.1 MB against 90.9 MB at gzip -9. The
   file is over GitHub's 100 MB limit, so it lives in a public R2 bucket. Uploaded with `Content-Encoding: br`, it is
   served with that header and with CORS for austegard.com, and the decoded bytes hash the same as the file. Another
   site would host its own copy (`build.py` makes it). (hosting)

## Method

**Fixture.** The ten narration lines of the `colbert-pooling-explainer` video in this repo (337 words), spoken with a
2:1 blend of the British voices `bm_fable` and `bm_lewis` at speed 0.88 (Kokoro's speed factor). The PyTorch reference
and a repeat of it (`torch312`, `torch312b`: kokoro 0.9.4 on Python 3.12) come from `kokoro-onnx-kinescript`, which
also found and fixed a flaw in the export's STFT that tilted its sound by about 3 dB.

**Model files.** `build.py` starts from kokoro-onnx's `model-files-v1.1` `kokoro-v1.0.onnx`, writes the
PyTorch-matching STFT weights into it, and replaces each large weight by an fp16 copy plus `Cast`, or by int8 codes
(one scale per output channel, codes ±127) plus `DequantizeLinear`, so every consumer node is unchanged. Which
weights go to int8 is decided by the consuming node's module path: durations come from BERT → the duration encoder
(`/text_encoder/lstms.*`) → `/lstm` → `/duration_proj`, while the vocoder (`/decoder/`), the acoustic text encoder
(`/text_encoder/cnn`, `/text_encoder/lstm`) and the F0/N predictors only shape the sound.

| variant | int8 | raw MB | gzip MB |
|---|---|---|---|
| fp32 | none | 325.5 | — |
| fp16 | none (all fp16) | 163.5 | 150.2 |
| mixed | vocoder | 110.6 | 99.1 |
| mixed+te+n (shipped) | vocoder, acoustic text encoder, N | 101.9 | 90.9 (88.1 brotli) |
| mixed2 | + F0 and its LSTM | 96.8 | 86.0 |

**Python check.** `run_py.sh` speaks the lines with kinescript-kokoro's onnx engine on each file, inside
claude-workspace's jail; `compare.py` and `tilt.py` (copied from `kokoro-onnx-kinescript`) read word times, band
energies and correlation against PyTorch; `pitch.py` compares Praat F0 every 10 ms where both are voiced.

**Browser check.** `bake.py` writes phoneme chunks (misaki plus KPipeline's chunking), the vocabulary and fp16
voice packs. `web/kokoro.js` runs the session and ports KPipeline's word-timing rule. `bench.mjs` serves the page
(with or without isolation headers) to headless Chromium, speaks every line, records resident memory of the
renderer every 100 ms and once after a forced garbage collection, and writes WAVs and word times for the same
comparisons. All timings come from this container's 4 vCPUs.

## Log

### Build round 1: fp16, int8 vocoder (2026-11-13)

All four files (fp32, fp16, mixed, mixed63) kept 337 of 337 word times exact in Python. `mixed` was within 0.04 dB
per band and pitch 4.0 / 16.7 cents; `mixed63` (codes ±63, the range this repo's `ettin-native` used) gzipped to 93.0 MB but put +0.17 dB
on the top band. Python speed was the same for every file (28.2–29.0 s for 125 s), peak memory 963 MB (fp32) and
819 MB (mixed).

### Browser round (2026-11-13)

| run | model | threads | speak s (125 s audio) | peak during load MB | peak MB |
|---|---|---|---|---|---|
| wasm4-mixed | mixed | 4 | 92.6 | 797 | 1,166 |
| wasm1-mixed | mixed | 1 | 260.8 | 772 | 1,125 |
| wasm4-fp32 | fp32 | 4 | 90.7 | 1,299 | 1,359 |
| wasm4-fp16 | fp16 | 4 | 91.5 | 1,328 | 1,422 |
| wasm4-mixed+te+n | mixed+te+n | 4 | 93.0 | 747 | 1,076 |

Browser output against Python's for `mixed`: correlation 0.968,
bands within 0.01 dB.

### Build round 2: what else can be int8 (2026-11-13)

`mixed2` (int8 for everything outside the duration path) kept word times and band energies but dropped correlation
to 0.31. Praat showed the pitch moving (37.9 cents at the 90th percentile). One group at a time on top of `mixed`:

| added int8 | median cents | p90 cents | corr |
|---|---|---|---|
| none (mixed) | 4.0 | 16.7 | 0.971 |
| acoustic text encoder | 3.9 | 16.4 | 0.973 |
| N predictor | 3.9 | 16.2 | 0.972 |
| F0 predictor | 5.4 | 30.9 | 0.580 |
| shared LSTM (feeds F0, N) | 5.3 | 26.7 | 0.600 |
| text encoder + N | 4.0 | 16.5 | 0.970 |

### Memory round (2026-11-13)

First three lines, `mixed+te+n`, 4 threads. Empty page 86 MB.

| page | after load, forced GC | peak |
|---|---|---|
| keeps the model bytes | 615 | 964 |
| releases them | 498 | 811 |
| releases them, no arena or memory patterns | 511 | 820 |

wasm memory grows and never shrinks, so whatever onnxruntime touched while building the session stays resident.

### Service-worker round (2026-11-13)

Served with no isolation headers, `coi.js` registered, reloaded once, and the page ran on four threads (two lines:
25.3 s of speech in 21.7 s, 497 MB after load).

### Live check (2026-11-13)

`live_check.mjs` opened the published page in headless Chromium through this session's proxy. The service worker
made it cross-origin isolated (4 wasm threads), the model came from R2 brotli-encoded and decoded to 101.9 MB in
4.0 s, the session started in 2.4 s, and line 1 (29 words, 10.8 s of speech) took 8.8 s, with no page errors. The
first deploy failed: Pages publishes only listed web file types, and `.f16` was not one, so the voice packs are
now `.f16.bin`.
