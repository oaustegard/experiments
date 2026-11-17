# RESULTS: designing your own tiny-voca voice, in the browser

[tiny-voca](https://huggingface.co/n4ze3m/tiny-voca) is a 4.5M-parameter English text-to-speech model with two
voices, F2 and M2. Oskar asked to tweak its voices, then to turn that into a page where anyone can tune their own
voice and hear it in the browser, as `kokoro-web` did for Kokoro. The page is
[live](https://austegard.com/experiments/tiny-voca-voices/web/). The question here is which controls the model has
that change a voice without retraining, and how far each goes before speech stops being intelligible.

## Answer

A voice designed with eleven sliders saves as a 4.0 KB file that plays in other pages and in Python. <!-- @evidence[[results/headline.json#reuse_voice_kb]] -->
The sliders feed one 10.0 MB ONNX file. <!-- @evidence[[results/headline.json#model_mb_browser]] -->
It loads in 5.9 s in headless Chromium <!-- @evidence[[results/headline.json#browser_load_s]] -->
and speaks 15.8× faster than real time on one WebAssembly thread. <!-- @evidence[[results/headline.json#browser_realtime_factor_median]] -->
Voices come from the learned F2 and M2 vectors and four fixed random directions off that line, the model's own
pitch and speed controls, and a reshaped vocoder filter.

The voice vector is what limits intelligibility (Whisper transcribing exactly); of the other slider ends tested,
only speed ×2 lost a word. Off the F2–M2 line the combined distance decides, in units where 1.0 is as far as F2 is
from M2. Of 12 random voices with each axis uniform in ±1, 4 of 4 up to a distance of 1.13 transcribed exactly. <!-- @evidence[[results/headline.json#random_near_max_distance]] -->
Past 1.15, 6 of 8 garbled a word. <!-- @evidence[[results/headline.json#random_far_with_errors]] -->
The page caps each axis at ±1, warns past 1.0, and keeps "Surprise me" within 0.9.
All 16 of 16 such voices were clean. <!-- @evidence[[results/headline.json#bounded_random_clean]] -->
Pitch ran from 141 Hz <!-- @evidence[[results/headline.json#bounded_random_f0_min]] -->
to 383 Hz. <!-- @evidence[[results/headline.json#bounded_random_f0_max]] -->

Nobody has run a listening test; one could change which settings count as good.

## Findings

1. **Distance off the F2–M2 line limits intelligibility.** Every random voice up to a distance of 1.13 transcribed
   exactly, and past 1.15, 6 of 8 did not. <!-- @evidence[[results/headline.json#random_far_with_errors]] -->
   One axis at a time, at ±0.75 or ±1 on F2 and on M2, was clean in 31 of 32 cases. <!-- @evidence[[results/headline.json#single_axis_pm1_clean]] -->
   The exception was axis 2 at −1 on M2, which turned "and a" into "in a".
   At ±1.5, 7 of 8 had errors. <!-- @evidence[[results/headline.json#single_axis_pm15_with_errors]] -->
   (Round 5, browser checks 2 and 3)
2. **A saved voice plays the same in the tuner, in other pages and in Python.** The voice file holds the two
   embeddings and six controls the model reads, plus the slider settings for reloading. Spoken from the same file,
   the tuner, `tinyvoca.js` on `example.html` and `speak.py` (onnxruntime, no PyTorch) each produced 133632 samples. <!-- @evidence[[results/headline.json#reuse_samples]] -->
   Against the tuner, pitch differed by 4.4 cents at p90 for the library <!-- @evidence[[results/headline.json#reuse_library_pitch_cents_p90]] -->
   and 5.0 cents for Python. <!-- @evidence[[results/headline.json#reuse_python_pitch_cents_p90]] -->
   Each transcribed exactly (0.0% WER), and the file loaded back into the tuner restored every slider. <!-- @evidence[[results/headline.json#reuse_wer_pct]] -->
   The noise signal is random in each runtime, so waveforms differ. (Round 6)
3. **The speech model sings in tune when told each sound's length and pitch.** Two graph inputs override the
   predicted frames per phoneme and the pitch per frame. With them off, the output is unchanged (max difference 0.0
   against the previous file). <!-- @evidence[[results/headline.json#sing_override_off_max_diff]] -->
   `tinyvoca.js` maps hyphenated lyrics onto the phonemes and notes onto the vowels, with glides and vibrato.
   Sung in four voices, the median note error was at most 1.0 cents. <!-- @evidence[[results/headline.json#sing_note_error_cents_max_median]] -->
   In both F2 voices, 32 of 32 notes landed within 50 cents. <!-- @evidence[[results/headline.json#sing_f2_notes_in_tune]] -->
   Whisper transcribed the sung lyrics with 0.0% WER in every voice. <!-- @evidence[[results/headline.json#sing_wer_pct_max]] -->
   In M2 an octave down, 13 of 15 measured notes were within 50 cents. (Round 8)
4. **The sliders other than the four axes stay intelligible at their ends.** Eleven ends were tested: the F2↔M2
   slider at −0.5 and 1.5, pitch ±12 semitones, formant scale 0.8 and 1.25, brightness ±4 dB per octave, breath +12 dB, and speed 0.5
   and 2. Of these, 10 of 11 transcribed exactly. <!-- @evidence[[results/headline.json#other_slider_ends_clean]] -->
   The one error was speed ×2, which slurred one word. Pitch movement and breath −12 dB were not tested at their
   ends. (Round 5, browser check 1)
5. **The weights can be stored as 8-bit integers (int8) everywhere except where durations are set.** The shipped
   file, "int8-dur", keeps the text encoder and duration predictor at 16-bit floats and stores every other matrix
   at int8. It is 10.0 MB.
   At fp32 the file is 21.39 MB. <!-- @evidence[[results/headline.json#model_mb_fp32]] -->
   100% of token durations stayed identical. <!-- @evidence[[results/headline.json#int8dur_durations_identical_pct]] -->
   Pitch moved 6.6 cents (hundredths of a semitone) at the 90th percentile. <!-- @evidence[[results/headline.json#int8dur_pitch_cents_p90]] -->
   int8 everywhere moved pitch 30.6 cents, and 0.5% of durations changed. <!-- @evidence[[results/headline.json#int8all_pitch_cents_p90]] -->
   A first visit transfers 11.22 MB compressed, down from 21.98 MB with Pyodide. <!-- @evidence[[results/headline.json#first_visit_transfer_mb]] -->
   The phonemizer is a JavaScript port of dizione that matches Python on 12450 of 12450 corpus sentences. <!-- @evidence[[results/headline.json#dizione_agree]] -->
   A second visit loads offline, in 1.1 s. <!-- @evidence[[results/headline.json#offline_second_visit_s]] -->
   This is the same split [`kokoro-web`](../kokoro-web/RESULTS.md) found for Kokoro (its Finding 1). (Round 4)
6. **The formant warp is exact at the filter.** For each 10.7 ms frame, the model predicts the filter as a
   cepstrum, a compact description of a spectrum's shape. Its Fourier transform gives the filter's log-magnitude
   and its phase directly, and both can be stretched along frequency by linear interpolation.
   A 0.90 setting moves the filter envelope by exactly 0.90 (`check_warp.py`). In the graph the warp is one
   interpolation matrix built from the input. A spectrum measured from the audio understates the shift at high
   pitch: 0.975 for F2 at 0.90, because F2's harmonics are far apart and sample the envelope coarsely. (Round 2)
7. **The tunable graph matches PyTorch.** Given the same noise, the ONNX graph and the PyTorch module produce the
   same number of samples for every case.
   They differ by at most 1.456% of peak. <!-- @evidence[[results/headline.json#onnx_parity_max_err_pct_peak]] -->
   tiny-voca's own export reports 0.6% for its stock graph. (Round 4)
8. **Stock F2 is pitched high for an adult.** Its median pitch is 268 Hz. <!-- @evidence[[results/headline.json#lab_f2_f0]] -->
   That matches the model card's note that F2 sounds younger than designed.
   The F2-adult preset brings it to 216 Hz and lowers the formants 6%. <!-- @evidence[[results/headline.json#lab_f2_adult_f0]] -->
   Stock M2 is at 175 Hz. <!-- @evidence[[results/headline.json#lab_m2_f0]] -->
   M2-baritone is at 131 Hz. <!-- @evidence[[results/headline.json#lab_m2_baritone_f0]] --> (Round 1)

## Method

**Model and controls.** Inputs are phoneme ids and stress from tiny-voca's phonemizer, dizione. The model has two
voice embedding tables, each with an F2 row and an M2 row of 192 values: one is added at the phoneme level, the other
at the frame level. The remaining inputs, with the page's slider names, are speed, pitch (the model's pitch shift, in
log F0; the page takes semitones), pitch movement (the model's pitch range), formant scale, brightness (tilt in dB
per octave around 1 kHz, on the voiced branch) and breath (dB, on the noise branch). The page builds each
table's embedding as (1−m)·F2 + m·M2 + Σ aᵢ·d·uᵢ, where d is that table's F2–M2 distance. The distance off the line is
the norm of (a₁…a₄), which is the same for both tables. The
uᵢ are four unit directions, drawn with seed 7 and made orthogonal to F2−M2 and to each other. They are written to
`web/voice-space.json`.

**Arms.** In PyTorch, `run/voice_lab.py` patches `cepstrum_response` and appends speaker rows. `run/export_tunable.py`
subclasses tiny-voca's `Exportable`. `run/pack.py` produces the fp32, fp16, int8 and int8-dur files. The browser page
in `web/` runs onnxruntime-web 1.20.1 on WebAssembly with one thread, and dizione through Pyodide 0.26.4. Its worker is
adapted from tiny-voca's own demo.

**Metrics.** Word error rate comes from Whisper small.en (faster-whisper, int8, beam 5) against the input text, with
"3:30" normalised. Median F0 and formants come from Praat. The envelope scale is the frequency scale that best
aligns cepstrally smoothed long-term spectra. For packing, each file is compared with fp32 given the same noise:
token durations, Praat pitch error in cents, the largest 8-band level difference and WER. Waveform correlation is
not used, since phase drift drives it to about 0. Browser audio is captured from the page by
`run/browser_check.mjs` and scored with `run/evaluate.py`.

**Controls and limits.** Both stock voices are scored alongside every derived voice. Each browser check speaks one
sentence, and the lab checks speak two. Whisper on one or two sentences is a coarse test, and this is no listening
test. The axis hints on the page come from one F2 sentence at +1.

**Running.** tiny-voca and dizione run only inside `claude-workspace`'s `scripts/jail.sh`. Clone them to
`/tmp/jail/tv/code` and `/tmp/jail/tv/diz`, put the checkpoint and `phonemizer/` in `/tmp/jail/tv/w/`, and copy
`run/*.py` to `/tmp/jail/tv/lab/`. The scripts expect that layout (`lab/…`, `w/…`, `out/…`). The browser and
scoring scripts run anywhere.

## Log

### Round 1 (2026-11-16): tweak the voices

Asked: "Could you tweak this voice model's voices?" Read the architecture: a FastSpeech-style transformer with a
source-filter head that predicts complex cepstra. Built `voice_lab.py` with mix, warp, tilt and breath, plus the
model's own controls, and rendered 10 presets on two sentences (`samples/`, `results/lab_scores.json`). All
transcribed exactly. Pitch landed within a few Hz of its target: F2-adult measured 216 Hz against 213, and
M2-baritone 131 against 131. A listening page with the samples went up as a private artifact.

### Round 2 (2026-11-16): is the warp real?

The audio-level envelope estimate read 0.975 for a 0.90 warp on F2. `check_warp.py` measured the scale at the filter
inside the model, where it was exactly 0.90 and 1.10 for both voices. The gap comes from measuring through sparse
harmonics, not from the warp.

### Round 3 (2026-11-17): off the F2–M2 line

Asked: make it a page where a user tunes their own voice, as `kokoro-web` did. With only two learned voices the
space of blends is a line, so `explore_dirs.py` added four random directions orthogonal to it. At up to the full
F2–M2 distance on F2, and half of it on M2, all 12 voices transcribed exactly
(`results/off_axis_scores.json`). The directions moved pitch from −1.2 to +3.4 semitones and shifted formants by up
to about 9%.

### Round 4 (2026-11-17): export, parity, packing

`export_tunable.py` exports the controls as graph inputs, first as mix weights and then as raw embeddings so the page
can build any voice. Given the same noise, ONNX and PyTorch differ by at most 1.456% of peak
(`results/onnx_parity.json`). Packing (`results/packing.json`):

| variant | MB | durations identical | pitch p90 (cents) | band max (dB) | WER |
|---|---:|---:|---:|---:|---:|
| fp32 | 21.39 | 100% | — | — | 2.09% |
| fp16 | 12.69 | 100% | 0.8 | 0.22 | 2.59% |
| int8-dur | 9.97 | 100% | 6.6 | 0.21 | 2.09% |
| int8 | 8.44 | 99.46% | 30.6 | 0.76 | 2.09% |

(WER is over 21 renders: 7 sentences in 3 voices. The fp16 WER difference is a single word.) int8-dur ships.

### Round 5 (2026-11-17): the page, in headless Chromium (three browser checks)

`web/` serves the page, a worker, tiny-voca's player, the 10 MB model, the voice space and a zip of the dizione
source. The phonemizer data loads from Hugging Face, pinned to a revision, with CORS. Browser check 1
spoke the 10 presets and 19 slider extremes. It loaded in 5.9 s, the presets were all clean, and every voice axis at
±1.5 failed except axis 1 at +1.5 (`results/browser_presets_extremes.json`). Check 2 tried ±0.75 and ±1 on each axis
on F2 and M2 (31 of 32 clean), plus 12 random voices drawn with axes uniform in ±1. Every random voice up to a distance of 1.13
was clean, and 6 of 8 past 1.15 were not (`results/browser_axes_random.json`). The axes were capped at ±1, a distance
readout with a warning past 1.0 was added, and "Surprise me" now draws within 0.9. Check 3 then gave 16 of 16 clean
voices (`results/browser_bounded_random.json`).

### Round 6 (2026-11-17): reuse outside the page

Asked: "I thought we were supposed to give the user a way to reuse the voice in their own efforts?" The tuner only
saved WAVs and a link. Added a voice file (`tiny-voca-voice/1`: name, model URL and hash, both embeddings, the six
controls, and the tuner settings) with Save and Load buttons; `web/tinyvoca.js`, a library that loads the model and
phonemizer and speaks text in any voice file, in a page or a worker (the tuner's worker now runs on it);
`web/example.html`, a 40-line page that uses it; and `speak.py` for Python. `run/reuse_check.mjs` designs a voice,
saves it with the page's button, reloads it into a fresh tuner and speaks it through `example.html`. `speak.py`
then speaks the same file in the jail, and `run/reuse_score.py` compares the three (`results/reuse.json`):

| runtime | samples | median F0 | pitch vs tuner, p90 | WER |
|---|---:|---:|---:|---:|
| tuner | 133,632 | 191.8 Hz | — | 0% |
| tinyvoca.js | 133,632 | 192.0 Hz | 4.4 cents | 0% |
| speak.py | 133,632 | 191.9 Hz | 5.0 cents | 0% |

### Round 7 (2026-11-17): download size and a text parameter

Asked for the total download and for the text as an optional query parameter. A fresh headless load of the live
page transferred 21.98 MB in 29 requests (`results/transfer.json`): model 8.4 MB (gzip from GitHub Pages), Pyodide and
numpy 8.0 MB, dictionary 2.9 MB, onnxruntime-web 2.5 MB. The page had claimed about 30 MB; corrected. Added
`?text=` to the tuner (the voice stays in `#v=`, and editing the text updates the address, so "Copy link" shares both)
and `?text=` plus `&voice=URL` to `example.html`. The ten presets ship as voice files in `web/voices/`
(`run/make_preset_voices.mjs`). Browsers start sound only from a click, so neither page speaks on load.
`run/query_check.mjs` passed every check, and the tuner and `example.html` gave the same 105,984 samples for
M2-baritone.

### Round 8 (2026-11-17): singing

Asked to make it sing the chorus of a Leonard Cohen song. That song's lyrics and melody are under copyright, so
the demo uses the first verse of "Amazing Grace" (Newton, 1779; tune "New Britain", 1829), and the page takes any
lyrics and notes a user enters. `export_tunable.py` gained `frames_override` (int64 per token, −1 = predicted) and
`f0_override` (Hz per frame, 0 = predicted). With both off the new int8-dur file reproduces the previous one exactly
(max difference 0.0 with the same noise). Doubling every duration and asking for 220 Hz gave exactly the requested
frame count and a median pitch of 219.9 Hz. Old voice files stay valid (the library accepts both model hashes).

`tinyvoca.js` `sing()` phonemizes each lyric line, runs the model once for its predicted durations, then gives
consonants their predicted length (at most 7 frames, about 75 ms) and vowels the rest of each note. It builds a
pitch track from the notes with a one-pole glide and vibrato on notes over 0.35 s, and runs the model again with
both overrides. `web/sing.html` is the page. `run/sing_check.mjs` sang the score in four voices, and
`run/sing_score.py` measured Praat pitch over the middle half of each note and Whisper WER
(`results/sing_scores.json`):

| voice | transpose | note error (median) | notes within 50 cents | WER |
|---|---:|---:|---:|---:|
| F2 | 0 | 0.8 cents | 16 of 16 | 0% |
| F2-adult | 0 | 0.7 cents | 16 of 16 | 0% |
| M2-baritone | −12 | 0.7 cents | 15 of 15 measured | 0% |
| M2 | −12 | 1.0 cents | 13 of 15 measured | 0% |

18.7 s of song took 1.5 to 1.75 s to make. The reuse, query and preset checks passed again on the new model with
the same numbers. Nobody has listened to the singing; the pitch and words are measured, while the tone is not.

### Round 9 (2026-11-17): one page, smaller, cached, no Python

Asked why the tuning sliders were gone from singing (they were on another page), and to cut the download and
cache it. Oskar then reported both pages stuck at "Starting the phonemizer" in Chrome on iPhone. iOS browsers all
use WebKit, and every check here runs Chromium. The cause was not reproduced, but the stall was inside Pyodide.

- **One page:** Sing became a tab of the tuner, so every voice slider applies to singing. `sing.html` redirects
  to it.
- **Smaller model:** the model's fixed DFT tables (STFT kernels and cepstrum matrices, 2.8 MB of float32) are
  computed at load by `run/compute_constants.py`. Each is checked against the stored table to 2e-7 before it is
  replaced. With the same noise the output is within 0.0002% of peak, and every duration is identical. Gzipped,
  the model went from 8.43 MB to 6.26 MB.
- **Caching:** model and data files are named by content hash. The onnxruntime wasm goes through the library's
  Cache Storage, `load()` is memoized, and old caches are pruned.
- **No Python:** a subagent ported dizione to JavaScript (`web/dizione.js` plus a 3.1 MB bundle, 2.3 MB gzip).
  It matched Python on 12,450 of 12,450 sentences (`results/dizione_agreement.json`). The corpus covered tests,
  Gutenberg prose, numbers, dates, money, units, chat forms, names, homographs, words outside the dictionary,
  other scripts, fuzz and edge cases. The ids and stress match exactly, and so do the null and error cases.
  Pyodide, numpy and the Hugging Face fetches are gone.

Live after the change (`results/cache_live.json`): the first visit transfers 11.22 MB and loads in 3.1 s. With
the network off, a second visit loads in 1.1 s and speaks. A second `TinyVoca.load()` returns the same instance.
The preset, singing, reuse and query checks all passed again. All four voices sang the score at a median note
error of at most 1.0 cents with lyrics transcribed exactly. Across the earlier Pyodide runs, two of 20 F2 renders
had the opening "Amazing" misheard, which matches run-to-run noise. The iPhone result still needs a test on a real
device.

### Round 10 (2026-11-17): the iPhone out-of-memory error

Chrome on an iPhone stopped at "Starting the model" with `no available backend found. ERR: [wasm] RangeError: Out
of memory`. The error appeared first in the worker, and again after iOS was switched to loading on the page, while
`example.html` worked. Wrapping `WebAssembly.Memory` in WebKit showed onnxruntime-web 1.20.1 creating
`{"initial":256,"maximum":65536,"shared":true}`: a shared memory with a 4 GB maximum, which has to be reserved in
full up front. 1.19 and later ship only that threaded build. 1.18.0 still ships a non-threaded `ort-wasm-simd.wasm`
that creates its own growable memory, and with one thread and no cross-origin isolation it selects that build.
After the switch, no shared memory is requested.

Two smaller changes reduce the memory in use when the runtime starts:
- Downloads are read into one buffer instead of collecting chunks and joining them. The chunk version doubled each
  file's peak.
- The dictionary's in-memory tables (about 14 MB, measured in Node) are built after the model session starts.

With 1.18, the checks gave the same results as before:
- presets and slider ends clean except speed ×2;
- singing at a median note error of 0.6–1.1 cents, with every measured note within 50 cents;
- identical sample counts in the reuse check;
- the second visit still works offline (onnxruntime's wasm comes from jsDelivr's HTTP cache, since 1.18 has no
  `wasmBinary` option).

F2's sung opening word was misheard once more, which makes 3 of 7 F2 runs; the other voices were clean. A real
iPhone has not confirmed the fix yet.

### Round 11 (2026-11-17): confirmed on the iPhone, through relay.js

The tuner now reports its own log to `gpu-bridge-comms` via `gpu-bridge/web/relay.js`, which runs only on
Oskar's devices and after `?log=1`. The first iPhone session ran with Chrome 157 on iOS 27.2.1, build relay1
(onnxruntime-web 1.18), loading on the page (`logs/tiny-voca-tuner/2026-11-17/030252-iPhone-702j4n.jsonl`):

- the model started without an out-of-memory error;
- the tuner was ready 0.8 s after the page opened, with the files already cached;
- it spoke 5.82 s of audio in 0.215 s and sang the 18.69 s score in 0.786 s.

The earlier iPhone errors came from builds diz1 and diz3, which the phone kept serving from cache after newer
builds were live. The page now checks `build.json` past every cache and reloads once when it is stale.

### Round 12 (2026-11-17): chorus

Asked to make it a chorus by singing several "Surprise me" voices at once. `tinyvoca.js` `chorus(score, parts)`
sings the score once per part and mixes the parts to stereo with equal-power panning, scaling the sum by
1/sqrt(n) and capping its peak at 0.99. Each part can set a fractional transpose (detune), a delay, a vibrato
rate and a pan position. In the tuner, the Chorus slider (solo to 8) makes the slider voice the lead. The other
voices come from the Surprise-me sampler, so they stay inside the tested-clear distance of 0.9. A seed in the link
fixes which voices are drawn.

Each added voice gets:
- up to ±8 cents of detune;
- up to 25 ms of delay;
- a vibrato rate of 5.5 ± 0.7 Hz;
- a place spread from −0.7 to 0.7 across the stereo field, with the lead nearest the centre.

With "octave" on (the default), a voice on the other side of the F2–M2 midpoint from the lead sings an octave
away. Save WAV writes stereo.

`run/chorus_check.mjs` checked seed 12345 (`results/chorus.json`):
- In Chromium and in WebKit with iPhone emulation, the seed drew the same voices.
- The container took 5.5 s to sing a 4-voice chorus of the 18.7 s score and 10–11 s for 8 voices.
- Whisper transcribed the mono fold-down exactly every time.
- 96% of voiced frames were within 50 cents of one of the melody's pitch classes (octaves folded).

`samples/chorus-amazing-grace-4-voices.mp3` is the 4-voice mix.

