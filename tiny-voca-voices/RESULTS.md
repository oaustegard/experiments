# RESULTS: designing your own tiny-voca voice, in the browser

[tiny-voca](https://huggingface.co/n4ze3m/tiny-voca) is a 4.5M-parameter English text-to-speech model with two
voices, F2 and M2. Oskar asked to tweak its voices, then to turn that into a page where anyone can tune their own
voice and hear it in the browser, as `kokoro-web` did for Kokoro. The page is
[live](https://austegard.com/experiments/tiny-voca-voices/web/). The question here is which controls the model has
that change a voice without retraining, and how far each goes before speech stops being intelligible.

## Answer

Eleven sliders, all inputs to one ONNX file of 10.0 MB. <!-- @evidence[[results/headline.json#model_mb_browser]] -->
It loads in 5.9 s in headless Chromium <!-- @evidence[[results/headline.json#browser_load_s]] -->
and speaks 15.8× faster than real time on one WebAssembly thread. <!-- @evidence[[results/headline.json#browser_realtime_factor_median]] -->
The voice comes from the model's learned voice vectors: a walk between F2 and M2, plus four directions
off that line, drawn once at random with a fixed seed. Pitch, pitch movement and speed are the model's own
controls; formant scale, brightness and breath reshape the filter its vocoder predicts.

The voice vector is what limits intelligibility (Whisper transcribing exactly); of the other slider ends tested,
only speed ×2 lost a word. Off the F2–M2 line the combined distance decides, in units where 1.0 is as far as F2 is
from M2. Of 12 random voices with each axis uniform in ±1, 4 of 4 up to a distance of 1.13 transcribed exactly. <!-- @evidence[[results/headline.json#random_near_max_distance]] -->
Past 1.15, 6 of 8 garbled a word. <!-- @evidence[[results/headline.json#random_far_with_errors]] -->
The page caps each axis at ±1, warns past 1.0, and keeps "Surprise me" within 0.9.
All 16 of 16 such voices were clean. <!-- @evidence[[results/headline.json#bounded_random_clean]] -->
Median pitch ran from 141 Hz <!-- @evidence[[results/headline.json#bounded_random_f0_min]] -->
to 383 Hz. <!-- @evidence[[results/headline.json#bounded_random_f0_max]] -->

Nobody has run a listening test; one could change which settings count as good.

## Findings

1. **Distance off the F2–M2 line limits intelligibility.** Every random voice up to a distance of 1.13 transcribed
   exactly, and past 1.15, 6 of 8 did not. <!-- @evidence[[results/headline.json#random_far_with_errors]] -->
   One axis at a time, at ±0.75 or ±1 on F2 and on M2, was clean in 31 of 32 cases. <!-- @evidence[[results/headline.json#single_axis_pm1_clean]] -->
   The exception was axis 2 at −1 on M2, which turned "and a" into "in a".
   At ±1.5, 7 of 8 had errors. <!-- @evidence[[results/headline.json#single_axis_pm15_with_errors]] -->
   (Round 5, browser checks 2 and 3)
2. **The sliders other than the four axes stay intelligible at their ends.** Eleven ends were tested: the F2↔M2
   slider at −0.5 and 1.5, pitch ±12 semitones, formant scale 0.8 and 1.25, brightness ±4 dB per octave, breath +12 dB, and speed 0.5
   and 2. Of these, 10 of 11 transcribed exactly. <!-- @evidence[[results/headline.json#other_slider_ends_clean]] -->
   The one error was speed ×2, which slurred one word. Pitch movement and breath −12 dB were not tested at their
   ends. (Round 5, browser check 1)
3. **The weights can be stored as 8-bit integers (int8) everywhere except where durations are set.** The shipped
   file, "int8-dur", keeps the text encoder and duration predictor at 16-bit floats and stores every other matrix
   at int8. It is 10.0 MB.
   At fp32 the file is 21.39 MB. <!-- @evidence[[results/headline.json#model_mb_fp32]] -->
   100% of token durations stayed identical. <!-- @evidence[[results/headline.json#int8dur_durations_identical_pct]] -->
   Pitch moved 6.6 cents (hundredths of a semitone) at the 90th percentile. <!-- @evidence[[results/headline.json#int8dur_pitch_cents_p90]] -->
   int8 everywhere moved pitch 30.6 cents, and 0.5% of durations changed. <!-- @evidence[[results/headline.json#int8all_pitch_cents_p90]] -->
   This is the same split [`kokoro-web`](../kokoro-web/RESULTS.md) found for Kokoro (its Finding 1). (Round 4)
4. **The formant warp is exact at the filter.** For each 10.7 ms frame, the model predicts the filter as a
   cepstrum, a compact description of a spectrum's shape. Its Fourier transform gives the filter's log-magnitude
   and its phase directly, and both can be stretched along frequency by linear interpolation.
   A 0.90 setting moves the filter envelope by exactly 0.90 (`check_warp.py`). In the graph the warp is one
   interpolation matrix built from the input. A spectrum measured from the audio understates the shift at high
   pitch: 0.975 for F2 at 0.90, because F2's harmonics are far apart and sample the envelope coarsely. (Round 2)
5. **The tunable graph matches PyTorch.** Given the same noise, the ONNX graph and the PyTorch module produce the
   same number of samples for every case.
   They differ by at most 1.456% of peak. <!-- @evidence[[results/headline.json#onnx_parity_max_err_pct_peak]] -->
   tiny-voca's own export reports 0.6% for its stock graph. (Round 4)
6. **Stock F2 is pitched high for an adult.** Its median pitch is 268 Hz. <!-- @evidence[[results/headline.json#lab_f2_f0]] -->
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
