# tiny-voca voices

Oskar, 2026-11-16: "Could you tweak this voice model's voices?
https://huggingface.co/n4ze3m/tiny-voca"

tiny-voca is a 4.5M-parameter English TTS model with two voices, F2 and M2.
This derives eight more from them without retraining. Every derived voice
transcribes exactly under Whisper small.en on both test sentences, and each
lands within a few hertz of its pitch target. Nobody has listened to them yet,
so whether they sound good is still open. Listening page:
https://claude.ai/artifact/6VEAYngPMRvxcctaLVCs1r (private).

## Controls

The model has a transformer front end and a source-filter waveform head: a
pulse train at the predicted pitch and a noise signal, each shaped by a
filter whose complex cepstrum the network predicts every 10.7 ms. That design
exposes four places to edit a voice.

| control | where it acts | effect |
|---|---|---|
| `mix` | weighted sum of the F2/M2 rows in `voices` and `frame_voices` (192-d each) | the network re-predicts timing, pitch and timbre for the new point; extrapolation (weights outside 0–1) works |
| `warp` | the filter's log-magnitude and phase, resampled along frequency | moves every formant by the factor, pitch unchanged |
| `tilt`, `breath` | dB/octave on the harmonic filter around 1 kHz; dB on the noise filter | brightness, breathiness |
| `semitones`, `range`, `speed`, `jitter` | the model's own synthesis arguments | pitch shift, pitch movement, rate, random pitch drift |

`voice_lab.py` monkeypatches `tiny_voca.model.cepstrum_response` for the
filter controls and appends a speaker row per blended voice, so a blend is an
ordinary speaker id and `--save-checkpoint` writes it into a checkpoint. The
filter controls are synthesis-time only; carrying them into the ONNX graph
would need an export with the patched function and extra inputs.

## Results

`scores.json`, from `evaluate.py` over both sentences in `texts.txt`:

| voice | recipe | F0 Hz | envelope × | WER |
|---|---|---:|---:|---:|
| F2 | stock | 268 | 1.000 | 0% |
| F2-adult | −4 st, warp 0.94, range 0.85, speed 0.95 | 216 | 0.970 | 0% |
| F2-adult-mix | mix 0.8 F2 + 0.2 M2, −2.5 st, warp 0.95 | 216 | 0.965 | 0% |
| F2-bright | mix 1.25 F2 − 0.25 M2 | 296 | 1.005 | 0% |
| F2-breathy | breath +6 dB, tilt −1.5 dB/oct | 267 | 0.995 | 0% |
| blend-50 | mix 0.5/0.5 | 214 | 0.965 | 0% |
| M2 | stock | 175 | 1.000 | 0% |
| M2-baritone | −5 st, warp 0.92, range 1.1, speed 0.95 | 131 | 0.930 | 0% |
| M2-light | +2 st, warp 1.05 | 196 | 1.040 | 0% |
| M2-mix-bright | mix 1.2 M2 − 0.2 F2 | 163 | 0.995 | 0% |

Stock F2 has a median pitch of 268 Hz, high for the adult woman in her mid
twenties it was designed as; the model card says F2 sounds younger than
intended. Stock M2 sits at 175 Hz, the top of the usual adult male range.
F2-adult and F2-adult-mix target the mature F2 the card says is planned.

Moving the embedding moves pitch on its own: F2→M2 is about −0.8 semitones per
10% of mix, and the 50/50 blend lands at 214 Hz.

The envelope column is the frequency scale that best aligns a voice's
cepstrally smoothed long-term spectrum with its source voice's. It understates
the warp. `check_warp.py` measures at the synthesis filter itself, where a
0.90 or 1.10 setting moves the envelope by exactly 0.90 or 1.10 for both
voices (correlation 1.0). The audio-level estimate for F2 at warp 0.90 reads
0.975, since F2's widely spaced harmonics sample the envelope coarsely. Praat
formant tracks (in `scores.json`) are too noisy at F2's pitch to confirm the
shift on their own.

## Running

The tiny-voca and dizione repos run only inside `claude-workspace`'s
`scripts/jail.sh`. Clone both under `/tmp/jail/tv/` (as `code/` and `diz/`),
put `tiny-voca.pt` and `phonemizer/` from the HF repo in `w/`, then:

```bash
sh scripts/jail.sh -C /tmp/jail/tv -e PYTHONPATH=/tmp/jail/tv/code/src:/tmp/jail/tv/diz/src \
  -e DIZIONE_DATA=/tmp/jail/tv/w/phonemizer -e TINY_VOCA_FRONTEND=own \
  -- python3 lab/voice_lab.py lab/presets.json out --text-file lab/texts.txt
```

`evaluate.py` (faster-whisper, praat-parselmouth, jiwer) scores an output
directory; `build_page.py` builds the listening page from `samples/`.
