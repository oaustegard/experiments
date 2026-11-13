# RESULTS: Kokoro voices for Kinescript on Python 3.13, without PyTorch

Kinescript's voice plugin (`plugins/kinescript-kokoro` in drknowhow/kinescript) speaks a scene's narration with
Kokoro-82M and returns the time of every word, which scenes use for cues and captions. It calls the `kokoro`
package, which needs PyTorch and declares Python below 3.13. So on a 3.13 system it runs from a separate
Python 3.12 environment of about 1.4 GB. This experiment asks whether it can run on 3.13 directly, and what that
costs in word timing, audio and pronunciation control.

## Answer

Yes. An ONNX export of the same model (published by the kokoro-onnx project) runs on Python 3.13 with no
PyTorch and no 3.12 environment. Text still goes through misaki, so pronunciation links, stress marks and number flags work unchanged. Against the PyTorch
plugin, on the ten narration lines of `colbert-pooling-explainer`, all 337 word start and end times matched
exactly, and so did every phoneme. <!-- @evidence[[results/compare.json#onnx.words_exact]] --> The ONNX
route is about 840 MB on disk against 1,716 MB <!-- @evidence[[results/disk.json#torch_route_total_mb]] -->, and
loads in about 3 s against 10.4 s once the model file is in the disk cache
(2.909 s <!-- @evidence[[results/compare.json#onnxb.load_s]] -->). Speaking time is about the same as PyTorch's.

The published export does not sound like the PyTorch model as shipped. It runs kokoro's `CustomSTFT`, which
gets two things wrong against `torch.stft`/`torch.istft`. The voice comes out +2.9 dB below 1 kHz <!-- @evidence[[results/tilt.json#onnx-export vs torch312.0-1k_db]] -->
and about 2.2 dB quieter above it. The plugin replaces three of the export's weight tensors when it loads the
model, after which ONNX and PyTorch differ by at most 0.02 dB in any band, the same as two PyTorch runs (Finding 6).
The plugin change is drknowhow/kinescript's `kokoro-onnx` branch.

What would change this: a listening test; every audio comparison here is spectral.

## Findings

1. **Word times and phonemes are identical between the ONNX and PyTorch engines.** All 337 words matched to the sample on start and end,
   <!-- @evidence[[results/compare.json#onnx.words_exact]] --> the phone lists are equal,
   and total audio is 125.1 s in both <!-- @evidence[[results/compare.json#onnx.audio_s]] -->. The duration
   predictor is the same network, and the plugin's ONNX engine applies KPipeline's own `join_timestamps` rule to
   the export's `duration` output. Pace is 171.8 wpm in both <!-- @evidence[[results/compare.json#onnx.wpm]] -->.
   (round 1)
2. **misaki 0.9.4 runs on Python 3.13 once its `requires_python <3.13` is bypassed.** Phoneme links
   (`[ColBERT](/kQlbˈɛə/)`), `[she](+2)` stress, `[150](#a&#)` number flags, the espeak fallback and spaCy 3.8.18
   with `en_core_web_sm` all work. The pin is stale, and an unreleased misaki pull request (#85) enables 3.13.
   `misaki[en]` cannot be installed as-is on 3.13, though. Its `spacy-curated-transformers` 2.x needs thinc 9,
   which no released spaCy 3.8 accepts, so pip falls back to an old spaCy that fails to build from source. Only
   `G2P(trf=True)` needs that package, and Kokoro never sets it, so leave it out. (round 1)
3. **PyTorch Kokoro 0.9.4 also runs on 3.13 under the same bypass.** Its word times are identical and its audio
   is within run-to-run noise (correlation 0.9848 <!-- @evidence[[results/compare.json#torch313.corr_mean]] -->
   against 0.9849 for a second 3.12 run). That removes the separate 3.12 interpreter, but not PyTorch
   (the 3.13 venv is 1.5 GB). (round 1)
4. **The kokoro-onnx export tilts the spectrum because of kokoro's `CustomSTFT`, not the ONNX conversion.**
   PyTorch Kokoro with `disable_complex=True`, the setting the export uses, shows the same tilt: +2.9 dB below
   1 kHz and −2.62 dB at 1–3 kHz <!-- @evidence[[results/tilt.json#torch312-customstft vs torch312.1-3k_db]] -->.
   The ONNX export matches it within noise (correlation 0.9897).
   <!-- @evidence[[results/tilt.json#onnx-export vs torch312-customstft.corr]] --> The inverse transform's
   one-sided inverse DFT does not double bins 1–9 and does not divide by the overlap-added squared window
   (1.5 at n_fft 20, hop 5). Against `torch.istft`, that scales DC and Nyquist by 1.5 and every other bin by
   0.75. Rescaling those rows gives an inverse that matches `torch.istft` to 1.2e-7 inside the signal. (round 2)
5. **The remaining ±0.5 dB comes from the forward transform's Nyquist phase.** The −sin row for bin 10 is
   about 4e-15 rather than 0 in float32. That bin's phase, an input to the vocoder, then lands at −π instead of
   π in 308 of 955 frames <!-- @evidence[[results/stft_isolation.json#nyquist_phase_flips.frames_differing_over_1rad]] -->.
   With the vocoder noise seeded, Kokoro with the PyTorch forward STFT and the fixed custom inverse matches stock
   PyTorch exactly (correlation 1.0, 0.00 dB per band). The custom forward with that row zeroed does too.
   (round 3)
6. **With all three weight tensors replaced, the plugin's ONNX engine matches PyTorch within run-to-run noise.**
   Energy per band differs by 0.00, −0.01, 0.00 and −0.02 dB (0–1, 1–3, 3–6 and 6–12 kHz), at correlation
   0.9838 <!-- @evidence[[results/tilt.json#onnx vs torch312.corr]] --> against 0.9849 between two PyTorch runs.
   Mean log-spectral distance is 1.27 dB <!-- @evidence[[results/compare.json#onnx.spec_db_mean]] --> against
   1.25 dB for that pair. With only the inverse fixed (round 2) the bands were off by up to 0.52 dB. (round 4)
7. **The int8 and fp16 exports are not worth using on CPU.** int8 runs 2.9 times slower than fp32 (86.5 s). <!-- @evidence[[results/compare.json#onnx-int8.synth_s]] -->
   Its durations move too: only 71 of 337 words
   keep their times exactly. fp16 saves no time. Its initializers are renamed, so the STFT fix cannot reach it,
   and its word lists diverge on one line. (rounds 1–2)
8. **Ettin-style packing works for the vocoder, not for the duration path.** Kokoro's token tables are 114k of
   81.8M parameters, so there is nothing to gain there. With seeded runs, quantizing the 28.4M parameters that set
   durations (PL-BERT, text encoder, predictor) to int8 moved most word times: 70 of 166 tokens stayed exact, and
   some moved up to 100 ms. Quantizing the voice packs to int8 moved 18 tokens. Keeping those parts at fp16 and
   packing the 53.2M decoder weights to int8 rows (codes ±63) kept all 166 exact, at correlation 0.9822
   <!-- @evidence[[results/pack_probe.json#arms.duration path fp16, decoder int8 ±63.corr]] --> and
   within 0.2 dB per band. That is about 124 MB against 337 MB. int4 blocks on the decoder cost 0.66 dB below 1 kHz
   (correlation 0.9676). The Snake activations' `alpha` vectors must stay unpacked: one rounded to 0 gives NaN. (round 5)
9. **A remex-style rotation does not rescue the duration path, and it hurts the vocoder; it does help the voice
   packs.** A random-sign Hadamard before per-row int8 left the duration path's weight error unchanged (0.0082
   relative), so word times still moved (88 of 166 exact, against 70 without it). On the vocoder, rotated Lloyd-Max
   codes spread the error evenly across each kernel's taps, which comes out as hiss: +4.0 dB at 6–12 kHz at 6 bits,
   where unrotated int4 blocks gave +0.14 dB. On the voice packs, rotated int8 kept 165 of 166 tokens exact <!-- @evidence[[results/remex_probe.json#voice packs: int8 rows, rotated.tokens_exact]] -->
   against 148 unrotated, so 28 MB of voices pack to about 7 MB. (round 6)

## Method

**Fixture.** The ten narration lines of `colbert-pooling-explainer/scene.json` (337 words), as `kine voice` sends
them: `[[mark]]`s removed, whitespace collapsed (`make_requests.py` repeats `split_marks` and `untag`). Speaker
`bm_fable,bm_fable,bm_lewis` (a 2:1 blend), rate 0.88, British pipeline.

**Arms.** Each runs `kinescript_kokoro.synth` from the plugin, under `bench.py`, inside claude-workspace's
`scripts/jail.sh` (no network, runs as `nobody`), on 4 CPU cores:

| arm | Python | engine |
|---|---|---|
| `torch312`, `torch312b` | 3.12.13 | kokoro 0.9.4 + PyTorch 2.15.0 CPU (the parent session's setup) |
| `torch313` | 3.13.18 | the same, installed with `--ignore-requires-python` for kokoro and misaki only |
| `onnx`, `onnxb` | 3.13.18 | the plugin's ONNX engine: misaki 0.9.4 + onnxruntime 1.31.1, `kokoro-v1.0.onnx` from kokoro-onnx's `model-files-v1.1`, STFT weights replaced |
| `onnx-export` | 3.13.18 | the same with `KOKORO_ONNX_STFT=export` (the export's own weights) |
| `onnx-fp16`, `onnx-int8` | 3.13.18 | the reduced-precision files from the same release |

`stft_probe.py` adds two PyTorch arms outside the plugin: `torch312-customstft` (`KModel(disable_complex=True)`)
and `torch312-customstft-fixed` (the same with the inverse rows rescaled).

**Metrics.** `compare.py` reads, for each arm against `torch312`: per-word start and end differences, phone-list
equality, wpm over speaking time, length difference, waveform correlation and mean log-spectral distance.
`tilt.py` reads total energy and energy per band (0–1, 1–3, 3–6 and 6–12 kHz). Kokoro's vocoder adds random
noise on every call, in both engines, so audio numbers are read against a second run of the same arm, not
against 1.0. The STFT isolation in round 3 used seeded PyTorch runs, which are deterministic, so those numbers are exact.

**Disk** (`results/disk.json`): PyTorch route = the 3.12 venv (1.4 GB, of which PyTorch is 776 MB) plus the
Hugging Face weights (316 MB). ONNX route = a 3.13 venv (503 MB, mostly spaCy 127 MB and onnxruntime 67 MB)
plus the model (311 MB) and voices file (27 MB). The voices file holds the same 54 voice packs as the `.pt` files,
bit for bit.

**Setup** (venvs under `/opt`, world-readable for the jail):

```bash
# ONNX route, Python 3.13: no torch, no spacy-curated-transformers
python3.13 -m venv /opt/konnx
/opt/konnx/bin/pip install --no-deps --ignore-requires-python misaki==0.9.4
/opt/konnx/bin/pip install onnxruntime numpy phonemizer-fork espeakng-loader addict regex num2words spacy \
  https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
# model: https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1/{kokoro-v1.0.onnx,voices-v1.0.bin}
```

## Log

### Round 1: does anything run on 3.13, and do the word times hold (2026-11-12)

Asked: can Kinescript's Kokoro voice run on 3.13 without the 3.12 environment; can kokoro-onnx give word times.

kokoro-onnx 0.6.1's own word timing works only for exports with a `duration` output. The `model-files-v1.0`
release's `kokoro-v1.0.onnx` has only `audio`. The `model-files-v1.1` release's file of the same name has
`waveform` and `duration` and embeds `config.json`. kokoro-onnx phonemizes with espeak. Using its `create_timed`
directly would therefore lose misaki's links, stress and number flags. So the plugin's new engine keeps misaki,
feeds its phonemes to the graph and times words with KPipeline's own rule (ported in `onnx.py`).

Installing PyTorch kokoro on 3.13 with `pip install --ignore-requires-python ... misaki[en]` failed: pip built
spaCy from source, and blis failed to compile. The cause was the `spacy-curated-transformers` extra (Finding 2).
Installing the dependencies normally, then only kokoro and misaki with `--no-deps --ignore-requires-python`,
worked.

| arm | words exact (of 337) | wpm | load s | speak s | peak RSS MB |
|---|---|---|---|---|---|
| torch312 | 337 | 171.8 | 10.4 | 36.8 | 1,881 |
| torch313 | 337 | 171.8 | 5.3 | 38.2 | 1,886 |
| onnx (export STFT) | 337 | 171.8 | 2.8 | 31.9 | 990 |
| onnx-fp16 | 324 | 171.8 | 3.0 | 33.4 | 1,016 |
| onnx-int8 | 71 | 171.6 | 2.5 | 90.3 | 814 |

Reading at the time: timing parity holds. However, ONNX audio correlated 0.900 with PyTorch, against 0.985 between
two PyTorch runs, and its log-spectral distance was 2.89 dB against 1.25 dB. The ONNX output was also not
deterministic (0.989 between its own runs), so the gap was not just noise.

### Round 2: where the tone difference comes from (2026-11-12)

The ONNX output was +1.5 dB louder overall and −2 dB above 6 kHz, a tilt rather than noise. PyTorch with
`disable_complex=True` reproduced it exactly. Reading `kokoro/custom_stft.py` gave the predicted per-bin factors:
1.5 for DC and Nyquist and 0.75 for the rest, which is +3.5 dB below about 600 Hz and −2.5 dB above. Bins are
1.2 kHz wide at n_fft 20. The measurement was +2.9, −2.62, −2.21 and −1.99 dB. Rescaling the inverse rows by
(2 or 1) / 1.5 fixed it in PyTorch.

The plugin applies the same rescaling through `onnxruntime.SessionOptions.add_initializer`, with no `onnx`
dependency and no rewritten model file. Zeroed weights produced silence on fp32 and int8, so the override reaches
those graphs. On fp16 it was ignored, because that file's initializers are renamed.

| pair | corr | 0–1k | 1–3k | 3–6k | 6–12k dB |
|---|---|---|---|---|---|
| torch312b vs torch312 (floor) | 0.9849 | −0.00 | −0.02 | +0.01 | −0.01 |
| onnx-export vs torch312 | 0.9005 | +2.90 | −2.62 | −2.20 | −2.00 |
| onnx-export vs torch312-customstft | 0.9897 | +0.00 | −0.00 | +0.01 | −0.01 |
| onnx (inverse fixed) vs torch312 | 0.9825 | −0.24 | −0.52 | +0.30 | +0.36 |
| onnx (inverse fixed) vs torch312-customstft-fixed | 0.9838 | +0.00 | +0.00 | −0.01 | −0.00 |

### Round 3: the last half decibel (2026-11-12)

The residual ±0.5 dB was systematic, since PyTorch's own spread is ±0.02 dB. With `torch.manual_seed(1)` before
each run, Kokoro is deterministic, which allows exact A/B tests on three lines:

| STFT used by the vocoder | corr vs stock | bands dB |
|---|---|---|
| PyTorch forward + fixed custom inverse | 1.0000 | 0.00 0.00 0.00 0.00 |
| custom forward + PyTorch inverse | 0.9963 | −0.25 −0.53 +0.34 +0.37 |
| custom forward, Nyquist −sin row zeroed + PyTorch inverse | 1.0000 | 0.00 0.00 0.00 0.00 |

The custom forward transform's Nyquist phase differed by more than 1 rad from `torch.angle` in 308 of 955
frames, and in none after zeroing the row. The plugin change that also overrides `weight_forward_imag` is written
but not applied or measured: the session's permission classifier stopped the command that applied and checked it.

### Round 4: the full fix in the plugin, and an end-to-end run (2026-11-12)

The plugin's ONNX engine now also overrides `weight_forward_imag` (DC and Nyquist rows exactly 0), and the
switch is `KOKORO_ONNX_STFT=export`. The arms `onnx`, `onnxb` and `onnx-export` were rerun in the jail:

| pair | corr | 0–1k | 1–3k | 3–6k | 6–12k dB |
|---|---|---|---|---|---|
| torch312b vs torch312 (floor) | 0.9849 | −0.00 | −0.02 | +0.01 | −0.01 |
| onnxb vs onnx (floor) | 0.9857 | −0.00 | −0.00 | −0.00 | −0.00 |
| onnx (full fix) vs torch312 | 0.9838 | 0.00 | −0.01 | 0.00 | −0.02 |
| onnx-export vs torch312 | 0.9012 | +2.90 | −2.63 | −2.19 | −2.00 |

Word times were again 337 of 337 exact. This session's machine was slower than in round 1, for both engines: the
export arm took 32.3 s to speak the ten lines here against 29.8 s in round 2. The first `onnx` run's 31 s load was
a cold disk cache, and the second run loaded in 2.9 s.

End to end: `kine voice` on `colbert-pooling-explainer/scene.json`, with `KINE_VOICE_CMD` running the plugin's
ONNX engine on Python 3.13 in the jail, made all ten sounds. `kine check` then reported 0 errors and the one
warning the PyTorch voice-over also gives (`audio-peak`, −0.9 dBTP against a −1.5 ceiling). In a fresh 3.13 venv,
`pip install --ignore-requires-python ./plugins/kinescript-kokoro` installed misaki, spaCy 3.8.18 and onnxruntime
with no PyTorch (487 MB before the spaCy model), and its `python -m kinescript_kokoro` honoured phoneme links.
Kinescript's test suite fails the same 16 tests on the branch as on `main` (docs examples, fonts and schema,
which depend on this environment), and the new `tests/test_kokoro_onnx.py` passes.

### Round 5: would ettin-native's packing shrink Kokoro? (2026-11-13)

Asked: can the token tables and weights be packed the way `ettin-native` packs Ettin (int4 tables, weight-only
int8 per output row with codes held to ±63)? `pack_probe.py` fake-quantizes weights or voice packs in PyTorch and
speaks the first four narration lines with the vocoder noise seeded, so fp32 against itself is exactly 1.0 and
every difference comes from the packing. Correlation is meaningless once durations move (the audio shifts), so
read word times first.

| arm | tokens exact (of 166) | max start shift ms | corr | bands dB (0–1, 1–3, 3–6, 6–12 kHz) |
|---|---|---|---|---|
| fp32 again | 166 | 0 | 1.0000 | 0.00 0.00 0.00 0.00 |
| int8 ±63, all weights | 69 | 50 | (shifted) | +0.04 +0.04 −0.05 +0.21 |
| int8 ±127, all weights | 70 | 100 | (shifted) | +0.02 +0.02 +0.04 −0.05 |
| int8 ±63, duration path fp32 | 166 | 0 | 0.9972 | +0.03 0.00 −0.09 +0.19 |
| duration path fp16, decoder int8 ±63 | 166 | 0 | 0.9822 | +0.03 −0.01 −0.08 +0.20 |
| int4 blocks of 32, duration path fp32 | 166 | 0 | 0.9676 | −0.66 +0.10 −0.07 +0.14 |
| int4 blocks of 32, all weights | 12 | 275 | (shifted) | −0.66 +0.21 −0.12 +0.17 |
| voice packs fp16 | 166 | 0 | 0.9784 | 0.00 −0.01 +0.01 0.00 |
| voice packs int8 rows | 148 | 25 | (shifted) | −0.01 +0.01 +0.02 −0.03 |
| voice packs int4 blocks | 53 | 100 | (shifted) | +0.04 +0.50 +0.06 −0.08 |

The first run produced NaN audio for every arm that touched the decoder. The cause was the Snake activations'
`alpha` (shape (1, C, 1)): one scale over the whole tensor rounds small alphas to 0, and Snake divides by alpha.
`packable()` now keeps vectors stored as 3-D tensors (`alpha`, `weight_g`) unpacked. The fp16 arms change no word
times, but not being bit-exact they do change the seeded noise path slightly, hence correlation below 1.

Reading: the duration path and the style vectors feed a rounding step (`round(duration).clamp(min=1)`), so tiny
weight errors flip whole frames. Packing belongs on the decoder only. This is size only: weights dequantized at load
run at fp32 speed, unlike onnxruntime's dynamic int8 export, which was 2.9× slower (Finding 7).

### Round 6: rotation and Lloyd-Max levels, remex-style (2026-11-13)

Asked: would a remex-like approach (rotate, then snap to Gaussian levels) fix the rounding problem? `remex_probe.py`
packs one part at a time (the rest fp32): each row is padded to a power of two, multiplied by a random-sign
Hadamard, coded, and rotated back, which is QuIP#/QuaRot-style incoherence processing undone at load.

| arm | weight rel. error | tokens exact (of 166) | max shift ms | corr | bands dB (0–1, 1–3, 3–6, 6–12 kHz) |
|---|---|---|---|---|---|
| duration path: int8 rows | 0.0081 | 70 | 100 | (shifted) | −0.02 −0.01 +0.05 0.00 |
| duration path: int8 rows, rotated | 0.0082 | 88 | 50 | (shifted) | −0.01 +0.02 0.00 −0.02 |
| duration path: 6-bit Lloyd-Max, rotated | 0.050 | 43 | 150 | (shifted) | −0.26 −0.27 −0.02 0.00 |
| duration path: 4-bit Lloyd-Max, rotated | 0.102 | 12 | 250 | (shifted) | −0.32 −0.33 0.00 +0.03 |
| duration path: 3-bit Lloyd-Max, rotated | 0.182 | 3 | 1013 | (shifted) | −0.26 −0.06 +0.14 −0.01 |
| decoder: 6-bit Lloyd-Max, rotated | 0.056 | 166 | 0 | 0.8599 | −0.52 +0.85 +1.58 +4.03 |
| decoder: 4-bit Lloyd-Max, rotated | 0.111 | 166 | 0 | 0.7740 | −1.32 −0.20 +1.65 +3.89 |
| decoder: 3-bit Lloyd-Max, rotated | 0.193 | 166 | 0 | 0.6403 | −2.63 −1.96 +1.75 +3.05 |
| voice packs: int8 rows | — | 148 | 25 | (shifted) | −0.01 +0.01 +0.02 −0.03 |
| voice packs: int8 rows, rotated | — | 165 | 0 | (shifted) | −0.01 +0.02 +0.02 −0.01 |
| voice packs: 6-bit Lloyd-Max, rotated | — | 147 | 25 | (shifted) | 0.00 −0.07 +0.02 +0.02 |
| voice packs: 4-bit Lloyd-Max, rotated | — | 33 | 125 | (shifted) | 0.00 −0.24 −0.20 +0.21 |

Reading: rotation pays when a few coordinates set the scale (remex's case: LateOn's vectors crowd into a cone).
Kokoro's weight rows have no such outliers, so per-row int8 is already near its error floor and rotation changes
nothing. The duration path does not fail on outliers: it fails because a 0.8% weight error is enough to tip some
phonemes across a frame boundary (`round(duration)`). What would pack it is a criterion other than exact parity:
the shifts are whole 25 ms frames, which a caption cannot show. Whether shifted durations sound worse is a
listening question. The voice packs (style vectors) do carry outlier coordinates, and there rotation helps.
