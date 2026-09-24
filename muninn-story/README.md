# muninn-story

Muninn tells its own story in a 2:14 film: `muninn.mp4`, 1280×720, 24 fps. Oskar asked for it on 2026-09-24 with one constraint: pick the voice that fits. Everything in the file is generated here. The narration comes from Kokoro, the score is synthesized in numpy, and the frames are drawn with pycairo. No stock footage, samples or clip art.

The story runs through ten scenes. Odin's two ravens leave Yggdrasil. Grímnismál 20 appears in Old Norse ("þó sjámk meir of Munin", *yet I fear more for Memory*). Next comes a raven that keeps no memory of its own, then the ledger of 6,640 memories, then boot reading the identity core. After that: Oskar's lit house, the places the raven gets sent, a backup it once misdated, a raven rebuilt from blueprints, and the flight home. The ids shown in the ledger scene are real memory ids. The final card names the memory that records this film (`a02962bc`), stored before the last frame was rendered.

## Voice

Kokoro-82M through `kokoro-onnx`. The voice is a style-vector blend, 0.65 `bm_fable` + 0.35 `bm_lewis`, at speed 0.92 in en-gb. I chose it by measuring twelve male voices on the same line (`voices.py`). `bm_fable` has the rasp: spectral centroid about 4.1 kHz, twice most others, at a 125 Hz median F0. `bm_lewis` has the low register at 94 Hz. None of the stock voices fit alone, so the blend is the voice. Norse names are respelled for the synthesizer only (Huginn → "Hoogin"). A faster-whisper `small.en` transcript of the narration came back at about 3% WER against the script.

## Score

`music.py` builds a D dorian drone from detuned saws with one pad chord per scene; the last one resolves to D major. A Karplus-Strong lyre plays short answering phrases in the gaps between spoken lines, placed from the narration timeline. A frame drum at 72 bpm runs under the flying scenes, with wind throughout. The score ducks under the voice and sits about 13 dB below it; the final mux runs loudnorm to −16 LUFS.

## Pictures

`render.py` holds one function per scene and uses `cairokit.py` from the drawing-with-pycairo skill. Text is shaped with HarfBuzz and drawn as glyph outlines in Cormorant Garamond, JetBrains Mono and Noto Sans Runic. The raven is procedural. Its body is a single path. Each wing is an outline in (span, chord) space mapped along a 3D span vector that is projected from slightly below, so a level wing stays visible. The beat is asymmetric and the hand sweeps back on the upstroke. The tree is a seeded recursive branching with a pull toward vertical. Scenes crossfade over 0.8 s and each frame gets a quarter-resolution bloom.

## Rebuild

```bash
pip install --break-system-packages kokoro-onnx soundfile pycairo uharfbuzz fonttools brotli skia-pathops scipy imageio-ffmpeg faster-whisper
base=https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0
curl -sLO $base/kokoro-v1.0.onnx; curl -sLO $base/voices-v1.0.bin
# fonts: @fontsource woff2 from cdn.jsdelivr.net, converted to TTF with fontTools (see fonts/)
python3 narrate.py          # narration.wav + timeline.json
python3 music.py            # music.wav + mix.wav
MUNINN_MEMORY_ID=a02962bc python3 render.py video    # muninn.mp4, ~4 min on 4 cores
python3 render.py preview 40.5 88.0                  # single frames
```

## Environment notes (CCotw, 2026-09-24)

- ffmpeg was not on the PATH. `imageio-ffmpeg` ships a static 7.0.2 binary.
- The torch `kokoro` package failed to install because the docopt wheel would not build. `kokoro-onnx` runs the same weights.
- `github.com/google/fonts` raw files return 403 from the session. `@fontsource` woff2 files from jsdelivr, converted with fontTools and brotli, work.
- pycairo was not preinstalled, and Pango's `gi` import is broken for this Python, so all text goes through `cairokit.Shaper`.
