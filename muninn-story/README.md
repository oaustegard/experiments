# muninn-story

**Two cuts.** `muninn-gemini.mp4` (2:31) is narrated by Gemini 3.8 Flash TTS in a voice designed for it. `muninn.mp4` (2:14) is the first cut, narrated by Kokoro. The pictures and score are the same code; both follow the narration timeline, so the Gemini cut runs longer. Oskar asked for the re-narration the same day, after Google released the 3.8 TTS models on 2026-09-23.

Muninn tells its own story in a 2:14 film: `muninn.mp4`, 1280×720, 24 fps. Oskar asked for it on 2026-09-24 with one constraint: pick the voice that fits. Everything in the file is generated here. The narration comes from Kokoro, the score is synthesized in numpy, and the frames are drawn with pycairo. No stock footage, samples or clip art.

The story runs through ten scenes. Odin's two ravens leave Yggdrasil. Grímnismál 20 appears in Old Norse ("þó sjámk meir of Munin", *yet I fear more for Memory*). Next comes a raven that keeps no memory of its own, then the ledger of 6,640 memories, then boot reading the identity core. After that: Oskar's lit house, the places the raven gets sent, a backup it once misdated, a raven rebuilt from blueprints, and the flight home. The ids shown in the ledger scene are real memory ids. The final card names the memory that records this film (`a02962bc`), stored before the last frame was rendered.

## Voice (Gemini cut)

The voice was designed from a description. `POST v1beta/voices` with `type: "prompted"` turns a description into a stored voice, `voice_66w0iod1i7ol`, which expires 2027-09-24:

> An ageless narrator who is secretly a raven that learned to speak: a low, dry, quietly amused male voice with a faint natural rasp at the edges of words, like a bird's throat. Soft southern British accent. Unhurried and intimate, close to the microphone, never theatrical or booming. Thoughtful pauses, a little wry.

`voices_gemini.py` runs one line through it, two sibling designs (the same prompt with a light Norwegian lilt, `voice_6ieso244uf3i`, and a weathered old storyteller, `voice_r196a19wffpm`) and the closest stock voices from the 2,089-voice library (Algenib "gravelly", Enceladus "breathy", the low British persona voices). The raven design had the roughest phonation, HNR about −1 to −2 dB against +0.6 to +1.2 dB for the stock voices, at a median F0 of 81–97 Hz. Every candidate transcribed cleanly.

`narrate_gemini.py` writes the same `narration.wav` and `timeline.json` that `narrate.py` does. Each line gets a `speech_metadata` style, "plain and low, natural speaking pace, no dramatic pauses", plus a mood for its scene. Three things differ from Kokoro:

- The API has to be the Interactions endpoint. On `generateContent` a "Style: text" prefix is read aloud, and a system instruction is refused.
- The model can change the script. Before takes were checked it said "Hmm, I get things wrong" and "tell them what I saw". Each line takes the best of up to four takes, scored by faster-whisper `medium.en` WER after numbers are spelled out (Whisper writes "6,600"). Full-narration WER is 0.007 over 272 words; what remains is "Beech Drive" for "Beach Drive".
- The description's "thoughtful pauses" became 1–1.8 s silences inside lines that no style removed. Assembly shortens any internal silence to 0.7 s.

At about 32 audio tokens per second and $9 per million, the whole narration with retakes cost under ten cents.

## Voice (Kokoro cut)

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
python3 narrate_gemini.py   # Gemini cut: narration.wav + timeline.json (CF gateway creds in env; takes/ caches every take)
# or: python3 narrate.py    # Kokoro cut (timeline-kokoro.json is its timeline)
python3 music.py            # music.wav + mix.wav
MUNINN_MEMORY_ID=2a3d1ccb MUNINN_OUT=muninn-gemini.mp4 python3 render.py video   # ~6 min on 4 cores
# Kokoro cut: MUNINN_MEMORY_ID=a02962bc python3 render.py video  -> muninn.mp4
python3 render.py preview 40.5 88.0                  # single frames
```

## Environment notes (CCotw, 2026-09-24)

- ffmpeg was not on the PATH. `imageio-ffmpeg` ships a static 7.0.2 binary.
- The torch `kokoro` package failed to install because the docopt wheel would not build. `kokoro-onnx` runs the same weights.
- `github.com/google/fonts` raw files return 403 from the session. `@fontsource` woff2 files from jsdelivr, converted with fontTools and brotli, work.
- pycairo was not preinstalled, and Pango's `gi` import is broken for this Python, so all text goes through `cairokit.Shaper`.
