# games-promo

[`games-promo.mp4`](games-promo.mp4): a 55 s vertical (1080×1920, 30 fps, 14 MB) narrated promo for the 17
browser games at [austegard.com/games](https://austegard.com/games/), composed and rendered in
[Kinescript](https://github.com/drknowhow/kinescript) 1.4.0 from gameplay captured from the live games.

Oskar, 2026-11-01: "Create a promo video with sound for the collection of games in austegard.com/games.
First re-read the READMEs then craft a narrative, then use your kine video skills."

## Narrative

[`STORYBOARD.md`](STORYBOARD.md). The game READMEs share three ideas and the film walks through them in
order: the phone is the controller (nine tilt games), the GPU does the work (gPu(doom), 10,000 Lemmings),
and nothing needs installing or a server (Mahjong and Poker tables joined by QR). It closes on the count
and the address.

## Pipeline

| stage | how | files |
|---|---|---|
| Capture | Playwright on SwiftShader WebGPU, fake clock stepped frame by frame, one script per game driving it with keys, synthetic tilt or the game's debug hook | `capture/lib.cjs`, `capture/games/*.cjs` |
| Voice-over | 11 lines, Gemini 3.8 Flash TTS voice Puck, 4 takes each, picked by faster-whisper WER and slot fit (the aeyu `assemble.py`) | `vo/script.json`, `vo/assemble.py`, `kine/vo/l01…l11.wav` |
| Score | original chiptune synthesized in numpy at 109.09 bpm, so one 4/4 bar is one 66-frame montage shot and every cut lands on a beat | `audio/music.py` → `kine/music/score.wav` |
| Compose | `build_scene.py` writes `promo.json` from one timeline array; the site's own fonts (Fraunces, DM Sans, JetBrains Mono, instanced to static TTFs) and dark-sheet colours, contour lines drifting behind | `kine/build_scene.py`, `kine/promo.json` |
| Mix | Kinescript's managed mix: voice at −18 LUFS, music ducked 14 dB under it, −16.1 LUFS / −2.0 dBTP overall | `audio` in `promo.json` |
| Render | `kine render -q high -j 4`, CPU backend: 1650 frames in 231 s (7.2 fps), then re-encoded at crf 23 for size | |

## Rebuild

```bash
# site served at http://localhost:8080 from an oaustegard.github.io checkout
node capture/run.cjs ball-maze plumb ...       # → public/shots/<game>/f0000.jpg…
kine/make_clips.sh                              # → kine/clips/<game>.mp4
python3 kine/build_scene.py                     # → kine/promo.json
kine check kine/promo.json && kine look kine/promo.json --audio && kine render kine/promo.json -o out.mp4 -q high
```

Captures (306 MB) and clips (55 MB) are not committed. Kinescript ran inside `claude-workspace/scripts/jail.sh`.

## Findings

- **Kinescript needs skia-python ≥ 144.** 138.0 fails at `space.py:79` (`skia.M44` has no `Rows`), reported
  as a Kinescript bug. `pyproject.toml` says `>=87`. It also still needs the system `libegl1` on the CPU backend.
- **A group's children are drawn from its anchor point, not its top-left corner.** A centre-anchored phone
  group needs its video child at `(-w/2, -h/2)`. `kine check` caught it as text-offscreen on a placeholder.
- **`loop: true` does not lengthen a video layer.** A video without `dur` ends with its trimmed clip even
  when looping; give it `dur`.
- **`kine check`'s tiny-text and read-time warnings changed the edit:** type went to ≥ 44 px and the
  montage sublines to five words or fewer, so they read in a 2.2 s shot.
- **Freezing Playwright's clock after load breaks most games**: the frame-time delta goes negative and the
  game stalls or flings sprites. Install the fake clock, reload, then pause. GPU games also need a wait for
  the GPU queue before each screenshot (`capture/games/_gpu.cjs`).
- **Staged shots**: Doom is keyboard play with `iddqd`/`idkfa` (the `?bot` TRM planner fell behind on
  SwiftShader); Asteroids 3D has 14 rocks added through its test hook; Mahjong and Poker use `?renderer=2d`
  because WebGPU dropped tiles under load.
