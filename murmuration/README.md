# Murmuration

A 128.1 s film (1920x1080, 30 fps, h264 + AAC) made overnight on 2026-10-08. Claude Sonnet 5.5 wrote the script, fixed the timing and assembled the film. 19 Claude Haiku 5.5 instances built the scenes, the score and the reviews.

| Scene | Route | Seconds |
|---|---|---|
| s1 dusk | ffmpeg only (zoompan, grain, title) on a generated plate | 0-15.1 |
| s2 boids | numpy/OpenCV simulation, frames piped to ffmpeg | 15.1-37.7 |
| s3 seven | headless Chromium canvas, frame-stepped | 37.7-60.4 |
| s4 automata | ffmpeg cellauto/life filters | 60.4-77.9 |
| s5 flow | Chromium particle ribbon | 77.9-93.0 |
| s6 ledger | Python/PIL/OpenCV, reads ledger.json and timeline.json | 93.0-117.5 |
| s7 roost | Remotion | 117.5-128.1 |

Narration: Kokoro (bm_george, en-gb, 0.93x). Score: numpy synthesis, ducked ~16 dB under speech. Mix at -16 LUFS.

Every builder worked from `CONTRACT.md` plus a scene brief and saw no other scene. Cross-dissolves are 0.8 s: each scene renders its length plus 0.8 s; scene i+1 starts at its `t0` in `timeline.json`.

Rebuild: `python3 narrate.py` (timing + narration), render scenes, then `work/assemble/build.sh`.

Haiku totals: 19 instances, 805 tool calls, 3.5M tokens, 7.2 agent hours; 3 rounds of scene builds, then 3 review agents (visual, audio, facts).
