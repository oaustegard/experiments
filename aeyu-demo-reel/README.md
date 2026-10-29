# aeyu-demo-reel

[`aeyu-reel.mp4`](aeyu-reel.mp4): a 35.5 s vertical (1080×1920, 30 fps, 5.5 MB) demo reel for [aeyu.io](https://aeyu.io),
built in [Remotion](https://github.com/remotion-dev/remotion) from captures of
the live app's `/demo` mode at iPhone size. It is the worked example for the
`making-remotion-videos` skill in `oaustegard/claude-skills`.

Oskar, 2026-10-29: "Build up a set of protocols for making use of
https://github.com/remotion-dev/remotion. Then create a demo reel video for
aeyu.io using it, with a phone form factor."

## Timeline

| Scene | Frames | Read |
|---|---|---|
| Title | 90 | aeyu.io, "The sound you make at the top of the climb" |
| Hook | 120 | "Strava crowns the fastest. aeyu.io celebrates the rest of your rides." |
| Dashboard | 150 | "Awards for the rest of us": scroll the activity list, tap River Loop |
| Ride replay | 150 | "Ride it again": the 3D power ribbon playing |
| Awards | 150 | "50 ways to be proud": scroll to the ride's awards |
| Power | 135 | "Power that knows you're tired": the fatigue-binned power curve |
| Heatmap | 135 | "Every ride, one map": the heatmap filling in by date |
| Privacy | 120 | "Your rides stay in your browser." |
| CTA | 135 | icon, wordmark, "Try the demo · aeyu.io/demo" |

Transitions are 15 frames and overlap, so the total is the sum minus 8 × 15
= 1065 frames. `Reel.tsx` derives it from the arrays.

## Rebuild

```bash
npm ci
NODE_PATH=$(npm root -g) node capture/capture.cjs     # ~12 min; writes public/shots/
NODE_PATH=$(npm root -g) node capture/measure.cjs     # public/shots/layout.json
bash /mnt/skills/user/making-remotion-videos/scripts/render.sh video Reel out/aeyu-reel.mp4   # 79 s on 4 cores
```

The stills and `layout.json` in `public/shots/` are committed. The two frame
sequences (67 MB) are not: `capture.cjs replay` and `capture.cjs heatmap`
regenerate them in about 8 minutes. `AEYU_URL` points the capture at another
deployment.

## Voiceover

`public/vo/` holds the narration tracks, each mixed to the reel's 35.5 s timeline
by `vo/assemble.py`: best of three takes per line by faster-whisper WER, then the
longest take that fits its slot, then loudnorm to -16 LUFS. `vo/script.json`
gives each line's text and its slot. The reel takes the track as a prop:

```bash
bash /mnt/skills/user/making-remotion-videos/scripts/render.sh video Reel out/reel.mp4 --props='{"vo":"vo/vo_gemini.wav"}'
```

- `vo_gemini.wav`: Gemini 3.8 Flash TTS, prebuilt voice Charon, style "energetic,
  brisk explainer". Line 3 reads "Best of the YEAR" (`vo/script_gemini.json`)
  because "Year bests" came out as "your best" in 3 of 3 takes.
- OmniVoice (k2-fsa/OmniVoice 0.2.1, the default engine of
  [debpalash/VoiceStudio](https://github.com/debpalash/VoiceStudio)): rendering.
  `vo/omnivoice_run.py` runs inside `claude-workspace/scripts/jail.sh` with the
  weights downloaded outside it.

"aeyu" is voiced as "ay-yoo", an unconfirmed guess at the name's pronunciation.

## Files

- `capture/capture.cjs`: stills (`fullPage`) and the two motion shots. The
  motion shots step Playwright's fake clock one output frame at a time.
- `capture/measure.cjs`: element boxes for the tap point, scroll targets and
  sticky header height.
- `src/Reel.tsx`: the timeline. `src/Phone.tsx`, `ScrollShot.tsx`,
  `FrameSequence.tsx`, `Tap.tsx` and `Text.tsx` hold the components.
- `src/theme.ts`: aeyu.io's own `:root` colours and its three fonts, vendored
  in `public/fonts/`.

## What went wrong on the way

Each of these is recorded with its signal in the skill.

1. `@remotion/google-fonts` failed every render with
   `ERR_CERT_AUTHORITY_INVALID`, because the render browser does not trust the
   egress proxy's CA. Fixed by vendoring the fonts.
2. A real-time CDP screencast of the WebGPU heatmap gave 4 fps. Replaced by
   fake-clock stepping.
3. Playwright's `iPhone 14` preset viewport is 390×664, so the first motion
   frames covered only three quarters of the phone screen.
4. The play-button locator `aria-label*="lay"` hit "Make a 30 second replay
   video" first. The first replay capture was 150 frames of an export counter
   over a still scene.
5. Both animations finished early in the captures (the replay by frame 13 of
   150). Playwright's `clock.install()` keeps the fake clock ticking at real
   speed, and each WebGPU screenshot takes about a second. `clock.pauseAt()`
   freezes it. Two wrong theories came first: that the "20s"/"30s" labels were
   nominal, and that rAF timestamps were mismatched.
6. The first contact sheet had every caption's subline under the phone's top
   edge. The phone went to 86% scale.
