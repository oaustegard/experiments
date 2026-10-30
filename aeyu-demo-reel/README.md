# aeyu-demo-reel

[`aeyu-reel.mp4`](aeyu-reel.mp4): a 46 s vertical (1080×1920, 30 fps, 7.7 MB) narrated demo reel for [aeyu.io](https://aeyu.io),
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
by `vo/assemble.py`. For each line it caps pauses inside the take at 0.22 s,
picks the take with the lowest faster-whisper WER and then the one closest to
the line's `target` (slot minus 0.25 s), and stretches it only when the change is under 4%,
using ffmpeg's Rubber Band filter with formants preserved. A take may run until
0.1 s before the next line starts. v3–v6 stretched with librosa's phase vocoder
by up to 12%, and Oskar heard "recorded from a laptop microphone in a slightly
echoey room" on the lines stretched 6–12% (six of nine). v7 leaves six lines as
recorded and stretches three by 2–3% The mixed track gets linear gain only, peaking at -2 dBFS (about -22 LUFS). Oskar, 2026-10-29,
on v2: "weirdly hurried, then with a gap between slides". Lines that merely fit
their slots ran at 180–280 wpm and left up to 1.7 s of silence before each cut.
`vo/script_v2.json` lengthens the lines so that about 160 wpm fills each slot. `vo/script.json`
gives each line's text and its slot. The reel takes the track as a prop:

```bash
bash /mnt/skills/user/making-remotion-videos/scripts/render.sh video Reel out/reel.mp4 --props='{"vo":"vo/vo_omnivoice.wav"}'   # default: vo_gemini.wav
```

- `vo_gemini.wav`: Gemini 3.8 Flash TTS, prebuilt voice Charon, style "energetic,
  brisk explainer". Line 3 reads "Best of the YEAR" (`vo/script_gemini.json`)
  because "Year bests" came out as "your best" in 3 of 3 takes.
- `vo_omnivoice.wav`: OmniVoice (k2-fsa/OmniVoice 0.2.1, the default engine of
  [debpalash/VoiceStudio](https://github.com/debpalash/VoiceStudio)), run on CPU
  inside `claude-workspace/scripts/jail.sh` (`vo/omnivoice_run.py`, weights
  downloaded outside the jail). Tag-designed voices came out flat, so the voice
  is cloned from a 6 s Gemini Charon clip in the energetic style, which shares
  no words with the script. Unpaced it ran at 211 wpm. With `duration=` set
  per line it runs at 182 wpm. Each line took about 17–36× real time on 4
  cores. The name uses the inline phoneme override `[EY1 Y UW0] dot I-O.`,
  because every plain spelling came out "eye-you".

**Current track: `vo_v7.wav`** (`vo/script_v5.json`, the composition default).
It uses Gemini prebuilt voice `en-us-training-6` ("Training Voiceover 6"), voice
7 of Oskar's eight-voice audition. v4 used voice 6 by mistake. The style is
"warm and lively, varied intonation, natural conversational pace, not rushed".
All nine lines have zero WER apart from "50" for "fifty". Loudness is −22.0
LUFS. It was −16.6 in v5: single-pass `loudnorm` to −16 LUFS needed about 5 dB
of limiting on TTS that already peaks near 0 dBFS, and Oskar heard that as
"amplified too much". v6 uses linear gain only.

The name is spelled, not spoken. "aeyu" is pronounced "AY-yoo", but every
auditioned voice mangled it. Opening: "A, E, Y, U, dot I, O. The sound you make
at the top of the climb." In v4 the opening dropped the name, which Oskar
caught. The title scene runs 7 s, and each letter of the wordmark lights as it
is spoken (`TITLE_AT`). Closing: "And it's all free. A, E, Y, U, dot I, O.
Every vowel on your keyboard, left to right." That is true of QWERTY: A on the
home row, then E, Y, U, I, O along the top row. The keyboard lights each vowel
as it is spoken (`VOWEL_AT`). Both frame lists were measured on the track with
energy segmentation (`librosa.effects.split`, top_db 20–30) and whisper word
timestamps. Onset detection alone was too noisy on the closing line.
`assemble.py` keeps 60 ms before each detected line start, because a hard cut
at the trim threshold clipped soft onsets.

The ending also changed per Oskar: "Try the free demo" sounded as if only the
demo were free. The privacy scene now reads "No ads · No tracking / Just you
and your Strava data / On your device, in your browser", and the button reads
"All free · aeyu.io".

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
