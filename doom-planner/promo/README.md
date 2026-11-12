# Doom by Rollouts: the narrated video

An 87-second, 1920×1080 video about this experiment, narrated in Muninn's
voice. It asks whether Haiku can play Doom, shows how the planner decides,
then shows each player on E1M1 with its numbers: the planner alone, Haiku
choosing from the screen, and Haiku choosing from the planner's played-out
futures. It closes on the answer. Made with
[Kinescript](https://github.com/drknowhow/kinescript) 2.5.1 and its Kokoro
voice plugin running on onnxruntime (Python 3.13, no PyTorch).

## Where the pictures and numbers come from

- **Footage** is cut from the runs' own recordings (`runs/<run>/video.mp4`):
  the start of `e1m1-explore`, its exit at 2:43, `e1m1-haiku-screen` at
  1:34 and 3:58, and `e1m1-haiku-plan` at its exit at 4:27.
- **The six option tiles** show the six options of decision 136 of
  `e1m1-explore` (tic 1089, 31 s in): each option's score and its three
  scenario scores, from `decisions.jsonl`. The footage inside the tiles is
  other moments of real play, not those options' futures.
- **The stat panels** come from `results/summary.json`. The two Haiku quotes
  are Haiku's own notes from `runs/e1m1-haiku-screen/judge.jsonl`.
- **The raven** is a 40×25 pixel sprite drawn in `raven.py`.
- **The drone** under the narration is brown noise and two low sine tones
  made with ffmpeg.

## Files

| file | contents |
|---|---|
| `build_scene.py` | writes the Kinescript `scene.json`: narration lines with word cues, the layers cued to those words, captions |
| `raven.py` | draws `raven.png` |

## Rebuild

The scene expects a folder holding `clips/`, `fonts/`, `raven.png` and
`drone.wav` (the commands that made them are in this folder's history: ffmpeg
cuts of the run videos, and Black Ops One, Saira Condensed and IBM Plex Mono
from the Google Fonts repository). In this setup `kine` runs inside the
sandbox `scripts/jail.sh` from `oaustegard/claude-workspace`, with the Kokoro
ONNX model files fetched beforehand.

```sh
python3 build_scene.py OUT_DIR
kine voice OUT_DIR/scene.json        # speaker bm_fable,bm_fable,bm_lewis, rate 0.95
kine check OUT_DIR/scene.json        # 0 errors
kine render OUT_DIR/scene.json -o doom-by-rollouts.mp4 -q high -j 4
```
