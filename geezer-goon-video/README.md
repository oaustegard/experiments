# Geezer Goon — animated music video

An animated video for Oskar's song about the Wednesday G2 ("Geezer Goon") group
ride out of Kensington, MD. The video is the release asset
[`geezer-goon-video-v1`](https://github.com/oaustegard/experiments/releases/tag/geezer-goon-video-v1):
1080p (104 MB) and 720p (27 MB), 4:09, 30 fps.

Flat vector animation drawn frame by frame with pycairo and encoded with
ffmpeg. The riders are side-on, the sky runs from golden hour to night over the
song, and the moon rises at the end. The lyrics light up word by word. Named
riders get tags when the lyrics mention them. A bike computer shows ride
clock (6:15 → 7:32 PM), miles, speed and climb. Scene props follow the
lyrics: street signs, the Jones Bridge and East-West traffic lights, the
Beach Drive half-gates (with a top-down inset of the pack going single file
through the middle gap on each "GATE UP!"), potholes on Ridge and Ross, the
road tilting up Mormon Hill, a flipping Tue/Thu calendar in the bridge.

## Timing source

The song's `.m4a` carries a `mov_text` subtitle track with line-timed lyrics
and zero-length `[Section]` cues, so no speech alignment was needed. Word
timing inside a line is spread by word length. Beats and energy come from
librosa (150 BPM) and drive text pulse and a small camera bob.

## Rebuild

The song itself is not in this repo.

```bash
pip install pycairo librosa imageio-ffmpeg
ln -sf "$(python3 -c 'import imageio_ffmpeg as f; print(f.get_ffmpeg_exe())')" /usr/local/bin/ffmpeg
# fonts: Anton, Bebas Neue, Permanent Marker (Google Fonts TTFs) into ~/.fonts, then fc-cache -f
ffmpeg -i Geezer_Goon.m4a -map 0:1 subs.srt
ffmpeg -i Geezer_Goon.m4a -ac 1 -ar 16000 song16k.wav
python3 analyze.py
python3 render.py still 47 121 178          # spot-check frames
# 7481 frames; ~0.25 s each at 1080p, so split across cores:
for i in 0 1 2 3; do python3 render.py chunk $((i*1871)) $(((i+1)*1871)) chunk$i.mp4 & done; wait
printf "file 'chunk%d.mp4'\n" 0 1 2 3 > list.txt
ffmpeg -f concat -safe 0 -i list.txt -i Geezer_Goon.m4a -map 0:v -map 1:a -c:v copy -c:a aac -b:a 192k -shortest out.mp4
```

## Snags

- Python's `hash()` is salted per process. Used for animation phase, it made
  each chunk's pedalling start at a different angle; phases use a character
  sum instead.
- A fade-out that reaches scale 0 makes cairo raise `invalid matrix`; every
  scaled overlay returns early below k = 0.01. The first full render lost 20 s
  of chunk 0 to this and the concat came out 3:50 instead of 4:09.
- "GATE UP!" was first drawn as a boom barrier swinging up. It is a shouted
  warning: gates close part of Beach Drive from each side, and the pack rides
  single file through the gap in the middle.
