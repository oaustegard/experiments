#!/bin/zsh
# usage: assemble.sh FRAMES_DIR OUT.mp4 [first_frame] [nframes]
# Encodes the composited frames with the song's audio (stream-copied), a touch of grade, vignette and grain; the
# "everybody's quiet" beat (127.1-130.8 s) is desaturated.
cd "$(dirname "$0")"
FR=${1:-overlay_out}; OUT=${2:-final/geezer-goon-v3-720p.mp4}
F0=${3:-0}
N=${4:-}
mkdir -p "$(dirname "$OUT")"
VF="hue=s='if(between(t,127.1,130.8),0.18,1)',eq=saturation=1.1:contrast=1.04:gamma=0.98,vignette=angle=PI/5:eval=init,noise=alls=7:allf=t"
EXTRA=()
[[ -n "$N" ]] && EXTRA=(-frames:v $N)
ffmpeg -y -loglevel error -framerate 30 -start_number $F0 -i "$FR/f_%05d.png" -i song.m4a -map 0:v -map 1:a -vf "$VF" \
  -c:v libx264 -preset medium -crf 18 -pix_fmt yuv420p -c:a copy -shortest $EXTRA -movflags +faststart "$OUT"
ffprobe -v error -show_entries format=duration,size -of default=nw=1 "$OUT"
