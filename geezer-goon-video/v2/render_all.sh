#!/bin/bash
# Resumable full render: 640-frame chunks, 4 parallel Chromium workers, then concat + mux.
cd "$(dirname "$0")"
mkdir -p chunks
N=7481; C=640
for i in $(seq 0 $(( (N + C - 1) / C - 1 ))); do
  f0=$((i * C)); f1=$(( (i + 1) * C )); [ $f1 -gt $N ] && f1=$N
  out=chunks/c$(printf %02d $i).mp4
  [ -s "$out" ] && continue
  echo "$f0 $f1 $out"
done | xargs -P 4 -L 1 sh -c 'python3 capture.py chunk $0 $1 $2.tmp.mp4 && mv $2.tmp.mp4 $2'
ls chunks/c??.mp4 | sed "s#chunks/##; s#.*#file '&'#" > chunks/list.txt
ffmpeg -hide_banner -loglevel error -y -f concat -safe 0 -i chunks/list.txt -i /root/.claude/uploads/77b9f837-2e64-53f0-a9fb-8e1cf3f25d88/a12edbcf-Geezer_Goon.m4a \
  -map 0:v -map 1:a -c:v copy -c:a aac -b:a 192k -shortest -movflags +faststart out_1080p.mp4 && echo DONE && ls -la out_1080p.mp4
