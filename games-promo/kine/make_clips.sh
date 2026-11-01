#!/bin/sh
# Encode each captured frame directory (../public/shots/<game>/f0000.jpg...) as clips/<game>.mp4 at 30 fps.
set -e
cd "$(dirname "$0")"; mkdir -p clips
for d in ../public/shots/*/; do
  g=$(basename "$d" | sed 's/\.html$//')
  [ -f "$d/f0000.jpg" ] || continue
  ffmpeg -v error -y -framerate 30 -i "$d/f%04d.jpg" -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -c:v libx264 -crf 17 -preset medium -pix_fmt yuv420p "clips/$g.mp4"
  echo "$g $(ls "$d" | wc -l) frames"
done
