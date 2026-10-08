#!/bin/bash
# Usage: build.sh  -> out/murmuration.mp4. Needs final/s1..s7.mp4, audio/music.wav, timeline.json
set -e
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$ROOT"
python3 work/assemble/mix_audio.py
python3 - <<'P' > work/assemble/filter.txt
import json
t = json.load(open("timeline.json")); S = t["scenes"]; X = t["xfade"]
parts = [f"[{i}:v]settb=AVTB,fps=30,format=yuv420p[v{i}]" for i in range(len(S))]
prev = "v0"
for i in range(1, len(S)):
    out = f"x{i}" if i < len(S) - 1 else "vout"
    parts.append(f"[{prev}][v{i}]xfade=transition=fade:duration={X}:offset={S[i]['t0']:.3f}[{out}]")
    prev = out
print(";".join(parts))
P
TOTAL=$(python3 -c "import json;print(json.load(open('timeline.json'))['total'])")
ffmpeg -y -hide_banner -loglevel error $(for i in 1 2 3 4 5 6 7; do echo -i final/s$i.mp4; done) \
  -filter_complex_script work/assemble/filter.txt -map "[vout]" -t $TOTAL \
  -c:v libx264 -preset medium -crf 22 -maxrate 9M -bufsize 18M -pix_fmt yuv420p -r 30 -an out/video_only.mp4
# audio: music ducked by narration, then loudnorm
ffmpeg -y -hide_banner -loglevel error -i audio/music.wav -i audio/voice_track.wav -filter_complex \
 "[0:a]atrim=0:$TOTAL,asetpts=N/SR/TB,volume=1.0[m];[1:a]atrim=0:$TOTAL,asetpts=N/SR/TB,asplit=2[vk][vm];\
[m][vk]sidechaincompress=threshold=0.02:ratio=6:attack=40:release=500:makeup=1[md];\
[md]volume=0.9[md2];[md2][vm]amix=inputs=2:duration=longest:normalize=0:weights='1 1.6'[mx];\
[mx]loudnorm=I=-16:TP=-1.5:LRA=11,aresample=48000,afade=t=in:d=0.5,afade=t=out:st=$(python3 -c "print($TOTAL-3)"):d=3[a]" \
 -map "[a]" -ar 48000 -c:a pcm_s16le out/audio_mix.wav
ffmpeg -y -hide_banner -loglevel error -i out/video_only.mp4 -i out/audio_mix.wav -c:v copy -c:a aac -b:a 192k -ar 48000 -movflags +faststart -shortest out/murmuration.mp4
touch logs/assemble.done
