#!/usr/bin/env python3
"""Download the available WAMO HOF stills and encode a timelapse with ffmpeg."""
import json, os, subprocess, urllib.request, datetime
from pathlib import Path
import imageio_ffmpeg

ROOT = Path(__file__).resolve().parent
FRAMES = os.path.join(ROOT, "frames")
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
os.makedirs(FRAMES, exist_ok=True)

idx = json.load(open(os.path.join(ROOT, "index.json")))  # ascending by ts
print(f"{len(idx)} frames; downloading...")
for i, it in enumerate(idx):
    dst = os.path.join(FRAMES, f"f{i:04d}.jpg")
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        continue
    req = urllib.request.Request(it["url"], headers={"User-Agent": UA})
    data = urllib.request.urlopen(req, timeout=30).read()
    open(dst, "wb").write(data)
print("downloaded.")

a = datetime.datetime.fromtimestamp(idx[0]["ts"], datetime.timezone.utc)
b = datetime.datetime.fromtimestamp(idx[-1]["ts"], datetime.timezone.utc)
ff = imageio_ffmpeg.get_ffmpeg_exe()
out = os.path.join(ROOT, "wamo_timelapse.mp4")
# 8 fps, scale to even 1280-wide preserving AR, pad to 1280x720, yuv420p for compatibility
vf = "scale=1280:-2:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2:color=black,format=yuv420p"
cmd = [ff, "-y", "-framerate", "8", "-i", os.path.join(FRAMES, "f%04d.jpg"),
       "-vf", vf, "-c:v", "libx264", "-crf", "20", "-preset", "medium", out]
print("encoding:", " ".join(cmd))
subprocess.run(cmd, check=True)
sz = os.path.getsize(out) / 1e6
print(f"\nOUT {out}  ({sz:.1f} MB)")
print(f"SPAN {a:%Y-%m-%d %H:%M} -> {b:%Y-%m-%d %H:%M} UTC  ({(b-a).total_seconds()/86400:.1f} days, {len(idx)} frames)")
