#!/usr/bin/env python3
"""Download the user-supplied WAMO HOF stills (filenames.txt) and encode a timelapse."""
import os, re, subprocess, datetime, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import imageio_ffmpeg

ROOT = Path(__file__).resolve().parent
FRAMES = os.path.join(ROOT, "frames_full")
BASE = "https://www.earthcam.com/hof/dc/washingtonmonument/"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
os.makedirs(FRAMES, exist_ok=True)

names = [l.strip() for l in open(os.path.join(ROOT, "filenames.txt")) if l.strip()]
# sort ascending by the ms-epoch prefix (file is newest-first)
def ts(n): return int(n.split("_")[0])
names.sort(key=ts)
a = datetime.datetime.fromtimestamp(ts(names[0]) / 1000, datetime.timezone.utc)
b = datetime.datetime.fromtimestamp(ts(names[-1]) / 1000, datetime.timezone.utc)
print(f"{len(names)} frames  {a:%Y-%m-%d %H:%M} -> {b:%Y-%m-%d %H:%M} UTC  ({(b-a).total_seconds()/86400:.1f} days)")

def dl(i_name):
    i, name = i_name
    dst = os.path.join(FRAMES, f"f{i:04d}.jpg")
    if os.path.exists(dst) and os.path.getsize(dst) > 1000:
        return (name, "cached")
    req = urllib.request.Request(BASE + name, headers={"User-Agent": UA})
    try:
        data = urllib.request.urlopen(req, timeout=30).read()
        if len(data) < 1000:
            return (name, f"tiny:{len(data)}")
        open(dst, "wb").write(data)
        return (name, "ok")
    except Exception as e:
        return (name, f"ERR:{e}")

ok = bad = 0
with ThreadPoolExecutor(max_workers=16) as ex:
    futs = [ex.submit(dl, (i, n)) for i, n in enumerate(names)]
    for f in as_completed(futs):
        name, status = f.result()
        if status in ("ok", "cached"):
            ok += 1
        else:
            bad += 1
            print("  ", name, status)
print(f"downloaded ok={ok} bad={bad}")

# renumber contiguously over what actually landed (ffmpeg needs gapless %04d)
have = sorted(f for f in os.listdir(FRAMES) if f.endswith(".jpg"))
SEQ = os.path.join(ROOT, "seq")
os.makedirs(SEQ, exist_ok=True)
for f in os.listdir(SEQ):
    os.remove(os.path.join(SEQ, f))
for j, f in enumerate(have):
    os.symlink(os.path.join(FRAMES, f), os.path.join(SEQ, f"s{j:04d}.jpg"))

ff = imageio_ffmpeg.get_ffmpeg_exe()
out = os.path.join(ROOT, "wamo_timelapse_full.mp4")
vf = ("scale=1280:-2:force_original_aspect_ratio=decrease,"
      "pad=1280:720:(ow-iw)/2:(oh-ih)/2:color=black,format=yuv420p")
cmd = [ff, "-y", "-framerate", "15", "-i", os.path.join(SEQ, "s%04d.jpg"),
       "-vf", vf, "-c:v", "libx264", "-crf", "21", "-preset", "medium",
       "-movflags", "+faststart", out]
subprocess.run(cmd, check=True)
sz = os.path.getsize(out) / 1e6
print(f"\nOUT {out}  ({sz:.1f} MB)  {len(have)} frames @15fps = {len(have)/15:.0f}s")
