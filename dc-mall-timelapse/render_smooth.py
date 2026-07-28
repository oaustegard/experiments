#!/usr/bin/env python3
"""Render the WAMO timelapse with a real-time-proportional time base, burned-in
DC-local timestamp, and crossfade dissolves. Frames are generated in Python and
piped raw to ffmpeg."""
import os, sys, bisect, subprocess, datetime, zoneinfo, json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageEnhance
import imageio_ffmpeg

def _env(name, default, cast):
    v = os.environ.get(name)
    return cast(v) if v is not None else default

def _date(s):  # YYYY-MM-DD
    return datetime.date(*map(int, s.split("-")))

ROOT = Path(__file__).resolve().parent
FRAMES = os.path.join(ROOT, "frames_full")
OUT = os.path.join(ROOT, _env("TL_OUT", "wamo_timelapse_smooth.mp4", str))
W, H = 1280, 720
OUT_FPS = _env("TL_FPS", 15, int)   # output frame rate (crossfaded stills — 15 is plenty)
SECONDS_PER_DAY = _env("TL_SEC_PER_DAY", 1.5, float)  # real 24h -> N s of video
XFADE = 0.5                # crossfade seconds at the end of each frame's hold
TAIL = 1.5                 # hold last frame this long
MAX_HOLD = _env("TL_MAX_HOLD", 1.0, float)  # cap any single frame's on-screen time (s)
MIN_HOLD = _env("TL_MIN_HOLD", 0.0, float)  # floor: guarantee each frame at least this long (s)
SATURATION = _env("TL_SATURATION", 1.0, float)  # >1 punches up color
START_DATE = _env("TL_START", datetime.date(2026, 5, 12), _date)  # first ET date kept
END_DATE = _env("TL_END", datetime.date(2026, 6, 18), _date)      # last ET date kept (inclusive)
DAY_START_HOUR = _env("TL_HOUR_START", 5, int)  # keep frames with ET hour in [START, END)
DAY_END_HOUR = _env("TL_HOUR_END", 21, int)
TZ = zoneinfo.ZoneInfo("America/New_York")
FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)

# --- build ordered (timestamp, path) list of survivors ---
INDEX = os.environ.get("TL_INDEX")
if INDEX:
    # pre-filtered index: JSON list of [ts_seconds, path], already daylight-filtered
    items = [(t, p) for t, p in json.load(open(os.path.join(ROOT, INDEX)))]
    items.sort()
    N = len(items)
    print(f"loaded {N} frames from {INDEX}", flush=True)
else:
    names = [l.strip() for l in open(os.path.join(ROOT, "filenames.txt")) if l.strip()]
    names.sort(key=lambda n: int(n.split("_")[0]))
    items = []
    drop_date = drop_night = 0
    for i, n in enumerate(names):
        p = os.path.join(FRAMES, f"f{i:04d}.jpg")
        if not (os.path.exists(p) and os.path.getsize(p) > 1000):
            continue
        t = int(n.split("_")[0]) / 1000.0
        dt = datetime.datetime.fromtimestamp(t, TZ)
        if not (START_DATE <= dt.date() <= END_DATE):    # outside the date range
            drop_date += 1
            continue
        if not (DAY_START_HOUR <= dt.hour < DAY_END_HOUR):  # outside ET hour window
            drop_night += 1
            continue
        items.append((t, p))
    N = len(items)
    print(f"kept {N} | dropped {drop_date} outside {START_DATE}..{END_DATE} + "
          f"{drop_night} outside {DAY_START_HOUR}:00-{DAY_END_HOUR}:00 ET", flush=True)
ts = [t for t, _ in items]
SCALE = 86400.0 / SECONDS_PER_DAY          # real seconds per video second
# cumulative video-time starts, capping any single hold at MAX_HOLD
starts = [0.0]
for k in range(1, len(ts)):
    starts.append(starts[-1] + min(max((ts[k] - ts[k - 1]) / SCALE, MIN_HOLD), MAX_HOLD))
total = starts[-1] + TAIL
nframes = int(round(total * OUT_FPS))
print(f"{N} frames | video {total:.1f}s @ {OUT_FPS}fps = {nframes} out-frames "
      f"| {SECONDS_PER_DAY}s/day", flush=True)

# --- lazy processed-frame cache (1280x720 RGB, AR-preserved, padded) ---
_cache = {}
def frame(i):
    im = _cache.get(i)
    if im is None:
        src = Image.open(items[i][1]).convert("RGB")
        # cover-crop to fill 1280x720 so every frame is full-bleed, identical size
        im = ImageOps.fit(src, (W, H), Image.LANCZOS, centering=(0.5, 0.5))
        if SATURATION != 1.0:
            im = ImageEnhance.Color(im).enhance(SATURATION)
        _cache[i] = im
        for k in [k for k in _cache if k < i - 1]:
            del _cache[k]
    return im

def label(t_epoch):
    return datetime.datetime.fromtimestamp(t_epoch, TZ).strftime("%b %-d, %Y   %-I:%M %p ET")

def stamp(img, text):
    img = img.convert("RGBA")
    ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    margin = 18
    bb = d.textbbox((0, 0), text, font=FONT, stroke_width=1)
    tw = bb[2] - bb[0]
    # top-right, no banner; thin dark outline keeps thin white text legible over sky
    d.text((W - margin - tw, margin), text, font=FONT, fill=(255, 255, 255, 235),
           stroke_width=1, stroke_fill=(0, 0, 0, 170))
    return Image.alpha_composite(img, ov).convert("RGB")

# --- container metadata: location, capture time, credit ---
first_dt = datetime.datetime.fromtimestamp(ts[0], TZ)
last_dt = datetime.datetime.fromtimestamp(ts[-1], TZ)
LOCATION = _env("TL_LOCATION", "National Mall, Washington, DC", str)
GPS = _env("TL_GPS", "+38.8895-077.0353/", str)   # ISO 6709 — Washington Monument
CREDIT = _env("TL_CREDIT", "EarthCam", str)
TITLE = _env("TL_TITLE", f"Washington Monument Cam — {first_dt:%b %-d, %Y}", str)
meta = [
    "-metadata", f"title={TITLE}",
    "-metadata", f"artist={CREDIT}",
    "-metadata", f"copyright=© EarthCam, Inc. — earthcam.com",
    "-metadata", f"location={LOCATION}",
    "-metadata", f"location-eng={LOCATION}",
    "-metadata", f"com.apple.quicktime.location.ISO6709={GPS}",
    "-metadata", f"date={first_dt:%Y-%m-%d}",
    "-metadata", f"creation_time={first_dt.astimezone(datetime.timezone.utc):%Y-%m-%dT%H:%M:%SZ}",
    "-metadata", (f"comment=Time-lapse of {LOCATION} from the EarthCam WAMO cam "
                  f"(earthcam.com/usa/dc/washingtonmonument). Captured "
                  f"{first_dt:%Y-%m-%d %H:%M}–{last_dt:%H:%M %Z}. Credit: {CREDIT}."),
]
ff = imageio_ffmpeg.get_ffmpeg_exe()
cmd = [ff, "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
       "-r", str(OUT_FPS), "-i", "-", "-an", "-vf", "format=yuv420p",
       "-c:v", "libx264", "-crf", "20", "-preset", "medium",
       *meta, "-movflags", "+faststart+use_metadata_tags", OUT]
proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL)

for k in range(nframes):
    t = k / OUT_FPS
    i = bisect.bisect_right(starts, t) - 1
    i = max(0, min(i, N - 1))
    if i >= N - 1:
        img, lab = frame(N - 1), ts[-1]
    else:
        d_i = starts[i + 1] - starts[i]
        xf = min(XFADE, d_i)
        fade_start = starts[i + 1] - xf
        if t < fade_start or xf <= 0:
            img, lab = frame(i), ts[i]
        else:
            a = (t - fade_start) / xf
            img = Image.blend(frame(i), frame(i + 1), a)
            lab = ts[i] if a < 0.5 else ts[i + 1]
    out = stamp(img, label(lab))
    proc.stdin.write(np.asarray(out, dtype=np.uint8).tobytes())
    if k % 300 == 0:
        print(f"  {k}/{nframes}", flush=True)

proc.stdin.close()
proc.wait()
sz = os.path.getsize(OUT) / 1e6
print(f"\nOUT {OUT}  ({sz:.1f} MB, {total:.1f}s)")
