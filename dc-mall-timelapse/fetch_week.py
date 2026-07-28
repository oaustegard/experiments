#!/usr/bin/env python3
"""Paginate the WAMO HOF feed back 7 days, keep daylight frames, download them."""
import os, re, json, time, datetime, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
import zoneinfo

ROOT = "/home/user/claude-workspace/experiments/dc-mall-timelapse"
DEST = os.path.join(ROOT, "frames_week")
os.makedirs(DEST, exist_ok=True)
TZ = zoneinfo.ZoneInfo("America/New_York")
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
# NOTE: no stale `id` param — that pins a cached/stale response and breaks paging.
BASE = ("https://www.earthcam.com/cams/common/gethofitems.php?hofsource=com&tm=ecn"
        "&camera=wamocam_stream&length=50&ec_favorite=0&cdn=0")
DAY_START, DAY_END = 6, 21          # keep ET hour in [6, 21): ~sunrise to ~sunset

now = time.time()
cutoff = now - 7 * 86400
print(f"now {datetime.datetime.fromtimestamp(now,TZ):%Y-%m-%d %H:%M ET}  "
      f"cutoff {datetime.datetime.fromtimestamp(cutoff,TZ):%Y-%m-%d %H:%M ET}")

seen, items = set(), []
start = 0
while start < 1000:
    raw = urllib.request.urlopen(
        urllib.request.Request(f"{BASE}&start={start}", headers={"User-Agent": UA}),
        timeout=30).read().decode("utf-8-sig")
    page = json.loads(raw).get("hofdata", [])
    if not page:
        break
    for it in page:
        if it["id"] in seen:
            continue
        seen.add(it["id"])
        items.append((int(it["date_added"]), it["image_source"]))
    oldest = min(int(x["date_added"]) for x in page)
    print(f"  start={start:4d} oldest {datetime.datetime.fromtimestamp(oldest,TZ):%m-%d %H:%M}")
    if oldest < cutoff:
        break
    start += 50
    time.sleep(0.3)

week = [(t, u) for t, u in items if t >= cutoff]
daylight = [(t, u) for t, u in week
            if DAY_START <= datetime.datetime.fromtimestamp(t, TZ).hour < DAY_END]
daylight.sort()
print(f"collected {len(items)} | past-week {len(week)} | daylight {len(daylight)}")

def dl(iu):
    i, (t, url) = iu
    dst = os.path.join(DEST, f"w{i:04d}.jpg")
    if os.path.exists(dst) and os.path.getsize(dst) > 1000:
        return (t, dst)
    try:
        data = urllib.request.urlopen(
            urllib.request.Request(url, headers={"User-Agent": UA}), timeout=30).read()
        if len(data) < 1000:
            return None
        open(dst, "wb").write(data)
        return (t, dst)
    except Exception as e:
        print("  ERR", url, e)
        return None

index = []
with ThreadPoolExecutor(max_workers=12) as ex:
    futs = [ex.submit(dl, iu) for iu in enumerate(daylight)]
    for f in as_completed(futs):
        r = f.result()
        if r:
            index.append([r[0], r[1]])
index.sort()
json.dump(index, open(os.path.join(ROOT, "week_index.json"), "w"))
a = datetime.datetime.fromtimestamp(index[0][0], TZ)
b = datetime.datetime.fromtimestamp(index[-1][0], TZ)
print(f"downloaded {len(index)} daylight frames  {a:%m-%d %H:%M} -> {b:%m-%d %H:%M} ET")
