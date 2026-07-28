#!/usr/bin/env python3
"""Download HOF frames for the July 5 2026 midnight fireworks window."""
import os, json, time, datetime, zoneinfo, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEST = os.path.join(ROOT, "frames_fireworks")
os.makedirs(DEST, exist_ok=True)
TZ = zoneinfo.ZoneInfo("America/New_York")
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
BASE = ("https://www.earthcam.com/cams/common/gethofitems.php?hofsource=com&tm=ecn"
        "&camera=wamocam_stream&length=50&ec_favorite=0&cdn=0")   # no stale id param
LO = datetime.datetime(2026, 7, 4, 23, 55, tzinfo=TZ).timestamp()  # ~5 min before midnight: catch the opening launch
HI = datetime.datetime(2026, 7, 5, 1, 0, tzinfo=TZ).timestamp()    # 1 AM ET

seen, items = set(), []
for start in range(0, 600, 50):
    raw = urllib.request.urlopen(
        urllib.request.Request(f"{BASE}&start={start}", headers={"User-Agent": UA}),
        timeout=30).read().decode("utf-8-sig")
    page = json.loads(raw).get("hofdata", [])
    if not page:
        break
    for it in page:
        if it["id"] in seen:
            continue
        seen.add(it["id"]); items.append((int(it["date_added"]), it["image_source"]))
    if min(int(x["date_added"]) for x in page) < LO:
        break
    time.sleep(0.2)

win = sorted([(t, u) for t, u in items if LO <= t < HI])
print(f"fireworks-window frames: {len(win)}")

def dl(iu):
    i, (t, url) = iu
    dst = os.path.join(DEST, f"fw{i:04d}.jpg")
    if os.path.exists(dst) and os.path.getsize(dst) > 1000:
        return [t, dst]
    try:
        data = urllib.request.urlopen(
            urllib.request.Request(url, headers={"User-Agent": UA}), timeout=30).read()
        if len(data) < 1000:
            return None
        open(dst, "wb").write(data)
        return [t, dst]
    except Exception as e:
        print("  ERR", url, e); return None

index = []
with ThreadPoolExecutor(max_workers=12) as ex:
    for f in as_completed([ex.submit(dl, iu) for iu in enumerate(win)]):
        r = f.result()
        if r:
            index.append(r)
index.sort()
json.dump(index, open(os.path.join(ROOT, "fireworks_index.json"), "w"))
a = datetime.datetime.fromtimestamp(index[0][0], TZ)
b = datetime.datetime.fromtimestamp(index[-1][0], TZ)
print(f"downloaded {len(index)} frames  {a:%H:%M:%S} -> {b:%H:%M:%S} ET")
