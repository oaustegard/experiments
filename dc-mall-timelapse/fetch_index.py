#!/usr/bin/env python3
"""Page EarthCam HOF index for the WAMO (Washington Monument) cam back to a cutoff date."""
import re, json, time, sys, urllib.request, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
CUTOFF = int(datetime.datetime(2026, 5, 1, tzinfo=datetime.timezone.utc).timestamp())
BASE = ("https://www.earthcam.com/cams/common/gethofitems.php"
        "?hofsource=com&tm=ecn&camera=wamocam_stream&start={start}&length=50"
        "&ec_favorite=0&cdn=0&date_start=undefined&date_end=undefined"
        "&id=443ba9e812822025dded0dcc3b219c95&callback=onjsonpload")
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

def fetch(start):
    req = urllib.request.Request(BASE.format(start=start), headers={"User-Agent": UA})
    raw = urllib.request.urlopen(req, timeout=30).read().decode("utf-8-sig")
    return json.loads(re.sub(r'^onjsonpload\(|\);?\s*$', '', raw))

def main():
    items, seen = {}, set()
    start = 0
    while True:
        j = fetch(start)
        page = j.get("hofdata", [])
        if not page:
            break
        new = 0
        for it in page:
            if it["id"] in seen:
                continue
            seen.add(it["id"]); new += 1
            items[it["id"]] = {"id": it["id"], "ts": int(it["date_added"]),
                               "url": it["image_source"]}
        oldest = min(int(x["date_added"]) for x in page)
        print(f"start={start:4d} got={len(page)} new={new} oldest={datetime.datetime.fromtimestamp(oldest,datetime.timezone.utc):%Y-%m-%d %H:%M}", flush=True)
        if oldest < CUTOFF or new == 0:
            break
        start += 50
        time.sleep(0.3)
    kept = sorted((v for v in items.values() if v["ts"] >= CUTOFF), key=lambda x: x["ts"])
    json.dump(kept, open(HERE / "index.json", "w"), indent=0)
    if kept:
        a = datetime.datetime.fromtimestamp(kept[0]["ts"], datetime.timezone.utc)
        b = datetime.datetime.fromtimestamp(kept[-1]["ts"], datetime.timezone.utc)
        print(f"\nKEPT {len(kept)} frames  {a:%Y-%m-%d %H:%M} -> {b:%Y-%m-%d %H:%M} UTC")

if __name__ == "__main__":
    main()
