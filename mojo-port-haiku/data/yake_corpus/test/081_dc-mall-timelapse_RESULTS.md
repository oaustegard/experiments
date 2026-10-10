# dc-mall-timelapse — EarthCam WAMO (Washington Monument) cam

**Ask:** timelapse of the DC mall webcam "starting from May 1" (2026), from
`earthcam.com/usa/dc/washingtonmonument/?cam=wamo`, using the `gethofitems.php`
"index of files" the user linked.

**Result (round 1):** the `gethofitems.php` endpoint only ever serves the newest
≤50 stills, so May 1 is unreachable through it. Built a ~4-day clip from those
50 (`wamo_timelapse.mp4`, Jun 8 → Jun 12 2026, 8 fps).

**Result (round 2 — final):** user supplied the full HOF gallery file list
(`filenames.txt`, 996 names back to May 1, scraped from the site's gallery UI).
Downloaded them from `https://www.earthcam.com/hof/dc/washingtonmonument/<name>`
and built **`wamo_timelapse_full.mp4`** — 559 frames, **May 4 → Jun 18 2026**,
1280×720, 15 fps (37 s). **437 of the 996 returned 404** (confirmed real, serial
re-test + redirect chase): EarthCam purges HOF stills on a **~45-day retention**
window — the earliest survivor (May 4 11:27) is exactly 45 days before "today"
(Jun 18). All of May 1–3, including the dense ~400-frame May-1 burst in the list,
was already gone; the gallery listing outlived the files. Surviving coverage is
near-daily with one ~2.6-day gap (May 10–11). Cadence is irregular (4–37
frames/day), so a constant-fps timelapse is non-linear in real time.

## Ground truth: what each EarthCam endpoint actually serves

| Source | Reality | May 1? |
|---|---|---|
| `gethofitems.php` (the "index") | Only the **newest ≤50** Hall-of-Fame stills. `start`, `date_end`, `date_start`, `last_item_id` are **all ignored** — every variant returns the identical newest page despite `fullcount:75000`. Tested `start` ∈ {0,21,42,169,1000} × `length` ∈ {21,50,500}; `length` caps at 50. | ❌ |
| Live HLS | Live frame only | ❌ |
| Archive VOD `MP4:events/dc/WAMO.mp4` | Rolling **~60 min** clip (361 segs, 3603 s, ENDLIST) | ❌ |
| Archive VOD `archives/4356/backup.mp4` | Rolling **~60 min** clip (262 segs, 3579 s) | ❌ |
| `api/cdata.php?id=4356` (config) | Generic HTML page, no usable JSON | ❌ |

EarthCam's deep, date-selectable archive (weeks back) is a **premium/paid
feature with no open API**. The HOF feed is a fixed newest-50 window, not a
pageable file index — and it also lags: newest item was Jun 12 while "today"
was Jun 18.

Camera internals (from page config): internal id **4356**, label `wamocam_stream`,
HOF image path pattern `hof/dc/washingtonmonument/<ms-epoch>_<rand>.jpg`.

## Tooling note (reusable)

No system `ffmpeg`, and apt's archive 404s. **`pip install imageio-ffmpeg`**
ships a static **ffmpeg 7.0.2** at `imageio_ffmpeg.get_ffmpeg_exe()` — the
reliable way to get ffmpeg in a CCotw container.

## Files

- `fetch_index.py` — pages the HOF endpoint to a cutoff (writes `index.json`).
- `build.py` — downloads stills → `frames/`, encodes `wamo_timelapse.mp4`.
- `index.json` — the 50 retrievable frame records.
- `wamo_timelapse.mp4` — the output (gitignored; delivered to user).
- `frames/` — downloaded stills (gitignored, regenerable).

## To extend coverage going forward

The only honest way to get a real multi-week timelapse from a free EarthCam
cam is to **capture it yourself over time** (grab the HOF newest-50 or a live
HLS frame on a schedule and accumulate). A CCotw session can't run a 7-week
capture (ephemeral); this would need an external cron/worker.
