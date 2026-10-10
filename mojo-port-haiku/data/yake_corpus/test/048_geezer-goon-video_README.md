# Geezer Goon — animated music video

An animated video for Oskar's song about the Wednesday G2 ("Geezer Goon") group ride out of Kensington, MD. 4:09, 30 fps, two versions.

| | Files | How it's drawn |
|---|---|---|
| **v2** (current) | [`geezer-goon-v2-720p.mp4`](geezer-goon-v2-720p.mp4) (27 MB) · 1080p master (115 MB) in R2, `austegard-media/geezer-goon/geezer-goon-v2-1080p.mp4` | HTML canvas in headless Chromium, one frame at a time, encoded with ffmpeg. Source in [`v2/`](v2/). |
| v1 | [`v1/geezer-goon-v1-720p.mp4`](v1/geezer-goon-v1-720p.mp4) | pycairo side-scroller. Source in [`v1/`](v1/). |

## v2

The road is the real terrain of a Wednesday ride. Strava's streams for the 2026-09-09 G2 (600 samples: GPS, altitude, speed) are trimmed to the ride proper, smoothed, and drive:

- the road profile: the pack tilts up the real grades (horizontal 6 px/m, vertical 13 px/m, elevation exaggerated 2.2x);
- the speedometer, which reads the ride's actual speed at that distance (about 25 mph through the bridge);
- the odometer, climb counter and ride clock (6:15 to 7:32 PM, scaled to the 77 minutes in the lyric);
- the route map (rotated so north is right, to fit a landscape frame; no basemap).

The song is laid onto the ride with an anchor table (`ANCH0` in `v2/web/core.js`, song second to metres along the activity). "Ridge on to Ross" lands on the real climb at 14.6 km, "Mormon Hill" on the largest climb in the trace (+35 m at 26.1 km), the bridge on the long fast stretch from Cedar to Knowles. Between anchors the rate is smoothed, so scroll speed has no kinks. Where the song skips a long stretch (the pre-chorus between Ross and the hill) the odometer fast-forwards and the HUD says so.

Time of day runs from golden hour to night across the song; the lyric timing comes from the subtitle track embedded in the song's `.m4a`.

Shot list: about 60 cuts over 13 scene types (side-view tracking with depth of field, top-down gate funnel, drone view of the pack, route map, bike computer, calendar, pothole chart, three-pane split, and the rest). `v2/web/timeline.js` holds the shot table, annotations and overlays in one place.

### Rebuild

```bash
pip install playwright librosa imageio-ffmpeg     # Chromium comes from PLAYWRIGHT_BROWSERS_PATH; capture.py points at it
cd v2
# inputs not in the repo: route_raw.json (Strava streams: location, altitude, velocity_smooth),
# subs.srt (ffmpeg -i song.m4a -map 0:1 subs.srt), analysis.json (v1/analyze.py on a 16 kHz wav of the song)
python3 prep.py                                   # writes web/data.js (committed, so the renderer runs without the inputs)
python3 capture.py still 28 73 168                # spot-check frames -> stills/
./render_all.sh                                   # 12 chunks, 4 workers, then mux; needs the song as the audio input
```

About 0.37 s per frame per worker, so roughly 12 minutes on 4 cores. Chunks are resumable: finished ones are skipped.

### Data

`web/data.js` holds route shape in metres relative to the route's own centroid, plus altitude and speed. It carries no absolute coordinates. The first 600 m of the Strava activity, the roll from home to the start on Beach Dr, is dropped in `prep.py` (`RIDE_START`), and the raw streams are not committed.

### Snags

- The Strava tool returns streams inline, not as a file. Copy them to disk and check the point count and maximum step (here 600 points, no step over 78 m) before trusting a hand copy.
- Fonts: Anton, Bebas Neue and Permanent Marker (OFL) are in `web/fonts/`.
- Early review caught the night frames too dark, headlight glows blown out, name tags colliding with lyrics, and a single-file gate stream that overlapped bikes. A first cut of the close-up face was replaced by the real rider rig at high zoom.
- Known flaws: on the longest chorus-2 line the lyric's last word sits under the HUD, and the two "get out the way" bubbles in the second gate scene overlap.
- Corrections from Oskar: Karim and Angelo have no beards; the ride starts and finishes on Beach Dr near the base of the temple hill, not at his house.

## v1

Flat vector side-scroller: pycairo frames piped to ffmpeg, four parallel chunks. Fake terrain and speed. `v1/render.py` + `v1/analyze.py`. Lessons kept from it: Python's `hash()` is salted per process, so never use it for animation phase in a chunked render; and a fade that reaches scale 0 makes cairo raise `invalid matrix`.
