# emeans-explainer — a narrated video of where e-means beats Lloyd's k-means

A 2½-minute explainer of [`emeans-pooled-residual`](../emeans-pooled-residual/RESULTS.md), made with
[Kinescript](https://github.com/drknowhow/kinescript) 2.5.0 in the Muninn brand (`brand/kinescript/` in
`oaustegard/muninn.austegard.com`) and narrated by Kokoro in Muninn's voice. After a title card it takes eight steps:

1. Lloyd's k-means: assign every point to its nearest centre, move each centre to its points' mean, 25 rounds
2. e-means: small batches, moving-average updates, empty clusters split off busy ones; 2 passes ≈ 25 Lloyd rounds
3. the timing grid (61 jobs): time for 25 Lloyd rounds ÷ time for 2 NumPy e-means passes, median by dimension with
   ≥ 25 points per cluster, 0.96× at d = 3 to 4.91× at d = 128, with e-means ending within 1% of Lloyd's error or
   below it at 32+ dimensions; the PyTorch version up to 52×, with its own search
4. where it falls short: below 25 points per cluster on synthetic data (up to 1.38× Lloyd's error), and d = 3 with
   1,024+ centres (0.1–0.5× Lloyd's speed; faiss batches cost up to 52× more per distance)
5. ColBERT: offsets from one average vs from the nearest of 16,384 centroids; SciFact 1-bit 0.730 → 0.751 against
   0.746 for 2 bits on the average
6. the centroid table: 2.3 KB per document either way on 5k documents, 1.5 KB without the table
7. the trainer: Lloyd-2 0.754, Lloyd-25 0.751, e-means-2 0.751 nDCG@10; fit 25 s, 321 s, 26 s
8. the rule: ≥ 32 dimensions and ≥ 25 points per cluster; keep Lloyd for small clusters and 3-d faiss jobs; check
   whether the metric notices

Every number on screen or spoken comes from `../emeans-pooled-residual/summary.json` (SciFact, pool factor 4,
K = 16,384 for the ColBERT scenes). The point clouds and the centroid diagram are illustrations, not data.

## Files

- `explainer.mp4`: 1920×1080, 30 fps, 2:31, opens on the title as its poster frame; −16.4 LUFS
- `poster.jpg`, `grid.jpg`: the poster frame and the grid chart frame (used on the blog post)
- `build_scene.py`: writes `scene.json` from one timeline (nine voice lines with `[[mark]]` word cues, layers cued to
  those words, karaoke captions per line)
- `scene.json`: the Kinescript scene; it expects the brand folder at `brand/`

## Rebuild

```
python3 build_scene.py
cp -r <muninn.austegard.com>/brand/kinescript brand
kine voice scene.json      # Kokoro, the brand's speaker bm_fable,bm_fable,bm_lewis, rate 0.9 → voice/ (not committed)
kine check scene.json      # 0 errors, 0 warnings
kine render scene.json -o explainer.mp4 -q high -j 4
```

Kokoro 0.9.4 needs Python ≤ 3.12 and the `kinescript-kokoro` plugin from the Kinescript repo; it ran from a 3.12 venv
through `KINE_VOICE_CMD` with the weights fetched beforehand, and `kine` ran inside claude-workspace's
`scripts/jail.sh` (no network). Voicing took 65 s, the render 112 s on 4 CPUs (40 fps).

## Notes

- **Pace.** Rate 0.9 measured 169 words per minute overall (142–194 per line).
- **Captions.** Kinescript's default caption plate is a dark box; on the brand's paper background the red active word
  was hard to read on it. The scene sets the plate to the brand's surface colour with the text colour for the rest.
- **The grid chart was redrawn after publishing.** Its first version labelled the bars as how much sooner e-means
  reaches Lloyd's error while a note said Lloyd never reached e-means' error: two directions on one chart (Oskar:
  "you've left the reader wondering what the heck the chart is saying"). It now shows one fixed-budget ratio, and
  line 4 was re-voiced to match.
- **Corrections before the final render.** The first renders said "sixty-six" jobs (61 ran), presented the
  grid's speedups without noting that most are lower bounds, and stated the under-25-points limit without saying it
  held only on synthetic data. Lines 4 and 5 were re-voiced and the scene re-rendered.
