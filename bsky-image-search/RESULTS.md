# RESULTS: Bluesky image search with EmbeddingGemma 2 vision and 2-bit codes

Run 2026-10-20/21 on Oskar's MacBook Air (M1, 8 GB), coordinated through `../embed-coord`
(slot `wasm`). Oskar's brief: semantic search over the most popular several thousand Bluesky
images, embedding the images with EG2's vision encoder and loading the whole index into the page.

## Answer

It works. 16,878 images from 12 popular public feeds are searchable from a static page:

- the index is 3.24 MB of remex 2-bit codes (192 B per image), 0.14 MB of scale/bias and
  1.97 MB of compact rows (~4.2 MB gzipped);
- the GPU scans the whole index in about 3 ms;
- queries go through EG2's own text model and query prompt.

On 5,248 alt-text queries over the full index, the image the alt text describes is the top
hit 56.0% of the time and in the top 10 80.3% of the time (2-bit codes; fp32 vectors: 56.6% /
81.3%). Alt text reads like a caption, so this is an upper bound for typed queries.

## Numbers

| What | Result | Source |
|---|---|---|
| Harvest | 16,881 unique images, 6,677 with alt text; 544 adult/graphic-labelled posts dropped | `run/harvest.py`, `out.harvest.log` |
| Browser EG2 gate | cats image vs the card's three queries: 0.740 / 0.462 / 0.508 at budget 280 (card: .740 / .462 / .508) | `web/index_images.html` |
| Throughput by vision budget, batch 8 | 280 tokens 0.34 img/s; 140 0.81; 70 1.63 | `web/profile.html` |
| Budget pilot, 200 images | hit@10 0.990 at both 70 and 280; hit@1 0.895 vs 0.905 | `results/pilot.json` |
| Full run at budget 70 | 16,878 embedded, 3 unavailable, 3.08 h, 1.52 img/s | `web/index_images.html` |
| Full eval, fp32 | hit@1 0.566, hit@10 0.813, MRR@10 0.648 | `results/eval.json` |
| Full eval, 2-bit | hit@1 0.560, hit@10 0.803, MRR@10 0.640 | `results/eval.json` |

## Method notes

- **Popularity** means the top of 12 popular public feeds (What's Hot, Art: What's Hot,
  Artists: Trending, photography, landscape, street, film, nature, cats, science, moss), ranked
  by likes. The median image has only 10 likes; the top has 31,664. What's Hot stopped at 1,900
  posts when its cursor exceeded the URL length limit (HTTP 414).
- **Images** go through a local proxy (`run/serve.py`), because cdn.bsky.app sends no CORS
  header. The demo displays them straight from the CDN; `<img>` tags need no CORS.
- **Vision token budget 70**, not the default 280: 4.8x faster, with the same hit@10 in the
  pilot. 70-token vectors sit at median cosine 0.958 to the 280-token ones.
- **Post text stays out of git** (out/ is ignored). The index rows hold only author did/handle,
  post rkey, image cid and like count.

## Query side: full EG2 text vs the 4-layer cut

`run/eval_query_encoder.py`, results/eval_cuts.json. The same 5,248 alt-text queries and the
same 2-bit image index, paired per query against full EG2 q4 text. The cuts come from
eg2-layer-student's release eg2-layer-onnx-20261021:

| query encoder | download | hit@1 | hit@10 | hit@10 minus EG2 [95% CI] | median query cos to EG2 |
|---|---|---|---|---|---|
| full EG2 text q4 | 175 MB | 0.560 | 0.803 | | 1 |
| 4-layer cut, q4 table | 29.0 MB | 0.307 | 0.563 | -0.240 [-0.252, -0.228] | 0.871 |
| 4-layer cut, 3-bit table | 25.4 MB | 0.305 | 0.559 | -0.244 [-0.256, -0.231] | 0.870 |
| 2-layer cut, 3-bit table | 18.2 MB | 0.270 | 0.513 | -0.289 [-0.303, -0.276] | 0.851 |

The cut was distilled only on Bluesky-style query text, where it nearly matches EG2 (0.907 vs
0.919 hit@10 on its own benchmark). Caption-like alt text is outside that, and the gap is 24
points. Retraining it on captions was planned and then parked (Oskar, 2026-10-21).

**What the site ships.** austegard.com/bsky/image-search/ runs the 4-layer cut with the 3-bit
table as its default query model, on a WGSL engine written by the cloud side. Full EG2 in the
page is a link away ("better results"). Measured on the M1 (Chrome 152, warm cache): index ready
in 1.0 s, first query 38 ms, then 12-13 ms per query, GPU scan of 16,878 images 2-4 ms.

## Open

- Typed-query quality: the alt-text eval is caption-like; a small hand-written query set would
  measure what people actually type.
- Closing the cut's 24-point gap on captions, if the retrain is picked up again.
