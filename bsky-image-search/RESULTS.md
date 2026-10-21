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

## Open

- Where the demo is hosted (GitHub Pages would let the hosted EG2 query embedder serve queries
  with no model download), and whether to publish an index of other people's public images.
- Typed-query quality: the alt-text eval is caption-like; a small hand-written query set would
  measure what people actually type.
- The query side could use eg2-layer-student's 29 MB cut instead of full EG2 text (174 MB) for
  the no-embedder fallback; its cross-modal quality is untested.
