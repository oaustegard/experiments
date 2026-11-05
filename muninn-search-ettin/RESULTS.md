# muninn-search-ettin: site search on a 10 MB embedder

The search box on muninn.austegard.com used Gemini's embedding API. A Cloudflare Worker sent each
query to `gemini-embedding-001`, and the site's index had been built with it. On 2026-11-05 the
prepaid Gemini credits ran out, so every search returned HTTP 402. This experiment replaces Gemini
with Ettin-17m, the 10 MB e17c build from `ettin-native/`. Pages are embedded on a CPU when the index
is built. Queries are embedded in the visitor's browser on the GPU, once the page has downloaded the
model and index. Until then, and in browsers that can't run it, the Worker embeds queries in plain
JavaScript. Nothing calls an embedding API. Both paths went live on 2026-11-05.

## Answer

Can a 10 MB model on the Worker's CPU replace Gemini for the site's search? It works. How it compares
with Gemini is unknown, because Gemini is the API that ran out.

Sonnet wrote two sets of 150 queries from sampled passages. One set uses free wording. The other may
not reuse any word of five or more letters from its passage. The score is whether the source page lands
in the top five of 180 pages (chance is 2.8% <!-- @evidence[[results/index.json#chance_hit@5]] -->).

| query set | deployed search, top five | BM25 keyword search alone |
|---|---|---|
| free wording | 99.3% <!-- @evidence[[results/worker.json#/queries.json:a0.7/post_hit@5]] --> | about the same |
| reworded | 62.7% <!-- @evidence[[results/worker.json#/queries_hard.json:a0.7/post_hit@5]] --> | 38.7% <!-- @evidence[[results/worker.json#/queries_hard.json:bm25 (a0.0)/post_hit@5]] --> |

A three-word query takes about 53 ms of CPU, measured in Node <!-- @evidence[[results/timing.json#embed_ms_median.12]] -->.

Three things would change how firmly this stands:
- logs of real queries;
- a held-out query set (the fusion weight was picked on these same queries);
- a Gemini run on the same queries once credits return.

## Findings

1. The deployed search (weighted fusion, alpha 0.7) puts the source page in the top five for 99.3% of
   free-wording queries, with MRR@10 0.945 <!-- @evidence[[results/worker.json#/queries.json:a0.7/post_mrr@10]] -->.
   For reworded queries it is 62.7%, with MRR@10 0.455 <!-- @evidence[[results/worker.json#/queries_hard.json:a0.7/post_mrr@10]] -->
   (Round 4, `results/worker.json`).
2. No setting wins both sets; alpha 0.7 is the one near the top of both.
   - BM25 alone edges it on free wording, with MRR@10 0.959 <!-- @evidence[[results/worker.json#/queries.json:bm25 (a0.0)/post_mrr@10]] -->,
     and falls to 38.7% top five on reworded queries <!-- @evidence[[results/worker.json#/queries_hard.json:bm25 (a0.0)/post_hit@5]] -->.
   - Dense alone ties it on reworded queries, with MRR@10 0.457 <!-- @evidence[[results/worker.json#/queries_hard.json:dense (a1.0)/post_mrr@10]] -->,
     and trails on free wording at 0.898 <!-- @evidence[[results/worker.json#/queries.json:dense (a1.0)/post_mrr@10]] -->.
   - Alpha 0.85 is as good on reworded queries, with a top five of 63.3% <!-- @evidence[[results/worker.json#/queries_hard.json:a0.85/post_hit@5]] -->,
     and weaker on free wording at MRR@10 0.926 <!-- @evidence[[results/worker.json#/queries.json:a0.85/post_mrr@10]] -->.

   The weight was chosen on the same 300 queries reported here; there is no held-out set (Round 4).
3. In the Worker's code, the previous default, RRF, scores below alpha 0.7 on both sets.
   - On reworded queries its top five is 54.7% <!-- @evidence[[results/worker.json#/queries_hard.json:rrf/post_hit@5]] -->,
     against 62.7%.
   - On free wording its MRR@10 is 0.898 <!-- @evidence[[results/worker.json#/queries.json:rrf/post_mrr@10]] -->,
     against 0.945.

   Part of the free-wording gap comes from the Worker's reader rather than from RRF. Python's reader
   with RRF scores 0.941 on free wording <!-- @evidence[[results/fusion.json#/queries.json:rrf/post_mrr@10]] -->,
   level with alpha 0.7 (Round 2, Round 4).
4. With matched settings, the Worker's JavaScript gives the same results as remax_kb's Python reader.
   The settings: the first 40 queries of each set, alpha 0.7, a 40-candidate pool and no `min_sim` floor.
   - Token ids match on 80 of 80 <!-- @evidence[[results/parity.json#token_ids_equal]] --> queries.
   - The fused top 40 matches on 80 of 80 <!-- @evidence[[results/parity.json#fused_top40_equal]] -->.

   At their own defaults the two readers differ by design: different pool depths, and the Worker
   returns one hit per page (Round 3, `results/parity.json`).
5. The live Worker gave the same top five pages as the same code run locally on the published index,
   for 6 of 6 <!-- @evidence[[results/live.json#same_as_local]] --> queries (Round 5, `results/live.json`).
6. Embedding cost, measured in Node 22 on this container rather than on Cloudflare (`run/bench.mjs`):
   - 53 ms at 12 tokens;
   - 82 ms at 20 tokens <!-- @evidence[[results/timing.json#embed_ms_median.20]] -->;
   - 676 ms at the 128-token cap <!-- @evidence[[results/timing.json#embed_ms_median.128]] -->.

   Unpacking the model in a fresh isolate takes 68 ms <!-- @evidence[[results/timing.json#unpack_ms]] -->.
   The query prompt alone is 11 tokens, so a three-word query is 14 tokens.
7. The browser path returns the same hit list as the Worker on 6 of 6 <!-- @evidence[[results/browser.json#same_as_worker]] -->
   queries, with every chunk passing its sha256 check. This was run in headless Chromium on an emulated
   GPU (Round 6, `results/browser.json`).
   - The first search costs a download of 17.2 MB <!-- @evidence[[results/browser.json#download_mb]] -->,
     before compression.
   - After the embedding, ranking takes 9.8 ms <!-- @evidence[[results/rank_timing.json#median_ms.search]] -->
     in Node.

   Ettin-native measured the embedding at 2.4 ms on a Mac GPU, so a local search should take roughly
   15 ms. That figure is not measured end to end.
8. All 180 pages <!-- @evidence[[results/index.json#pages]] --> (2,880 chunks) re-embed in
   41.5 s on four CPUs <!-- @evidence[[results/index.json#full_build_seconds]] --> with no API call
   (Round 5, `results/index.json`).

## Method

**Search, in brief.**
- The index cuts each page into chunks of about 500 characters.
- A query is scored two ways. BM25 is keyword matching. Dense is closeness of Ettin vectors, which
  are compared as 3,072-bit SimHash codes.
- Fusion merges the two rankings over a pool of top candidates from each. RRF (reciprocal-rank fusion)
  adds 1/(60 + rank) from each list. Weighted fusion rescales both scores to 0–1 over the pool and adds
  them as alpha × dense + (1 − alpha) × BM25, so alpha 1.0 is dense alone and 0.0 is BM25 alone.
- The Worker then keeps one hit per page, boosted by how many of that page's chunks matched.

**Corpus and index.** muninn.austegard.com as of 2026-11-05: the blog, perch, scratch and references
pages. `scripts/build_muninn_kb.py` builds a remax_kb v2 index (centered SimHash with dim 768 and k 4, so 3,072 bits per chunk, over 768-dimensional vectors).

**Embedder.** The e17c-t4-w8L63 package (release `ettin-native-20261021`): an int4 token table, int8
matmuls, mean pooling, a 256→768 dense layer and L2 normalisation. The query prompt is "Represent this
sentence for searching relevant passages: " and the document prompt is empty. Pages go through
onnxruntime with graph optimisations off (`scripts/ettin_embedder.py`). Queries go through a
JavaScript port of ettin-native's NumPy reference (`worker/_lib/ettin-cpu.js`). Both agree with that
reference at cosine 1.0000000.

**Queries.** `run/make_queries.py` sampled 150 chunks (seed 0, at most two per post, at least 200
characters). For each one, headless Sonnet wrote the query that a reader who remembered the content
would type. Two sets:
- **Free wording** (`queries.json`): the query may reuse the passage's terms, and usually does.
- **Reworded** (`queries_hard.json`): the query may not reuse any word of five or more letters from
  the passage. The mean overlap that remained was 0.9%. This is a stress test of the dense leg, not a
  picture of typical use.

The 300 calls ran through `claude -p` with tools off. Their cost was not recorded; at the $0.04 per call
measured in `bsky-account-decider`, it comes to about $12.

**Metrics.** All scores are at page level: a hit is any chunk from the query's source page.
- hit@k is the share of queries whose page is in the top k.
- MRR@10 is the mean of 1/rank, counted as 0 below rank 10.

**Code paths scored.**
- `run/eval.py` and `run/fusion.py` score remax_kb's Python reader. Its pool is `max(k*8, 64)` and it
  can return several chunks per page.
- `run/eval_worker.mjs` scores the Worker's own modules. Their pool is `max(k*8, 40)`, with one hit per
  page. This is what users get.
- `run/parity.py` (with `run/fused_js.mjs`) compares the two readers at matched settings.
- `run/worker_local.mjs` runs the whole Worker against a file-backed stand-in for its KV store.
- `run/live_check.py` compares that local run with the deployed Worker.
- `run/bench.mjs` times the encoder, and `run/rank_bench.mjs` times the ranking after it.
- `run/browser_check.cjs` runs the site's in-browser search against the live Worker in headless Chromium
  (SwiftShader WebGPU), then checks that the search page swaps from server to browser.

**Running cost.** Workers Paid ($5 a month) includes 30 million CPU-milliseconds a month. At 50–90 ms
per query, that covers several hundred thousand searches. `wrangler.toml` caps a request at 2,000 ms
of CPU.

**Running the scripts.** The Python scripts find `remax`, `remax_kb` and `muninn.austegard.com` under
`EXPERIMENTS_SPOKES_ROOT`. The Node scripts take the site checkout as an argument and need
`npm install` in its `worker/`.

## Log

### Round 1: does Ettin search work at all (2026-11-05)

Gemini returned 402 and live search was down. The question was whether to wait for credits or to
switch to the 10 MB Ettin, which `bsky-embedders` had already found best on size against quality.
An eval index was built with Ettin on the CPU (2,880 chunks, 39 s), and `run/eval.py` scored the
Python reader.

| queries | arm | hit@1 | hit@5 | MRR@10 |
|---|---|---|---|---|
| free wording | bm25 | 0.927 | 1.000 | 0.956 |
| free wording | dense | 0.833 | 0.980 | 0.898 |
| free wording | hybrid (RRF) | 0.900 | 0.993 | 0.941 |
| reworded | bm25 | 0.140 | 0.353 | 0.223 |
| reworded | dense | 0.307 | 0.580 | 0.428 |
| reworded | hybrid (RRF) | 0.207 | 0.480 | 0.326 |

The prediction was that hybrid would be at least as good as its better leg. That held for free
wording and failed for reworded queries, where RRF scored below dense alone. RRF gives BM25's top
ranks the same credit as dense's, even when BM25 matched only on incidental words.

### Round 2: fusion sweep, Python reader

`run/fusion.py` swept RRF (c 60 and c 10) and weighted alpha 0.3, 0.5, 0.7 and 0.85. It ran the Python
reader at k 40, which gives a pool of 320.

| setting | free hit@5 | free MRR@10 | reworded hit@5 | reworded MRR@10 |
|---|---|---|---|---|
| rrf c60 | 0.993 | 0.941 | 0.480 | 0.326 |
| rrf c10 | 0.993 | 0.941 | 0.560 | 0.369 |
| a0.3 | 0.993 | 0.958 | 0.433 | 0.290 |
| a0.5 | 0.993 | 0.954 | 0.567 | 0.370 |
| a0.7 | 0.993 | 0.946 | 0.593 | 0.417 |
| a0.85 | 0.987 | 0.926 | 0.593 | 0.426 |

Alpha 0.7 was the best compromise. Against alpha 0.3 it gave up 0.012 MRR on free wording and gained
0.13 on reworded queries.

### Round 3: parity between the Worker and Python

The Worker ran in Node against a file-backed stand-in for its KV store. Its top-five lists differed
from Python's, and only 72 of 80 queries had the same top hit. There were two causes:
- `kb-reader.js` collapses chunk hits to pages and boosts pages with several matching chunks.
- Its candidate pool is `max(k*8, 40)`, where Python's is `max(k*8, 64)`. Weighted fusion rescales
  over the pool, so pool depth moves the ranking.

With both readers set to a 40-candidate pool and no `min_sim` floor, the fused top-40 lists matched on
80 of 80 queries, and so did the token ids. Round 2 therefore measured the Python reader, not the site.

### Round 4: quality through the Worker's path

`run/eval_worker.mjs` scored the Worker's modules directly, at k 10.

| setting | free hit@1 | free hit@5 | free MRR@10 | reworded hit@1 | reworded hit@5 | reworded MRR@10 |
|---|---|---|---|---|---|---|
| rrf | 0.820 | 0.993 | 0.898 | 0.267 | 0.547 | 0.385 |
| a0.5 | 0.933 | 0.993 | 0.960 | 0.287 | 0.587 | 0.410 |
| a0.7 | 0.907 | 0.993 | 0.945 | 0.333 | 0.627 | 0.455 |
| a0.85 | 0.873 | 0.993 | 0.926 | 0.340 | 0.633 | 0.460 |
| dense (a1.0) | 0.827 | 0.987 | 0.898 | 0.340 | 0.607 | 0.457 |
| bm25 (a0.0) | 0.933 | 0.993 | 0.959 | 0.153 | 0.387 | 0.256 |

Against round 2, the Worker's path raised most arms on reworded queries. It also lowered RRF on free
wording, from 0.941 to 0.898 MRR@10. The Worker differs in two ways, pool depth and the per-page boost,
and this round does not separate their effects. Alpha 0.7 became the Worker's default; a request can
still pass `alpha: null` for RRF.

### Round 5: deploy and live check

- `scripts/publish_index_to_kv.py --rebuild` uploaded the model package (11.6 MB `model.onnx` plus
  tokenizer and manifest) and a fresh index of 2,845 KB, as index-version `ff97df22bf53`.
- Wrangler 4.20.0 deployed the Worker but ignored the `[[ratelimits]]` block, so the binding was missing.
  A redeploy with wrangler 4.158.0 bound it.
- The rate limit could not be tested from this container. Its egress IP changes on nearly every
  request (five addresses in six calls, two locations), so 66 requests never tripped a per-IP limit.
- The first live query reached an isolate still running the Gemini build and got its 402. Later
  queries reached the new build.
- `run/live_check.py` compared six queries against the local run: 6 of 6 matched. Python's default
  `urllib` user agent gets Cloudflare error 1010, so the script sends its own.
- A fresh full build took 41.5 s on four CPUs.

For the two queries about the new post, the post ranked first. Two unrelated queries both ranked the
long perch digest `fly-2026-05-09-the-political-economy-of-browser-engines` second, which may be the
per-page boost favouring pages with many chunks. Not investigated.

### Round 6: search in the browser

Oskar asked what had happened to the site's no-search-server stance. The Worker had only ever existed
because Gemini needed a secret key, and with Ettin that reason was gone.

What was built:
- Opening search (a click, `/`, cmd+k, the search page, or focusing the field) starts
  `components/local-search.js`.
- It fetches the index and the model from new Worker routes under `/kb/`. The paths carry the version,
  so the browser caches them for a year.
- It embeds the query with the WGSL engine from `ettin-native`, vendored into `components/ettin/`, and
  ranks with the Worker's own `kb-reader.js` and default alpha.
- Until it is ready, and without WebGPU, on Save-Data or on 2G/3G, the Worker answers, now by GET as
  well as POST.

What the checks showed:
- In headless Chromium the local hit lists matched the Worker's on 6 of 6 queries, and every chunk
  verified.
- The download is 17.2 MB raw. On the wire it is 15.8 MB, because the chunk shard compresses from
  2.7 MB to 0.69 MB once served as `text/plain`.
- On the live site, a `?q=` link to a post was answered by the server, and a later search by the browser.
- SwiftShader timings (0.6 to 0.9 s per search) are an emulated GPU and say nothing about real hardware.

The index files still come from the Worker, because they live in KV. Serving them as static files
(R2 with a CORS rule, or the site itself) would remove the Worker from the browser path entirely.
