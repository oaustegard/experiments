# RESULTS: EmbeddingGemma 2 query embedder on a Cloudflare Container

Commands to rebuild, check and deploy are in [`README.md`](README.md). Result
files are in `results/`.

## Answer

Can a page that embeds its corpus with EmbeddingGemma 2 in the browser send
casual queries to a hosted EG2 instead of loading the 174 MB model first? Yes.
No provider hosts EG2 per call yet (released 2026-10-06), so this runs the same
`model_q4.onnx` the browser uses on CPU onnxruntime in a Cloudflare Container,
behind a Worker that allows listed origins, rate-limits per IP and caches by
exact text. It is live at `eg2-query-embedder.austegard.workers.dev`. The vector
it returns matches the local server's to six decimals, and the local server
reproduces the model card's scores (0.8542 for the Mars document <!-- @evidence[[results/parity.json#scores.1]] -->
against the card's 0.854).

A cold start costs about 3.1 s inside a 1-vCPU container before the port
opens (3,141 ms), <!-- @evidence[[results/local_standard-2_t1.json#listening_ms]] -->
plus whatever Cloudflare spends starting the container. Most of it is the
tokenizer, not the model: building EG2's 262,144-token tokenizer took
2,140 ms at 1 vCPU <!-- @evidence[[results/local_standard-2_t1.json#runs.2.server.tokenizer_load]] -->
and native Rust `tokenizers` is no faster than the JS library. A warm query takes
110 ms of inference at 1 vCPU locally, <!-- @evidence[[results/local_standard-2_t1.json#warm_roundtrip_median_ms]] -->
about 155–210 ms on Cloudflare, and roughly 500 ms round trip from this
session, so the page should still embed locally whenever it already has the
model cached.

## Findings

1. The tokenizer dominates a cold start. On 4 unrestricted cores the server is
   ready in 2,011 ms, <!-- @evidence[[results/host_4core.json#runs.2.server.ready]] -->
   of which building the tokenizer is 1,554 ms <!-- @evidence[[results/host_4core.json#runs.2.server.tokenizer_load]] -->
   and an inference after warm-up 48 ms. <!-- @evidence[[results/host_4core.json#warm_roundtrip_median_ms]] -->
   The 32 MB `tokenizer.json` (514,906 merges <!-- @evidence[[results/tokenizer_load.json#merges]] -->)
   takes 391 ms to `JSON.parse` <!-- @evidence[[results/tokenizer_load.json#js_parse_ms]] -->
   and 850 ms to construct. <!-- @evidence[[results/tokenizer_load.json#js_construct_ms]] -->
   (`results/host_4core.json`, `results/tokenizer_load.json`)
2. Three ways to shorten it barely helped. Rust `tokenizers` 0.23.2 loads the
   same file in 1,365 ms, <!-- @evidence[[results/tokenizer_load.json#rust_tokenizers_0_23_2_load_ms]] -->
   about what the JS library takes. A V8 startup snapshot of the built tokenizer
   cut the time to listening from 2,356 to 1,934 ms <!-- @evidence[[results/tokenizer_load.json#v8_snapshot.listening_ms_drop_caches]] -->
   for a 97.6 MB blob, because deserializing it costs most of what it saves. Saving
   onnxruntime's optimized graph at build time changed nothing: 2,297 ms to
   listening with it, <!-- @evidence[[results/tokenizer_load.json#optimized_graph.listening_ms_optimized]] -->
   2,243 ms without. <!-- @evidence[[results/tokenizer_load.json#optimized_graph.listening_ms_not_optimized]] -->
   None of the three shipped. (`results/tokenizer_load.json`)
3. Match onnxruntime's threads to the vCPU quota. By default it starts one
   thread per visible core; under a 1-vCPU Docker quota that doubles warm
   latency, 203.1 ms <!-- @evidence[[results/local_standard-2.json#warm_roundtrip_median_ms]] -->
   against 109.8 ms with `THREADS=1`. <!-- @evidence[[results/local_standard-2_t1.json#warm_roundtrip_median_ms]] -->
   At half a vCPU, 493.8 ms <!-- @evidence[[results/local_standard-1.json#warm_roundtrip_median_ms]] -->
   against 207.4 ms. <!-- @evidence[[results/local_standard-1_t1.json#warm_roundtrip_median_ms]] -->
   (`results/local_*.json`)
4. Cold start scales with the CPU quota. Time to listening was 3,141 ms at 1 vCPU, <!-- @evidence[[results/local_standard-2_t1.json#listening_ms]] -->
   5,612 ms at half a vCPU <!-- @evidence[[results/local_standard-1_t1.json#listening_ms]] -->
   and 16,303 ms at a quarter (`basic`, default threads). <!-- @evidence[[results/local_basic.json#listening_ms]] -->
   The deploy uses a custom 1 vCPU / 3 GiB instance, the smallest memory
   Cloudflare allows at 1 vCPU, since memory is billed on the provisioned size.
   (`results/local_*.json`)
5. On Cloudflare, the first query after the very first deploy took 21,795 ms, <!-- @evidence[[results/live_first_deploy.json#first.roundtrip_ms]] -->
   which includes Cloudflare fetching the image and is not a steady-state cold
   start. A cached query answered in 54 ms. <!-- @evidence[[results/live_first_deploy.json#cache.roundtrip_ms]] -->
   (`results/live_first_deploy.json`)
6. The per-IP limit (20 per 60 s) first refused request 21 in the window, <!-- @evidence[[results/live_first_deploy.json#ratelimit.first_429_at_request]] -->
   then let every other request through for the rest of the burst. Cloudflare's
   rate-limit binding counts approximately, so treat it as halving traffic
   beyond the limit, not stopping it. (`results/live_first_deploy.json`)
7. Requests with no `Origin` header, or an unlisted one, get 403. <!-- @evidence[[results/live_first_deploy.json#no_origin_status]] -->
   That blocks scripts too unless they send an allowed origin, which they can;
   the origin list is a courtesy to browsers, and the rate limit, the 512-character
   cap and the cache are what bound cost. (`results/live_first_deploy.json`)
8. The container did not sleep on schedule. After about 14 idle minutes against
   a 10-minute `sleepAfter`, the instance started at 04:03Z was still running, and
   the next query took 980 ms round trip, <!-- @evidence[[results/live_cold_after_idle.json#first.roundtrip_ms]] -->
   a warm answer. So no cold start after an idle sleep has been measured on
   Cloudflare yet, and an instance may bill for longer than `sleepAfter`
   suggests. (`results/live_cold_after_idle.json`, `results/instance_state_0426.json`)

## Method

- **Server** (`server/server.cjs`): Node 22, `onnxruntime-node` 1.30.0 CPU,
  `@huggingface/tokenizers` 0.2.0 (the library the browser page uses),
  `onnx-community/embeddinggemma-2-ONNX` at revision `daa72c51`, `model_q4.onnx`.
  The `query` prompt from `config_sentence_transformers.json` is prepended;
  `sentence_embedding` comes back unit length; a Matryoshka prefix (512, 256,
  128) is renormalized. The port opens only after a warm-up inference.
- **Correctness gate** (`server/test_parity.mjs`): the model card's
  text-search example must reproduce the card's four scores within 6e-4, and a
  512-dim prefix must be unit length. It runs inside `docker build` as the
  runtime user, so an image that embeds wrongly or cannot read its model fails
  to build.
- **Cold-start bench** (`server/bench_cold.mjs`): each rep spawns a fresh
  server (process or `docker run` under `--cpus`/`--memory`), with the page cache
  dropped first, and times time to listening, the first `/embed`, then a warm
  median over ten short queries. Three reps per arm; medians reported. Machine:
  4-core Xeon container, Docker 29.8.
- **Live check** (`worker/live_check.mjs`): first-query latency, warm median,
  cache hit, query vector against the local server, origin rejection, 512-dim
  output, and a 25-request burst against the rate limit.
- **Worker** (`worker/src/index.js`): origin allow-list, then the per-IP limit,
  then input checks, then `caches.default` keyed by model revision, dims and the
  SHA-256 of the exact text, then the container. `POST /warm` starts the
  container without waiting, for a page to call on search-box focus.

## Log

### 2026-10-19 — build, measure, deploy

Asked: host EG2 for query embedding so a client-side page need not load the
model for a casual query, with abuse limits (Oskar). Built the Node server,
the gate and the Worker; deployed after Oskar's go.

What broke on the way:
- `ADD --chmod=644` also applies to the `model/` directory ADD creates, which
  then lacks +x; the server could not read its model as the `node` user. The
  card check had run as root and passed. Fixed with 755 and by running the
  check after `USER node`.
- Wrangler's own `docker build` cannot reach npm through this session's
  proxy, and buildx's attestation manifest made the first `wrangler containers
  push` fail with "blob unknown to registry". Deployed instead from an image
  built locally with the proxy passed in, pushed to Cloudflare's registry as
  `eg2-query-embedder:v1`.
- The first bench hung: Docker's port proxy accepts a connection before the
  server listens, and the health fetch had no timeout. Killing `docker run`
  also left the container running and holding the port. The bench now times out
  each probe, names its container and removes it with `docker rm -f`.

First deploy ran on `standard-1` (half a vCPU) with onnxruntime's default
threads. After findings 3 and 4 it was redeployed on 1 vCPU / 3 GiB with
`THREADS=1` and a 10-minute idle sleep. The first live check after that
redeploy reached a container that was still warm (921 ms first query,
`results/live_after_redeploy.json`); spot checks put inference at 155–210 ms
and the round trip from this session at about 500 ms.

### 2026-10-19 04:26Z — idle cold-start probe

Asked: time a cold start on Cloudflare after the container's 10-minute idle
sleep. The last request had been at 04:11Z (7.8 s round trip, 145 ms inference;
not explained, since the Cloudflare API shows the same instance running since
04:03Z). At 04:26Z, `live_check.mjs --skip-ratelimit` got its first answer in
980 ms with 227.7 ms of inference, warm median 293 ms, cache hit 88 ms
(`results/live_cold_after_idle.json`). The API then showed the instance started
at 04:03:00Z, state `running`, location `mia04` (`results/instance_state_0426.json`):
it had not slept after about 14 idle minutes. Next: read the instance state
again after a longer idle period, without sending a request, before timing a
cold start.

### 2026-10-19 04:57Z — sleep check, no request sent

The Cloudflare containers API at 04:57:40Z still showed the instance started at
04:03:00Z as `running` in `mia04`. Worker analytics show requests at 04:13Z (2)
and 04:42Z (2) that did not come from this session, so the container's idle
stretches were 13 minutes before the 04:26Z probe and 15 minutes before this
check. Both exceed the 10-minute `sleepAfter`, and the instance never restarted
(`results/instance_state_0457.json`). Either `sleepAfter` is not stopping the
container, or the API reports a placed instance whose process has stopped. The
warm 980 ms answer at 04:26Z points to the first. Not yet resolved; finding 8
stands.
