# gpu-bridge

A page that lets Claude Code on the web sessions run WebGPU work on Oskar's
laptop GPU. The container has 4 vCPUs and no GPU; this tab is the GPU.

Sessions push jobs to the private repo `oaustegard/gpu-bridge-comms`
(`jobs/<id>/job.json`, `job.js`, raw `.bin` inputs). The page, open at
https://austegard.com/experiments/gpu-bridge/web/, polls the branch head with a
conditional request (a 304 costs no rate limit), runs each new job in a
long-lived WebGPU worker, commits `results/<id>/result.json` plus one `.bin`
per typed-array output through the Git Data API, and comments on the repo's
open PR from branch `wake`. A session subscribed to that PR is woken by the
comment, so it can end its turn instead of polling.

The session side is `scripts/gpu_bridge.py` in `oaustegard/claude-workspace`;
the protocol and the job's `ctx` API are in that repo's `docs/gpu-bridge.md`.

The page needs a fine-grained token with access to `gpu-bridge-comms` only
(Contents and Pull requests: read and write). It is kept in this origin's
localStorage, or sessionStorage if "Remember on this device" is off, and is
sent only to api.github.com.

It replaces the artifact version (claude.ai artifact `3QoLs5HN`, same worker
runtime, results through the artifact database). That one needs no token, but a
session cannot wait on it from the shell and nothing wakes the session without
a tap.

## Files

- `web/index.html`: the page; `?api=` points it at another API base (the mock).
- `web/runtime.js`: the worker runtime (`ctx.gpu`, `ctx.compute`, `ctx.cache`, `ctx.log`, `ctx.progress`, `ctx.fetchBytes`). A running job shows a bar on the page: the job's own `ctx.progress(f, label)` when it reports one, else elapsed time against the timeout.
- `run/e2e.js`: end-to-end check. It runs `gpu_bridge.py` submit/status/fetch
  against `run/mockgh.js`, a GitHub REST mock over a local bare repo, with the
  page in headless Chromium on SwiftShader. 2026-10-25: a 1,024-float square
  job round-tripped exactly, a throwing job came back as an error result, and
  both posted wake comments.

SwiftShader runs here are correctness checks only, not timing (METHODS.md).

## Runs on real hardware

| Job | Device | Result |
|---|---|---|
| `hello-1` (2026-10-25) | iPhone, Chrome 155 / iOS 27.0.1 (WebKit WebGPU; adapter "apple") | 16M-float saxpy correct, median 5.0 ms ≈ 40 GB/s (the worker clock is quantized to 1 ms here) |
| `knn-1` (2026-10-25) | same | `jobs/knn/`: exact top-10 by dot product, 256 queries × 32,768 docs × 256 dims (4.29 GFLOP), one fused WGSL kernel. Same order as numpy for all 2,560 results, max score error 1.2e-7. GPU timestamp median 18.1 ms (≈ 237 GFLOP/s effective) vs numpy 62–76 ms on the container's 4 vCPUs. 32 MB input fetched by the phone from the comms repo |
| `scan-1` | same | `jobs/scan/`: exact exclusive prefix sum of 16,777,216 u32 (Blelloch, 512 per workgroup, 3 levels). 0 mismatches; checksum, samples and total agree with numpy. GPU 6.73 ms vs numpy `cumsum` 67 ms on 4 vCPUs |
| `sort-1` | same | `jobs/sort/`: LSD radix sort of 4,194,304 u32 keys, 8 passes × 4 bits (histogram, scan, stable scatter). Exact. GPU 36.9 ms vs numpy `sort` 23.7 ms on 4 vCPUs and JS `TypedArray.sort` 81 ms on the phone: the naive 8-pass design loses to numpy |
| `knn-f16` | same | `knn-1` with docs as float16 (`shader-f16`; first run of that path anywhere, SwiftShader lacks it). Same order as the float16 numpy reference, recall@10 1.0 vs float32. GPU 13.95 ms vs 18.1 ms in float32 (−23%), 16 MB sent instead of 32 |
| `knn-cached` | same | 1,024 new queries against the float16 docs `knn-f16` left in `ctx.cache` (1 MB sent). Same order as numpy's float16 reference, recall@10 0.9997 vs float32. GPU 54.9 ms vs numpy 205 ms |
| `knngraph-65k` (2026-10-26) | same | `jobs/knngraph/`: exact 10-NN graph of 65,536 points × 256 dims (2.2 TFLOP), points generated on the phone, 64 batches of 1,024 queries. Every row has the same neighbour set as the CPU reference; 6 of 655,360 positions swap order on near-ties. GPU 18.0 s (batches 256–306 ms, no throttling trend) = 122 GFLOP/s, half of `knn-1`. The CPU reference (numpy matmul + torch.topk, 4 vCPUs) took 17.9 s: a tie. With 4 queries per workgroup the kernel does 2 FLOP per byte of docs streamed, so at the phone's ~56 GB/s DRAM read ceiling it tops out near 112 GFLOP/s; `knn-1`'s 32 MB of docs likely stayed in the system cache and its 64 MB here did not. More queries per workgroup is the lever |

`knn-1` also exposed two open tabs running the same job: one failed its 32 MB
download ("Load failed") and committed an error, the other overwrote it with
the good result. Page version 3 claims a job with a commit before running it
(a second tab sees the claim and backs off) and retries failed downloads;
`run/e2e.js` now runs two tabs on one repo and checks each job runs once.

The four jobs above were queued at 23:20Z while the phone's tab was in the
background, and started when it was reopened at 23:24Z. Once running, each took
11–41 s end to end over in-flight Wi-Fi, against 0.2–0.6 s of job time: about a
dozen sequential GitHub API calls per job (claim, input download, result
commit, comment), plus 16 MB for `knn-f16`. The bridge pays off when the GPU
work is large next to the bytes moved, as in `knn-cached`.

The same run showed one tab overlapping its own jobs. Its two "page opened"
commits are 4 ms apart, so `connect()` ran twice in one tab and started two poll
loops. Page version 4 keeps one loop per tab (a generation counter) and holds
the busy flag from job selection until the result is committed. `run/e2e.js`
reproduces it with two simultaneous connects in one tab, plus a second tab,
behind a mock that adds 250 ms per request. Version 3 gave 1 overlap and one
job run twice; version 4 gave 0 overlaps, twice.
