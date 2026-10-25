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
- `web/runtime.js`: the worker runtime (`ctx.gpu`, `ctx.compute`, `ctx.cache`, `ctx.log`).
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

`knn-1` also exposed two open tabs running the same job: one failed its 32 MB
download ("Load failed") and committed an error, the other overwrote it with
the good result. Page version 3 claims a job with a commit before running it
(a second tab sees the claim and backs off) and retries failed downloads;
`run/e2e.js` now runs two tabs on one repo and checks each job runs once.
