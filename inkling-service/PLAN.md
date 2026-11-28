# inkling-service: Inkling 2 Mini as a shared, lane-batched local service

## Question

Can one resident Inkling 2 Mini process on the M5 Max serve requests from several Claude sessions
at once, putting concurrent requests into lanes, and how much does that buy over the one-batch-per-call
MCP (`inkling-fact-forwarding/tooling/inkling_batch_mcp.py`)? Second question: what part of
running it should a model decide (the coordinator), and what part has to be code?

## What exists (2026-11-28)

From `inkling-fact-forwarding/ENGINE.md` (measured on this Mac) and upstream erisinky `4be5bed`
(read, not run):

- **The lane engine already does continuous batching, inside one call.** `generate_independent`
  (`src/batch.rs:1629`) keeps a queue of prompts, admits queued prompts into freed lanes
  (`prefill_packed`), retires finished ones, and decodes the rest in one shared step
  (`run_pipelined_decode_until`, which returns early once enough lanes free up). It runs over a fixed
  prompt list and exits. Upstream has no resident batched server; its `serve` is the unrelated
  multi-stream LoRA web UI.
- **Decode scales with lanes; prefill does not.** Aggregate decode 37.6 / 119 / 194 / 328 / 530 /
  825 tok/s at 1 / 4 / 8 / 16 / 32 / 64 lanes (short prompts); prefill 3.5-4k tok/s at any lane count.
  A 20K-token request costs about 5 s of prefill, so a service is prefill-bound first: about 720
  such requests an hour at most.
- **One lane is slower than the single-stream path.** The lane engine's 1-lane path decodes at about
  38 tok/s; the dedicated single-stream path at 90-105. Lanes win from about 4 concurrent requests.
- **No prefix reuse across calls.** Each call prefills its prompts from scratch, and
  `generate_independent` clears `self.prefix`.
- **Lane outputs depend on neighbours.** Each lane count repeats itself exactly, but a lane's output
  differs from a solo run on 2-3 of 8 prompts. Fine for a service; experiments that need exact
  reproduction must not share it.
- **Measurements so far were contended.** A `bonsai-metal-bench` `llama-bench` ran on the GPU during
  the lane tests and cut single-stream decode to 11-13 tok/s. `_lib/gpulease.py` (2026-11-28) now
  gives every GPU job a lease; `bench.py`, `ane-workloads` and the batch MCP take it.

## Design

### The service (code, in the engine)

A new `inkling serve-batch` mode, as a patch on upstream `4be5bed` like `erisinky-batch-json.patch`:

1. **Resident process, one `BatchEngine`.** Weights loaded once. Requests arrive over a Unix socket
   as JSON lines: `{id, messages | prompt, max_new, temperature, top_p, top_k, seed, session}`. Replies
   stream tokens, then `{id, finish, prompt_tokens, gen_tokens, queue_ms, ttft_ms, wall_ms}`.
2. **Scheduler = `generate_independent`'s loop, fed by a channel.** The fixed `queue` becomes a
   receiver. The decode loop also returns when a request is waiting and a lane is free (check the channel
   every few steps), so new arrivals do not wait for a whole batch to drain.
3. **Per-request sampling.** Sampling config and RNG are per request, seeded from the request's seed.
   Today they are per call, and the RNG is indexed by slot (`ENGINE.md` §5).
4. **Chunked prefill.** Admitting a 20K-token request stalls every running lane for about 5 s. Prefill new
   requests in chunks of at most about 2K tokens, interleaved with decode steps. If `prefill_packed` cannot resume
   a sequence mid-prompt, cap the prompt tokens admitted per scheduler pass as a first step and measure the
   stall.
5. **Memory admission.** Admit a request only when its KV pages fit under `--max-mem`; otherwise it queues.
6. **GPU lease.** Hold a shared `_lib/gpulease.py` lease while serving. Check `drain_requested()`
   between scheduler passes; on a drain, stop admitting, finish what is running, release, and re-acquire
   once the exclusive job is done.
7. **Client.** `inkling_batch_mcp.py` sends its prompts to the socket when `INKLING_SERVE_SOCKET` is set,
   and falls back to `batch-json` otherwise. Any Claude session with the MCP shares the one process.

### Session prefix cache (code, in the engine; the big win for agent traffic)

A Claude session's successive requests share a growing prefix. Keep each finished sequence's KV,
keyed by its token prefix in 64-token blocks. When a new request's tokens extend a cached prefix,
prefill only the new tokens. Evict least-recently-used entries when memory runs short. On inky's
16 x 8K shared-head case, the same idea cut end-to-end time about 100x
(`robotgamedevelopers/inky/docs/PERFORMANCE.md` §4B).

### The coordinator (a model, at the policy level)

Lane scheduling happens every 30-80 ms per decode step and stays in code. A model fits the decisions
made every few seconds or minutes, where the inputs are messy:

- **Routing:** for each incoming request, decide local Inkling, local Bonsai or cloud Claude, from the task text,
  the current queue wait and the lane count. Jev (`decide`, about 1 s, a fraction of a cent) is fast
  enough to call per request. Start with Jev's choice logged beside a fixed rule, and switch only where
  Jev measurably does better.
- **GPU time:** decide when an exclusive job (a benchmark, a long Bonsai run) gets the GPU. That means draining
  the service through the lease at a quiet moment and resuming it after. A Claude session reading
  `python3 -m _lib.gpulease status` and the service's stats can do this; so can a small daemon that asks
  Jev "start the queued exclusive job now?".
- **Batch planning:** decide which queued offline work (fact-forwarding arms, embedding jobs, benchmark rounds) to
  run overnight, and in what order.

## Tasks (on the M5 Max)

Run every GPU command under the lease, e.g.
`python3 -m _lib.gpulease run --exclusive --name <job> -- <command>`, after stopping resident
holders (`inkling-fast`'s `inkling serve`, `mlx_lm.server`, Unsloth Studio). Record
`python3 -m _lib.gpulease status` before and after each run; `unleased_gpu_processes` must be empty.

### Task 0. A clean reading

Re-measure on a quiet GPU, under an exclusive lease, with the `batch-json` build from
`inkling-fact-forwarding/tooling`:

- single-stream decode and prefill (`inkling bench`, 2.4K-token prompt, greedy and sampled at temp 1.0);
- the lane sweep (`inkling bench-batch`, 1 / 4 / 8 / 16 / 32 / 64 lanes, independent and shared, 64 new tokens);
- 64 lanes at 5K-token prompts, which `ENGINE.md` did not run;
- prefill of one 20K-token prompt and of 4 x 5K.

Three repeats each. Write `results/<run>/` and a `RESULTS.md` (repo shape, evidence links),
and compare with `ENGINE.md` §7-8, saying which of its numbers moved.

### Task 1. `serve-batch`

Build the service as designed above, as `erisinky-serve-batch.patch` against `4be5bed` (GPL-3.0-or-later,
like the batch-json patch), plus the client change in `inkling_batch_mcp.py`. It is done when:

- 1, 4, 16 and 64 concurrent clients each get complete, well-formed replies;
- a single request with a fixed seed reproduces itself across runs;
- the lease drain works: an exclusive job started mid-service gets the GPU within one scheduler pass
  after in-flight requests finish.

Measure, against the one-shot `batch-json` path, for 1 / 4 / 16 / 64 concurrent clients with 2K-token
and 20K-token prompts:

- queue wait, time to first token and wall time per request;
- aggregate decode tok/s;
- the decode stall that a 20K-token arrival causes in already-running lanes.

### Task 2. Session prefix cache

Add the cache. It is done when a session's second request, extending its first by 1K tokens, prefills
only those tokens, and its output is consistent with a cold run (same seed; a lane-neighbour difference is
acceptable). Then measure time to first token for a replayed multi-turn Claude session: the
`tests/replay`-style trace from inky's repo, or recorded MCP calls.

### Task 3. Coordinator, first version

- A routing hook in the MCP that asks Jev, per request, to choose between local Inkling and cloud Claude.
  Log Jev's choice and a fixed rule's choice side by side; do not act on Jev's choice yet.
- A `gpu-coordinator` script that, given a queue of exclusive jobs, drains and resumes the service through
  the lease.

Report how often Jev's choice and the rule disagree and, on a sample, which choice was right.

## Predictions (to check, not assumed)

- P1. Task 0 single-stream decode on a quiet GPU lands at 95-110 tok/s, and 64-lane aggregate decode
  above 800 tok/s.
- P2. Through `serve-batch`, 16 concurrent 2K-token requests finish at least 2x sooner in total than the
  same 16 sent as separate one-shot calls.
- P3. Without chunked prefill, a 20K-token arrival stalls running lanes about 5 s; chunked at 2K, the
  longest stall drops below 1 s.
- P4. With the prefix cache, time to first token on a session's later turns falls more than 5x on a replayed
  agent trace.
