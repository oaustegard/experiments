# ENGINE.md: what runs lanes on this Mac

**Status: Step 0 is mostly done (2026-11-27).** Upstream `4be5bed` is now built at `tools/erisinky-batch`, and the
first lane checks have run (section 7). Every claim is tagged **[measured]** (a command ran and I saw it),
**[source]** (read in code, not run) or **[unverified]**. Section "Open checks" lists what remains before the
harness is written.

## Headline

The plan assumed a lane-batched ErisInky build is already on the Mac. **It is not.** What is built here cannot run
lanes. The upstream repo has since gained a batch engine, but nobody has built it on this machine.

| Candidate | Where | Runs N lanes? |
|---|---|---|
| erisinky `3890bd5` + local `serve` patch | `tools/erisinky`, binary built 2026-11-25 | **No.** One sequence. [measured: `inkling` usage lists chat/run/bench/serve/check only] |
| `llama-server` b12649 (Unsloth mix) | `~/.unsloth/llama.cpp/build/bin/llama-server` | **Yes, 4 slots by default.** HTTP, continuous batching, prompt cache. Not running now. [measured: `/health` refused connection; `INKLING-CONFIG.md` records 4 slots] |
| erisinky upstream `main` = `4be5bed685e937e4cb6f0143bb17afa5cb51a763` | `tools/erisinky-batch` (source tarball, no `.git`; tarball sha256 `871d1203aa6063a9f4df50b59414cea239be95b09471d416e29c911c64b1fc0a`), built 2026-11-27 with `cargo build --release` in 27 s | **Yes**, `inkling batch` runs 1–8 lanes. [measured, section 7] **This is the engine to use.** |

The plan cites HEAD `0e8c39d` and found no batched server. Upstream HEAD is now `4be5bed` [measured: `git ls-remote`],
and the local checkout is behind both (`3890bd5`, README line "Multi-sequence batching ... not implemented").

## 1. Repo, commit, model file

| Item | Value |
|---|---|
| Hardware | Apple M5 Max, `hw.memsize` 137,438,953,472 B (128 GiB) [measured] |
| Local erisinky | `tools/erisinky` @ `3890bd5`, working tree has our patch (`src/main.rs` +3 lines, untracked `src/serve.rs`) [measured] |
| Local binary sha256 | `65cbe15d702ae85f0efa92a0616f0ee83057da00fd22f9031e8ab99fa8f4bc45` [measured] |
| Model for erisinky | `~/.lmstudio/models/lmstudio-community/Inkling-2-Mini-GGUF/Inkling-2-Mini-Q4_K_M.gguf`, 17,478,094,048 B, sha256 `dac3702abcb5684cbcb1fbd148d94b2d531283be974630ac61964d194db483de` [measured] |
| Model for llama-server | `Inkling-2-Mini-UD-Q4_K_XL.gguf` (HF snapshot `45fd0876…`), plus DSpark drafter `dspark-Inkling-2-Mini-Q8_0.gguf`. Hash not taken. erisinky cannot load this file. [`INKLING-CONFIG.md`] |
| llama-server | `version: 1.0.0-dev (build 12649, commit 189f431bf)` [measured] |
| Upstream pin to test | `4be5bed685e937e4cb6f0143bb17afa5cb51a763` [measured via `ls-remote`] |
| Toolchain | rustc 1.98.1, cargo 1.98.1 [measured] |

The two engines load different files of the same model (Q4_K_M vs UD-Q4_K_XL). **Arms must not mix engines.**

## 2. How to submit N concurrent requests

**Upstream `batch` (the lane engine), all [source]:**

- `inkling batch -b N --prompts FILE [-n MAX] [--temp T --top-p P --top-k K --seed S] [--raw] [-q] [-c CTX] [--max-mem GB]`. `-q` is the quiet flag (`--quiet`
  is rejected as an unknown option [measured]). `-c` defaults to 16384, so 20K-token prompts need `-c 32768`.
- `--prompts FILE`: one prompt per line, each trimmed, blank lines dropped. **A prompt cannot contain a newline**, so
  a 4–5K-token shard needs escaping or a patch (`--prompts-jsonl`). Without `--raw`, each line is wrapped as
  system "Thinking effort level: E" + user turn (`first_turn`).
- `-b` is the lane count. With more prompts than `-b`, the extra prompts queue and take freed lanes (continuous
  batching, `generate_independent`). All active lanes share one decode step.
- `generate_shared` (no `--prompts`): one prefix, prefilled once, then `b` lanes decode from it with cascade
  attention. This fits `vote` / `passn` (same full context, different seeds) and R2 of `fwd` (same pooled facts to
  every lane). It cannot give each lane its own suffix.
- One process per call: load plus warm-up was 2.4 s + 0.13 s on the old build [`INKLING-CONFIG.md`]. A three-round arm
  is three calls, or a harness-owned process like our stdio `serve`. **There is no resident batched server upstream.**
  Upstream's `serve` is the WebUI / multi-stream LoRA server (`webui::run_server`, default adapter path is
  another user's) and is unrelated.
- `--max-mem GB` caps batch working memory. Default cap is min(10 GiB, below `recommendedMaxWorkingSetSize` minus
  weights minus 10 GiB). 64 lanes of 5K tokens may need a higher `--max-mem`. [source]

**llama-server (fallback), [`INKLING-CONFIG.md`] unless marked:** HTTP on `127.0.0.1:8080` via `start_server.sh`;
`-np/--parallel N` sets slots (default auto, 4 observed); `/v1/chat/completions` per request; continuous batching on
by default. Both bind and connect to localhost need `dangerouslyDisableSandbox`.

## 3. Prefix cache

- Upstream `batch`: **no cache across calls** (new process). Inside a call, only `generate_shared` reuses a prefix, and
  `generate_independent` sets `self.prefix = None`, so per-shard prompts that share a system prompt do **not** share
  it. [source] Discount "prompt tokens by cache hit rate" (plan, Equal budget) is therefore 0% for R1 on this engine.
- llama-server: prompt cache on by default (`--cache-prompt`), per slot, with `--cache-reuse` and
  `--slot-prompt-similarity` options. Hit rate is not reported to us except via server timings. [source of `--help`, measured]

## 4. Per-request tokens, wall-clock, aggregate tok/s

Upstream `batch` reports **aggregate only**, on stderr [source]: `prefill <tok> @ <tok/s> (<ms>) | decode <tok> in
<steps> @ <tok/s> (<tok/s/seq>, <ms/step>) | mem KV+CS+Act`. Stdout, unless `-q`, gives `=== [i] (n tokens) ===`
per sequence, so **completion tokens per lane are available, prompt tokens and per-lane wall-clock are not.** Needed
for the harness: a small patch that prints per-sequence prompt tokens and time-to-finish, or count prompt tokens with
the model's tokenizer (the erisinky tokenizer is internal; `llama-server /tokenize` can count them for the same
vocabulary, **[unverified]** that the two tokenizers agree on chat-template tokens).

llama-server returns `usage` and `timings` per request [`INKLING-CONFIG.md`, `inkling_chat` output].

## 5. Decoding, seeds, bit-identity

- **Greedy loops on this model.** Three sources agree: our llama-server notes ("loops verbatim until the turn cap"),
  the upstream README (recommends temp 1.0, top-p 0.95, top-k 64 for agent loops), and the engine author's coding
  matrix (eriskii.net/research/inkling-ikv, pasted by Oskar 2026-11-27: `--greedy` run M8 was 110.7 s slower than the
  matching sampled run, looped through five overlapping slices, and scored 4.5/6 against 5.0–6.0 for the rest).
  That is one run on a coding task, not a test on ours. The plan's "greedy decoding in every arm" collides with this
  for the extraction round (R1). **Decision needed in Step 1:** either run all arms at one sampled setting with
  fixed per-task seeds, or run a 20-task greedy pilot and check for loops before committing.
- Upstream seeding: lane `i`'s RNG is `seed + i * 0xD1B54A32D192ED03 ^ 0x9E3779B97F4A7C15` and the RNG state is indexed by
  **slot**, not by request (`init_rng`, `generate_independent`). With queued prompts, which request lands in which
  slot depends on scheduling, so sampled output may not reproduce per request unless every prompt has its own lane
  (`-b >= number of prompts`). [source, not run]
- `solo` vs lane bit-identity: **not bit-exact on this build.** Greedy 80-token outputs differ between 1 lane and 2, 4
  or 8 lanes for some prompts, while each lane count repeats itself exactly (section 7, [measured]). The plan's
  note that inky's batched path is bit-exact does not hold for upstream `4be5bed`.
- KV precision: the default is fp16, and `--kv8` is opt-in and inexact [measured: local `--help`; source: batch engine prints
  `fp16`/`kv8`]. The plan's "keep KV at default precision" is satisfied by not passing `--kv8`. Never pass it.
  Do not confuse it with the sparse `Kv256kPool` / `--kv-bits`, which are 3–4 bit and not for this experiment.

## 6. Power and machine state (2026-11-27, 21:5x)

[measured, `pmset -g`]: `powermode 0`, "Now drawing from 'AC Power'", battery 80%, AC attached, not charging. No
`lowpowermode` line printed, which `pmset` does when it is on, so Low Power Mode is off. The harness must re-check
`pmset -g | grep -i lowpowermode` and `powermode` at every launch and refuse to run if either is nonzero.

Contention before any timing run: **an `inkling serve` process (PID 58792, up ~2.15 days) is holding the model**
for the `inkling-fast` MCP [measured, `fast_status`: `n_past` 17,257, `dirty` false]. It looks idle but is resident.
Stop it, and Unsloth Studio if running, before wall-clock measurements, or report timings as contended.
`llama-server` is not running.

Sandbox: Metal needs `dangerouslyDisableSandbox` (llama-bench failed with `failed to allocate context` in the sandbox,
per `INKLING-CONFIG.md`), and `git` could not write `.git/config` or `.git/hooks` in it today [measured]. I have not
tried `cargo build` in the sandbox.

## 7. First lane run on the built `4be5bed` (2026-11-27) [measured]

Setup: `tools/erisinky-batch/target/release/inkling batch -b N --prompts p8.txt --temp 0 -n 80 --effort none`, run
unsandboxed. `p8.txt` is 8 short one-line prompts, one of them (index 6) ~2,000 tokens of filler plus a question.
Model: the Q4_K_M file above, default fp16 KV, 16.3 GiB mapped. The idle `inkling-fast` engine was still resident, so
timings are contended. Load took 6.3 s cold and 0.09 s on later runs. Scratch files are not kept in the repo.

**Repeatability.** Each lane count reproduces itself: `-b 1` run twice gave 8/8 identical sequences, and `-b 8` run
twice gave 8/8 identical.

**Lanes do not match solo.** Per sequence, 1 = identical to the `-b 1` run:

| Compared with `-b 1` | Seq 0 1 2 3 4 5 6 7 | Differ |
|---|---|---|
| `-b 2` | 1 1 1 1 1 1 0 0 | 2 of 8 (seq 6 ended at 79 tokens, others 80) |
| `-b 4` | 1 1 1 1 1 1 0 0 | 2 of 8 |
| `-b 8` | 1 1 1 0 1 1 0 0 | 3 of 8 |

Sequences diverge partway through the reasoning text (token position not measured) and which sequences diverge depends on the lane count, so
a lane's answer depends on what runs beside it. Likely cause: batched kernels sum in a different order. Not diagnosed.
Consequence for the experiment: **an exact-match answer can flip between runs of the same task if the lane
count or the other lanes change.** Arms must run with a fixed lane layout per task, and `solo` should be run through
the same engine at `-b 1`, not compared with a different engine. How much this noise moves accuracy on real
tasks is not measured.

**Throughput** (aggregate decode, 619 tokens across 8 sequences):

| Lanes | tok/s aggregate | ms/step | tok/s per lane | vs `-b 1` |
|---|---|---|---|---|
| 1 | 36.9–38.5 | 26.3–27.4 | 36.9–38.5 | 1.0x |
| 2 | 63.0 | 31.1 | 31.5 | 1.7x |
| 4 | 103.8 | 37.7 | 26.0 | 2.7x |
| 8 | 159.5–169.9 | 46.1–49.1 | 19.9–21.2 | 4.2–4.4x |

So lanes pay: 8 lanes give about 4.3x aggregate decode, at about half the per-lane speed. The plan's "more than 500 tok/s"
at 64 lanes is not tested; 8 lanes top out near 170 tok/s here. Prefill was 2.4–3.4k tok/s on a 2.5K-token batch, too
small to read anything into. Not yet measured: prefill on 4–5K-token shards, which is the reason for splitting.

## 8. Shard-shape timing and lane sweep (2026-11-27, same build) [measured]

Conditions: the `inkling-fast` engine was stopped first (`fast_reset stop_engine`), so no other Inkling process was
loaded. One idle `mlx_lm.server` (PID 1306, a Nemotron model, 0.0% CPU) was still resident, and I did not stop it.
AC power, `powermode 0`. Upstream HEAD re-checked as `4be5bed` just before the run. Greedy, `--effort none`, `-n 100`,
`-c 32768`, three runs per row. The text is the same ~19.6K tokens of random-syllable words in every row; it
is only split differently. Prompts include the chat-template overhead, so the totals differ by up to 1%.

| Shape | Prefill tokens | Prefill time | Prefill tok/s | Decode ms/step | Wall (load + prefill + 100 decode) |
|---|---|---|---|---|---|
| 1 × 20K, `-b 1` | 19,620 | 4.77 / 5.00 / 5.53 s | 3.5–4.1k | 27.9 | 7.9 / 8.1 / 8.6 s |
| 4 × 5K, `-b 4` | 19,721 | 5.58 / 5.59 / 5.70 s | 3.5k | 33.7–34.5 | 9.3 / 9.4 / 9.5 s |
| 8 × 2.5K, `-b 8` | 19,853 | 5.50 / 5.45 / 5.73 s | 3.5–3.6k | 41.9–42.1 | 10.1 / 10.0 / 10.3 s |
| 4 × 5K, `-b 1` one after another | 19,721 | 5.43 / 5.29 / 5.36 s | 3.6–3.7k | 27.0–27.2 | 16.5 / 16.4 / 16.5 s (400 decode tokens) |

**Splitting does not shorten prefill on this engine.** The same ~19.7K tokens take 4.8–5.5 s as one prompt and
5.3–5.7 s as 4 or 8 shards, at about 3.5–4k tok/s whichever way they are cut. The plan's expectation that
full-attention cost "drops more than linearly" does not show up at 20K. A likely reason (not tested) is that only 8 of
the 48 layers use full attention, so attention is not the main prefill cost. Lanes do not prefill in parallel;
the GPU looks saturated by one 4K-token prefill chunk.
In one round, the 4-shard `fwd` shape finished slower than `solo` (9.3–9.5 s vs 7.9–8.6 s), because each decode step
is slower with more lanes. R2 and R3 would add more. So the plan's P5 first clause (`fwd` faster than `solo` at equal
tokens) is likely to fail on wall-clock. I have not edited the registered predictions in `PLAN.md`.

**Lanes pay across tasks, not within one.** Aggregate decode, `bench-batch`, 23-token prompt, 64 new tokens, sampled
at temp 1.0 [measured]:

| Lanes | Independent: tok/s (ms/step) | Shared prefix: tok/s (ms/step) | Speedup vs 1 lane (shared) |
|---|---|---|---|
| 1 | 37.6 (27.0) | 37.5 (27.1) | 1.0x |
| 4 | 119.4 (34.0) | 121.6 (33.4) | 3.2x |
| 8 | 194.1 (41.9) | 205.6 (39.5) | 5.5x |
| 16 | 327.5 (49.6) | 349.6 (46.5) | 9.3x |
| 32 | 530.4 (61.3) | 556.0 (58.5) | 14.8x |
| 64 | 825.3 (78.8) | 907.6 (71.6) | 24.2x |

So the plan's "up to 64 lanes, more than 500 tok/s aggregate" holds for decode on short contexts (825–908 tok/s at 64). The
5K-shard rows above show no decode penalty for context at 4 and 8 lanes (120 and 192 tok/s, against 119 and 194 on
23-token prompts); 64 lanes at 5K context was not run. Prefill stays about 3.2–4k tok/s at any lane count, so a run
of many concurrent tasks is prefill-bound first: at 20K tokens per task, one task needs about 5 s of prefill, or about 720
tasks/hour at most, however many lanes are used. Decode gains from lanes apply to `solo`, `vote` and `fwd`
alike when tasks run concurrently, so tasks per hour will mostly measure total prefill tokens, not the arm's lane count.
`fwd` adds R2/R3 prefill (the pooled facts, once per round) on top of R1's.

## Open checks (these block `harness.py`)

A. ~~Build upstream `4be5bed`~~ **Done**, `tools/erisinky-batch`. Our `serve.rs` patch is not ported, and upstream's
   `serve` is a different thing (section 2).
B. ~~Solo vs lane bit-identity~~ **Done: not identical** (section 7). Still to check: sampled output with fixed seeds
   (slot-dependence from section 5), and whether `-b 1` through the lane code path matches the old single-lane
   `inkling run` (probably not needed).
C. ~~Prefill and wall-clock on the real shape~~ **Done** (section 8). Not run: 64 lanes at 5K context, and the
   `--shared` path with a 20K shared prefix for `vote`.
D. **Newline-safe prompt input and per-sequence stats**: the small patch described in sections 2 and 4.
E. **Greedy-loop pilot** (section 5) before fixing the decoding setting.
F. **Tokenizer agreement** if llama-server is used to count prompt tokens.

## Plan changes this survey implies

- Section "Why the M5 Max makes this worth testing" quotes lane speedups (1.92x at 4, 1.32x at 8, 2.83x at 64) from other
  builds. On this build the measured aggregate speedups are 2.7x at 4 lanes and about 4.3x at 8. Use these in P5, and
  re-measure after check C.
- The plan's `solo` arm should be `-b 1` on the same engine. Lane noise (section 7) goes in the writeup as a limit, and
  each task's lane layout is fixed so that a rerun reproduces it.
- The harness cannot assume a resident lane server. Either one process per round (about 3 s overhead each, 0.1 s once
  the model is in the page cache) or a stdio batch mode, which is new code.
- **"Prefill is practically instant" (the author's Bluesky posts, pasted by Oskar 2026-11-27) means precompiled
  `.ikv` cartridges, not faster raw prefill.** The posts say prefill is "amortized to idle periods or other nodes" and
  loads "instantly into context"; the article puts the cost at 0.57 s of stitching for 46K tokens against 94.6 s raw (sparse mode).
  What section 8 measures is raw prefill of unseen text, about 3.5–4k tok/s, which agrees with the README (2,200 tok/s on an
  M5 Pro). Our shards are seeded, so they could be precompiled, but compiling costs the same prefill moved elsewhere, the
  pasted KV is approximate (author's KL 0.015–0.03), near-identical filler is the author's own blur case (2/10 on
  repeated logs), and `inkling-ikv` is not on this machine. The main runs stay on raw prefill. A precompiled variant
  would answer a different question ("does `fwd` still pay when prefill is free?") and is not planned. If prefill is
  free, the only cost left is decode, where lanes are slower per step, so this does not help P5′a.
- The author's article describes KV pasting between agents (TokenDance Master-Mirror, `kv-share-probe`), which shares
  context instead of forwarding facts. That is a different arm from this plan's, so it is not added here.
