# deepseek-batch

An overnight batch coding harness for a local model, and the benchmark
landscape that decided its shape. Nothing here has been run against a live
endpoint — the runner passes `bash -n` and nothing more.

The number that sets the design is prefill, not decode. Oskar reports
DeepSeek V4.1-Flash at 2-bit on an M5 Max 128 GB doing **17 tok/s decode and
~40 tok/s prompt processing**. The decode figure sits on the published
ceiling for that chip and quant. The prefill figure is roughly 20× below it.

## What has been measured, by whom

| source | hardware | build | decode | prefill |
|---|---|---|---|---|
| [antirez, ds4/DwarfStar](https://github.com/antirez/ds4) | M5 Max 128 GB | Q2, SSD-streamed experts | 16 t/s | ~800 t/s |
| [Vontra, HF model card](https://huggingface.co/Vontra/DeepSeek-V4.1-Flash-MLX-2bit-MTP) | M3 Ultra 256 GB | MLX affine 2-bit g64 + MTP | 9.46 t/s | not measured |

Both are self-reports. [`evanwtf/local-llm#321`](https://github.com/evanwtf/local-llm/issues/321)
is open at P2 asking for an independent M5 Max 128 GB reproduction of the ds4
numbers, and records them as "external measurements, not yet reproduced."

The Vontra build is not a 128 GB path: 239 GB on disk, 160.88 GiB text
backbone plus 57.22 GiB of Engram tables on SSD, 166.69 GiB peak RSS. Its MTP
weights measured *slower* than the non-MTP adapter (8.54 t/s, 1.33–1.00
accepted drafts per block). The card warns that two-bit requantisation "can
substantially reduce quality relative to the already-quantised source;
repetition, incoherence and task failures remain possible."

A [2026-09-10 write-up](https://www.modemguides.com/blogs/ai-infrastructure/run-deepseek-v4-1-flash-locally-hardware-reality-check)
states that V4.1-Flash "cannot be run on a 128GB machine at any quantization
level" and that "no released engine serves the deepseek_v41 architecture."
Both claims were false on the day they were published — ds4 shipped it and
Oskar is running it. The one datum worth carrying out of that piece is its
quality claim: 78% top-token agreement with the full model at 2-bit.

DeepSeek V4.1-Flash itself: released 2026-09-10, MIT, 552B total / 16B active
MoE, native multimodal, global KV cache 890 B/token.

## Why prefill decides whether any of this works

An agent turn is a large prompt and a small reply, so the loop is
prefill-dominated. At 30k tokens of context and 600 tokens out:

| prefill rate | per turn | 25-turn task | 10-hour night |
|---|---|---|---|
| 800 t/s | ~75 s | ~30 min | ~20 tasks |
| 40 t/s | ~13 min | 5+ hours | 2 tasks |

Recovering the 20× is worth more than any harness design, so check the engine
before building on top of it: ds4 rather than MLX or llama.cpp, the SSD
experts cache warm (antirez reports an 8 s load to full encoder residency),
and the full-resident-encoder prefill strategy rather than the small-prefill
expert-cache or layer-major ones.

The runner therefore probes prompt-processing speed once per run and writes
the figure into the report, so a collapsed prefill rate shows up as a number
in the morning rather than as a mysteriously empty queue.

## Harness choices

**Aider headless, one process per task.** `--message-file` + `--yes-always` +
`--auto-test --test-cmd` gives the edit → lint → test → feed-failures-back →
fix loop natively. It is git-native, so every task is an inspectable branch,
and the repo-map keeps context small, which matters more here than anywhere.

**Not OpenCode.** [`anomalyco/opencode#16589`](https://github.com/anomalyco/opencode/issues/16589)
documents it stalling on long unattended runs — "it will do a dozen or so and
then stop."

**Nothing that compacts context.** Compaction rewrites the middle of the
prompt, so the cached prefix diverges from the previous turn's and the whole
thing re-prefills. On a prefill-bound machine that is disqualifying. DeepSeek
Harness has four modes; only `minimal` (fixed prompt, bash + `str_replace_editor`,
no compaction) is viable, and it is the right pick for anyone who would rather
use the model's own harness.

**Server resident 24/7.** Never restart between tasks; pin a shared system
prefix with ds4's `--prefix-file`.

**`launchd` with `StartCalendarInterval`, wrapped in `caffeinate -s`.** Not
cron — sleep/wake handling.

## The gate

At 2-bit, the gate matters more than the agent. A degraded model running
unsupervised produces confident garbage at scale, and the cost lands on the
morning triage rather than the overnight compute.

So every queued task carries a machine-checkable acceptance command or it is
not queueable. `deepseek-batch.sh` enforces:

- one git worktree per task, off `origin/$BASE`; the working tree is never touched
- **baseline before the agent runs.** A fix task whose test is already green is
  a no-op; a refactor task whose suite is already red cannot be judged
  afterwards. Either way the run is worthless, so the 60 s spent finding out
  is the cheapest check in the file.
- after the run: the targeted test, then the full suite for collateral damage
- a diff-size bound, because "fix the null check" should not return 4,000 lines
- pass → commit and push the branch; fail → keep branch, patch and logs as
  diagnostic material, move to the next task
- a deadline hour, so a slow queue stops starting new tasks instead of running
  into the working day

Two of those come straight from [`../harness-bench`](../harness-bench/RESULTS.md),
which measured this loop rather than assuming it. Certifying each task both ways
before any agent runs is its practice, and it is what caught a shared
`CARGO_TARGET_DIR` scoring a `todo!()` stub as 23 tests passed — a gate that
cannot fail is worse than no gate, since it ships the garbage with a green tick.
The runner therefore points every build cache it knows about inside the
worktree; extend that list for any toolchain whose cache lives elsewhere.

That experiment also supplies the reason to expect this to work at all. At a
fixed model, single-shot edits scored 3/12 on Aider Polyglot, handing the test
output back once scored 8/12, and letting the agent run the suite itself scored
11/12 — 8 recovered, 0 regressed, exact McNemar p=0.0078. The test loop is the
lever, and most of what it buys is not reasoning. That is the part a 2-bit model
can still collect.

`METHODS.md` also gives the sustained-throughput form properly:
`1/(1/decode + fresh_ratio/prefill)`, where prefix caching makes
`fresh_ratio = (1 − hit_rate) × billed_ratio`. The per-turn table above is that
expression at `hit_rate = 0`, which is the honest assumption for a harness that
has not yet been shown to hold a stable prefix.

Task selection is half the design. Give it work where verification is total:
test-writing against existing code, type annotations, dead-code removal,
deprecation and lint sweeps, porting one pattern across N files, fixing a
failing test with a known-good spec. Not design, not API shape, not anything
where "looks right" is the only available check.

## Usage

```bash
export BATCH_HOME=~/batch                       # default
export BATCH_API_BASE=http://127.0.0.1:8000/v1  # ds4's server
export BATCH_MODEL=deepseek-v4.1-flash

mkdir -p $BATCH_HOME/queue
cp example.task $BATCH_HOME/queue/010-hints.task   # numeric prefix = run order
./deepseek-batch.sh --dry-run                      # check it resolves
./deepseek-batch.sh
```

Reports land in `$BATCH_HOME/reports/<run-id>.md`, per-task logs in
`$BATCH_HOME/logs/<task-id>/` (agent transcript, baseline/post/full test
output, the staged patch).

A `.task` file is sourced as bash — it is your own file, and it is code.

For unattended scheduling, point a `launchd` plist at
`caffeinate -s ./deepseek-batch.sh`.

## Not verified

- The runner has never run against a live endpoint. `bash -n` passes;
  shellcheck was unavailable where it was written.
- Every performance figure above is someone else's self-report.
- Aider's exact flag spellings are from its docs, not from a local run.

Start with `--dry-run` and a single task whose outcome you can predict.
