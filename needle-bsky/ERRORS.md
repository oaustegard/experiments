# Errors — needle-bsky

What was wrong, how it surfaced, and which way it pushed the conclusion. Base
rate matters more than any individual entry.

## 1. `loss 0.0000` was truncation, not convergence

**What.** The first LoRA run reported `step 5/270 loss 0.0000` and I nearly read
it as "the task is trivial for the model".

**Cause.** Every training row declared all 18 tools. `gen_data.py` wrote the
full catalogue into each row's `tools` field, which tokenizes to **1,642
tokens** against `--max-len 1024`. `finetune.py` truncates at `max_len`
(`_encode`, line 215), so the answer — which follows the tool block — fell
outside the window on every row. All label positions were masked; the loss was
exactly zero because there was nothing to predict.

**How caught.** The value itself. A loss of exactly 0.0000 at step 5, before
warmup finished, is not a number a real run produces.

**Verified by a disjoint path.** Rather than trust the reading, the row was
tokenized directly with `needle.model.tokenizer.SANTokenizer` — the same
tokenizer the trainer uses, but reached without going through the trainer:
18 tools → 1,642 tokens, 5 tools → 467 tokens, current rows → 461 median.

**Fix.** `gen_data.py -k 5`: declare five tools per row (correct one plus four
seeded distractors), which is also what retrieval actually renders at inference
time, so the training context now matches the decode context.

**Direction.** Would have produced a "fine-tuning does nothing" negative result
from an adapter that had seen no gradient. Caught before any number was
reported.

## 2. `init_seconds` measured nothing

**What.** `eval.py` timed `Router(...)` construction and reported it as engine
init. The three arms reported 1.87–3.51 s, which looked plausible.

**Cause.** `needle.Needle.__init__` does not bind the engine; `_bind()` runs on
first use. The number was measuring schema construction and a `dlopen` that had
not happened yet. The real ~2 s lands on the first `complete()` of a session.

**How caught.** A separate `tool_index_path` probe reported `init 0ms` for the
second and third agents in the same process, which is impossible if init were
being measured where it was claimed.

**Direction.** Cosmetic — it never entered a comparison. Stated in RESULTS.md
because anyone timing Needle setup will hit the same lazy bind.

## 3. `pkill -f 'needle finetune'` matches itself

**What.** Two cleanup calls returned exit 144 and aborted the rest of their
compound command, so a "kill then relaunch" never relaunched, and a later poll
read a stale log as if it were the new run's.

**Cause.** `pkill -f` matches against full command lines, including `pkill`'s
own, so the pattern kills the pkill.

**Fix.** `pgrep -f 'bin/needle' | xargs -r kill`, or a pattern that cannot match
the matcher. Long jobs moved to the harness's own background mechanism rather
than `setsid`, which the environment does not reliably keep alive.

**Direction.** Cost wall-clock and one misread poll; no number was affected. It
did briefly support a wrong story ("the fine-tune was OOM-killed") that the
process table did not actually support.

## 4. A claim about write-refusal that the data did not support

**What.** The first draft of RESULTS.md said the eval set had five write-shaped
queries and the router refused four. It has four, and the router refuses two or
three depending on the arm.

**How caught.** Checking the claim against `evalset.jsonl` and the per-row
results before publishing, rather than after.

**Direction.** Would have overstated the model's refusal behaviour on exactly
the axis that matters for safety. Corrected in place, and the caveat now says
the read-only catalogue — not the model — is the boundary.

## 5. Two trainers running at once, from a misread process table

**What.** After two `ps` checks appeared to show no `needle finetune` process, I
concluded the run had been OOM-killed and relaunched it — twice. All three were
alive. At the point I noticed, two trainers were competing for four cores and
9.6 GB, which is why the second one looked stuck in compile for 16 minutes.

**Cause.** Both `ps` invocations produced output large enough to be truncated
into a persisted file, and I read the preview rather than the file. Absence of a
line in a preview is not absence of the process. The third relaunch also broke
the log: two processes held `> ft.log`, one `rm -f` unlinked the inode the
survivor was writing to, and the visible `ft.log` afterwards belonged to the
dead run. The survivor's output was only readable through
`cat /proc/<pid>/fd/1`.

**Fix.** Kill the duplicate, read progress from `/proc/<pid>/fd/1`, give each
run its own log path. When a `ps` result is truncated, grep it rather than
skim it.

**Direction.** Cost roughly 40 minutes of wall-clock and produced one wrong
diagnosis in this file's own draft ("OOM-killed"), which is corrected here.
No measured number was affected.

## 6. Every latency number measured while the trainer ran is inflated ~12x

**What.** A probe of the agent-switch cost returned 3,644 ms for a single
five-tool agent, against 284 ms measured earlier for the same configuration.

**Cause.** The LoRA run was using ~175% of four cores throughout. Needle
inference on CPU is compute-bound, so a concurrent trainer moves every timing by
an order of magnitude.

**Fix.** All latency in RESULTS.md is from runs with nothing else on the box;
the switch-cost probe was re-run after training finished. Accuracy is unaffected
— decoding is deterministic and the calls were byte-identical — so the two-stage
accuracy numbers collected during training stand.

**Direction.** Caught immediately, because the number contradicted a measurement
already in the writeup. A latency finding measured only once, with no baseline
to contradict it, would have shipped.

## 7. Two confusion counts stated from memory, not from the rows

**What.** The two-stage section shipped one commit claiming "every `plumbing`
query went to `follow_graph` or `find_content`, and 7 of 8 `account` queries
went to `find_content`". Counting the rows: 15 `account` queries, of which 9
went to `find_content` and 3 were right; 8 `plumbing` queries, of which 1 was
right and the rest scattered across four groups including `one_post`.

**Cause.** I read the error list printed by `two_stage.py`, which shows only
the failures, and described the distribution from it. A list of errors does not
carry the denominator or the successes.

**Fix.** Counted with `collections.Counter` over `group_expected` /`group` and
replaced both figures. `recheck.py` cannot catch this class — the claim is prose
about a distribution, not a headline number — so the guard is counting rather
than describing.

**Direction.** Overstated how systematic the failure was, in the direction of
my own argument. Corrected within the hour, one commit later.
