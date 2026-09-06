# Errors — brief-compression

What was wrong, how it was caught, which way it pushed the conclusion.

## 1. The structured brief's `top_user (first report row)` was read as the row

8 of 10 Sonnet-structured runs wrote `top_user` as an object
`{"user": ..., "bytes": ...}`. The prose brief said "the first user in the
report" and the telegraphic brief said `top_user (row1)`; both were read as a
string every time. Caught by the scorer's `top_user_ok` check firing on one
arm only, then confirmed by reading the summaries. The three arms were meant
to carry identical information; on this one field the structured arm was
ambiguous and the other two were not.

Direction: this makes the structured arm look worse than it is. With the
check excluded the structured arm is 10/10 on Sonnet and 9/10 on Haiku, level
with prose. The headline (telegraphic loses, prose does not) does not depend
on it. Reported both ways in RESULTS.md.

## 2. The generator wrote `\r\n` line endings

`csv.DictWriter` defaults to `\r\n`. Nothing in the task mentioned line
endings; one Haiku run rewrote the CSV with `\n` and was scored as modifying
the input. Caught when the `input_unmodified` check fired with the file list
intact; diffing against a regenerated original showed zero content changes
and a line-ending change only.

Direction: one extra failed check on a run that had already failed
`rows_kept`, so no change to any success count. P4 ("no arm modifies the
input") is recorded as failed on that run because the check is byte-level and
the writeup should not soften a pre-registered check after the fact.

## 3. Task A did not discriminate

The first task was designed with 6 constraints and 9 checks and came back
22/22 before any cell had 10 runs. The pre-registered n = 20 per cell would
have spent ~4M subagent tokens confirming a ceiling. Task A was capped at 6
per cell and Task B written, with its own predictions, before Task B was
dispatched (PREDICTIONS.md amendment 1). The Task A predictions P1–P4 are
scored against n = 6.

Direction: none on the finding; Task A is reported as the ceiling it was.

## 4. Dispatch bookkeeping miscounted once

After batch 2 the dispatched set was computed as the first 32 ids in the
shuffled order when only 22 had actually been sent, so the "fill each cell to
6" pass skipped 10 runs. Caught when the per-cell counts in the scorer did
not match the plan; the missing 8 were dispatched afterwards. No run was sent
twice; the replies file is keyed by run id and would have shown it.

Direction: none.

## 5. Sonnet leaked self-talk before `DONE` on 3 Task A runs

"Good, the file itself is correct (64 hex chars, 3 keys) — the display
artifact above just had a copy-paste rendering glitch. Everything checks out.
DONE". Scored as a `reply_done` failure under the literal rule. All three were
prose or structured briefs; none telegraphic. n is too small to say the
register caused it.
