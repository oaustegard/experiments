# ERRORS — caps-emphasis

What was wrong, how it was caught, and which way it pushed the conclusion.

## 1. Item screen compared a whole word against a single token

`screen_items.py` first tested `top1_string == forbidden_word`, which fails for
every multi-token word: Lisbon decodes as `' Lis'`, Warsaw as `' Wars'`, Sahara
as `' Sah'`. Six valid items were dropped as "model does not know this" when the
model in fact put the correct first token top-1 at p > 0.85. Caught by reading
the `top1` column, where the dropped strings were visibly prefixes of the target
rather than wrong answers. Fixed by comparing token ids: the target's *first*
token must be argmax. Direction: it shrank n (19 -> 43 after the fix plus a
second tranche of items), which would have cost statistical power, not biased
the effect.

## 2. Two copies of the suppression sweep ran at once

The first launch chained `cd … && nohup … &` with a `tail` that ran after the
harness reset the shell's cwd. The `tail` failed, the launch did not, and the
error message ("cannot open suppression.log") read as "nothing started". A
second launch then put two identical jobs on four cores. Caught by
`ps -eo pid,etime,cmd`, not by the log. No effect on results — the script is
deterministic and both jobs would write identical output — but it halved
throughput for three minutes. Lesson already in the repo's METHODS.md as
"verify the watcher": the absence of an expected log line is not evidence the
job is absent.
