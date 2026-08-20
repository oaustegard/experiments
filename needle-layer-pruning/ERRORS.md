# Errors — needle-layer-pruning

What was wrong, how it surfaced, and which way it pushed the conclusion.

## 1. `PREREG.md` says eight confounded arms; there are seven

**What.** The pre-registration states that cuts starting at 0–2 and 12–15
destroy an Engram site, then calls them "those eight arms". It is 3 + 4 = 7.

**Cause.** Arithmetic written in prose rather than derived. `analyze.py` computes
the set from `destroys_site()` and reports 17 of 24 arms leaving both sites, so
the artifacts were always right.

**Direction.** None on any measurement. Left in `PREREG.md` uncorrected, since
amending a pre-registration after seeing results defeats its purpose; corrected
in `RESULTS.md` with a pointer here.

## 2. The k=5 timing measured `needle_init`, not a turn

**What.** The first five-tool timing reported ~1,100 ms per query for both arms,
against `needle-bsky`'s ~180 ms for the same configuration.

**Cause.** `timing_k5.py` wrapped `perf_counter` around *constructing* the agent
as well as routing. A fresh `needle.Needle` per query re-runs `needle_init` —
the engine holds one global session, which `METHODS.md` already records — so
every measurement carried a full init. `needle-bsky/oracle.py` builds an agent
per query too, but times only `route()` via `Decision.latency_ms`.

**How caught.** The number. 1,100 ms for a five-tool turn contradicts a measured
sibling result by 6×, which is not a discrepancy worth explaining away.

**Direction.** Would have added an identical constant to both arms, so the
*ratio* would have survived — but it would have buried the finding that a
five-tool turn is ~250 ms, and any reader reusing the script would have
inherited the bug.

## 3. Wall-clock said the pruned model was slower, and that was an artefact

**What.** With the init removed, the k=5 medians came out control 247 ms against
pruned 354 ms — a 43% *slowdown* from deleting 15% of the layers.

**Cause.** Per-turn wall clock is a function of how many tokens the model emits,
and a degraded model emits different, generally more, output. The confound is in
the metric, not the model.

**Fix.** Report `prefill_tps` and `decode_tps`, which the engine already
reports and which are per-token rates. Those come out +22.3% and +28.7%, at or
above the 17.4% that 23/27 layers predicts.

**Direction.** Taken at face value it would have produced a confidently backwards
claim — that pruning a transformer makes it slower — in a writeup whose whole
deployment argument is about latency.

## 4. `refusal_acc` is `None` on an evalset with no off-topic items

**Third occurrence of the same crash**, after `needle-tool-naming/ERRORS.md` #3.
`needle-bsky`'s `summarize()` returns `None` for a rate over an empty subset, and
every new harness that formats that dict with `:.3f` dies on a smoke set. Fixed
here the same way, by a local `num()` helper.

The recurrence is the finding: the fix keeps being applied at the call site
instead of in `summarize()`, so the next experiment to import it will hit it
again. Left as-is rather than editing a sibling's committed scoring code
mid-experiment, but noted in `METHODS.md`.

**Direction.** None. Smoke sets only; the 62-query runs always have the 8
off-topic items.

## 5. A chained background wait produced nothing and cost a re-run

**What.** `until ! pgrep -f run_sweep; do sleep; done; python3 run_sweep.py
--count 12` was launched as a background command, exceeded the foreground
timeout, was moved to the background, and then produced no output file and left
no process. The 12-layer arm had to be re-run directly.

**Direction.** Wasted about two minutes. Recorded because the failure mode is
silent — an empty output file and no running process reads identically to "not
started yet", and the arm would simply have been missing from the sweep had it
not been checked for explicitly.
