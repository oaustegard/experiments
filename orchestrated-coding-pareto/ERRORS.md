# Errors — what was wrong, how it was caught, which way it pushed

Convention per repo README: the base rate of caught errors is the most useful
calibration number about a body of work.

1. **Tier-1 difficulty misjudged (design error).** The original 8-task bank was
   built on the theory that edge-case-dense specs separate model tiers. Oskar asked
   whether the tasks were hard enough; early grading of the 6 haiku solutions then
   on disk showed 6/6 — the ceiling was already visible mid-run. Caught by grading
   early instead of waiting for all arms. Direction: would have produced a null
   with no diagnostic value (one tier, "all pass") rather than the calibrated
   three-tier saturation result. Fix: tiers 2 and 3, disclosed as a mid-run
   extension in RESULTS.md.
2. **`text_table` hidden test asserted a wrong expectation** (`test_top_alignment_of_short_cells`
   assumed column width 4 where the spec gives 1). Caught by validating the suite
   against the reference implementation before any model saw the task. Direction:
   would have failed *every* arm on a correct solution — deflating all pass rates
   equally, most likely triggering spurious "orchestration rounds" over a phantom
   failure.
3. **`glob_match` reference mishandled trailing `/**`** (matched `"a"` against
   `"a/**"` where the spec says it must not). Caught in a pre-validation re-read;
   the fix also forced the spec sentence to be rewritten from two self-correcting
   clauses into a plain rule. Direction: reference and tests would have agreed with
   each other and disagreed with the spec — workers implementing the spec correctly
   would have been marked wrong.
4. **`cron_next` reference contained a mangled dead expression** (`... if False else ...`)
   from an editing slip. Caught on re-read before validation; behavior happened to
   be correct but the code was unreadable. Direction: none on results; reduced
   trust in the artifact had it shipped.
5. **`interval_merge` reference carries two redundant dead sub-branches** (noted,
   left in place — behavior verified correct by the suite). Direction: none.

Not errors but near-misses worth recording: the retry-arm design assumed haiku
failures would exist (they did not — arms went vacuous rather than wrong), and the
per-arm token metering depended on running workflows sequentially; launching them
concurrently would have silently confounded `budget.spent()` deltas across arms.
