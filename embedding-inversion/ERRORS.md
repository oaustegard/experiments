# Error log — embedding-inversion

Every error found in this experiment, how it was caught, and which way it
pushed the conclusion. Direction is the column to read first.

| # | what was wrong | how caught | direction | cost |
|---|---|---|---|---|
| 1 | `BekkoEncoder.n_tokens` counted `len(e.ids)` with padding enabled, so every string in a batch reported the batch's padded length and the ≤ 40-token filter dropped the whole pool | first smoke build asserted `len(texts) == total` with 0 rows | none — build failed loudly | one rerun |
| 2 | `run_all.sh keep()` staged `results_*.json` in the same `git add` as the logs; while no results file existed the unmatched glob failed the entire add, so the first three stages' commits were no-ops | the stop-hook check of the build stage's log showed "nothing added to commit" | none on the science; artifacts were committed by hand until `results_float.json` existed, after which the glob matched and the driver's own commits resumed | ~5 manual commits |
| 3 | The driver was launched with `nohup … &` on the assumption it would survive between turns; the container restarted on the next user message (~1 min after the turn ended) and killed it at step 50 | resume hook fired, `pgrep` empty | none; relaunched, build stage skipped via sentinel | 20 min of compute |
| 4 | `train.py` only saved a best-dev checkpoint at epoch end, so a mid-epoch kill lost the whole epoch | consequence of #3 | none | added `.last.pt` resume at the epoch boundary; took effect from the bin1 stages onward (the float zero-step process had loaded the old file) |
| 5 | Corpus dedupe was case-sensitive: one test string is a case variant of a training string | memorization check in the float write-up (`test strings in train: 1`) | negligible — 1 of 1,000, and that item's final hypothesis was not the training string | none |
| 6 | Pre-registered budget of 4–5 h was based on step timings taken on an idle box; the run measured 3.3–3.4 s/step against 2.0 estimated, ~7.5 h for both arms | first epoch's wall clock | none on the conclusion; ETA reported wrong once | — |
| 7 | `Monitor` with `persistent: true` still times out at 30 min; the flag did not change the deadline | "Monitor timed out" events | none; re-armed each time | one tool call per 30 min |
| 8 | Prediction 2 compared "the gap in cosine" across conditions, but the bin1 verifier's cosine is between ±1 sign vectors and the float verifier's is between float vectors — different scales, so the pre-registered clause was not directly testable as written | noticed when the bin1 numbers landed (0.385 vs 0.635 is not a gap, it is two rulers) | none on the conclusion — resolved by re-embedding both arms' final strings in float space post hoc (`results_bin1.json → float_space`), which is the comparison the prediction meant; the post-hoc step is declared as such in RESULTS.md | one extra encode pass |
