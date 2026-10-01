# Errors

1. **Jev reported as free.** The first grade table summed only `claude -p` costs, so every
   `jev` row read `$0`, and PREDICTIONS.md #4 said Jev would cost under a cent per episode.
   Caught while writing RESULTS.md: `jev_tokens` in the step logs was 1.06M for the 96-chunk
   episode, $0.044 at TypeSafe's $0.042/Mtok. `grade.py` now adds a Jev $ column. Direction:
   overstated the cost gap; the corrected gap is still ~175×.
2. **`both` shares one cap and lets the notes file win it.** `relay.py` sizes Jev's store as
   `CAP - tokens(state.md)` before the subagent runs, then cuts `state.md` to whatever room is
   left. Once the subagent's notes grew, Jev's store was the side that shrank, and it evicted
   verbatim needles to make room for commentary. Not caught before the run; diagnosed from
   `work/both__unlabeled__96__0/progress.json` after recall came back at 0.755. Direction:
   makes `both` look worse than a design that reserves the store's share would. RESULTS.md
   reports the number with this cause attached rather than as a verdict on combining the two.
3. **First-draft figures in RESULTS.md.** The draft said "1,036 needles across five episodes"
   (Jev ran four: 835), "under two minutes" for the 96-chunk Jev episode (it took 3), and that
   the stores hit the cap "by chunk 73" (the first cut is at step index 73, chunk 74). Caught
   by checking each figure against `run.log` and the step logs before committing.
4. **Recall rounded twice.** `grade.py` rounded the mean to four places (0.7545) and the table
   then printed it to three, giving 0.754 for 375/497 = 0.75452. `recheck.py` caught it against
   the per-episode value; the mean is now kept unrounded and the table reads 0.755.
