# Errors

1. **Force-keeping every user-role message filled the budget.** The first
   selection rule kept all `type: user` text before ranking. Skill bodies,
   stop-hook feedback and peer messages are user-role turns (`isMeta: true`),
   and on a 181-chunk live session they took the whole 20k budget; the chunk
   holding the answer (the Sniff Test benchmark table) was dropped. Caught by
   running `preview.py` on this session while the first eval pass ran.
   Direction: it pushed the Jev arms down. Both Jev arms were re-run after the
   fix (`isMeta` turns are scored like tool results; real user messages are
   clipped to 1,500 chars and capped at a quarter of the budget). The v1 run is
   kept in `data/results_v1/`.

2. **The fact regexes were too literal.** Labellers wrote regexes from the
   transcript's own formatting (`3868`, `3 failed, 5 passed`), and correct
   deliverables that wrote "3.9 s" or "5 passed, 3 failed" scored zero. Caught
   when the full-transcript arm scored 0/3 on a deliverable that read correct.
   Direction: it pushed every arm down, unevenly. Fixed with a blind
   adjudicator over regex misses, validated in the same batches (TPR 100% on 40
   hits, TNR 95% on 40 wrong-task deliverables). Both scorings are reported.

3. **Usage tokens read 0 on some executor runs.** `claude -p --output-format
   json` top-level `usage` covers only the last model call; multi-call runs
   need `modelUsage`. Caught on the first analysis pass. Direction: none on the
   conclusions; token columns use the passed-context estimate, and the runner
   now sums `modelUsage`.

4. **Rediscovered a METHODS.md entry.** The gateway throttled at 4 to 12
   concurrent calls (HTTP 429, code 2003); METHODS.md already said to start
   gateway concurrency at 2. I had not grepped it before writing the filter.
   Direction: none on the results (failed calls were re-run); cost two re-runs.

5. **Misread a WAF block as a billing error.** HTTP 402 "Payment error from
   model using BYOK" first read as a transient billing blip, and the retry loop
   backed off on it. The body is TypeSafe's Cloudflare "Sorry, you have been
   blocked" page, deterministic for the same request bytes. Caught when the
   same four tasks failed identically on a second run. Fixed by detecting the
   block and bisecting the window.
