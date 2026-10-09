# Errors

What was wrong, how it was caught, and which way it pushed the result.

1. **My own concurrency memory was wrong twice.** Before the run I stored that
   Agent-tool fan-out had no 20 cap (from Oskar's "fan out allows 10 (or
   more)"). The 21st launch was refused: "You can run 20 subagents at once."
   Memory corrected the same session. No effect on results; it set the pacing.

2. **Wrong environment.** Memories describe two machines: the claude.ai chat
   container (`/home/claude`, `/mnt/project`) and Claude Code on the Web, where
   the audit ran. A path missing here proves nothing about the other machine.
   Caught when the pilot extracted bash-timeout claims about the chat container.
   Fix: extraction labels `env`, 131 claude.ai claims were set aside, and the
   recheck prompt names the trap. Two bash-timeout claims still slipped through
   as "outdated" and were excluded by hand before applying. Direction: without
   the fix, staleness is overstated.

3. **`probe` field did not match the command run.** In the pilot one verdict's
   evidence came from `oaustegard/sage` while its recorded probe named
   `oaustegard/remex`. Caught by re-running the probe. Fix: the prompt requires
   the probe copied character for character. The evidence itself was right.

4. **Stale versus contradicted.** The pilot marked "Sage has no write path yet"
   as contradicted; it was true when written and the write path came later.
   Fix: the prompt defines contradicted as false on the date written. The
   distinction only sets the `audit-contradicted` tag; both get corrected.

5. **First-pass false stale.** The recheck overturned 6 of 674, including
   "claude-workspace issue #163 is closed" marked stale when the issue is
   closed. Direction: without the recheck, about 1% more claims would be
   miscorrected.

6. **Apply script rebuilt its plan.** I removed the two env-mismatch memories
   from `apply_plan.json`, but `--apply` regenerates the plan, so the first
   launch would have superseded them. Caught 12 supersedes in; neither had been
   touched. Fix: an explicit exclusion list in the script.

7. **Rubber-stamp check.** A recheck chunk returned 25 of 25 confirmed on 18
   tool calls. I compared its probes to the first pass: 0 of 25 identical,
   evidence real (mostly one `gh api` call per issue). Not an error, recorded
   because the 99% confirmation rate invites the suspicion.
