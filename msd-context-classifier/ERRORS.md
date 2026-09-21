# ERRORS.md — msd-context-classifier

1. **Coverage-at-error-budget was defined so that it could not fail on the
   distribution that mattered.** `train.py` picks the confidence threshold on
   the page dev split and applies it to the question set. On pages the realized
   error at the chosen coverage is 0.02–0.07, as intended. On questions the same
   thresholds cover 21–90% of inputs with realized error 0.59–0.77, so the
   "coverage" column for questions is a number with no meaning. Caught when
   ettin-32m reported 0.90 coverage on questions next to 0.21 accuracy. Fixed in
   the writeup by reporting realized error beside coverage and scoring P6 as
   wrong; not fixed in code, because the honest fix is a dev split drawn from
   the deployment distribution, which is the experiment's own conclusion.
2. **The corpus worker launched its fetch under `nohup` inside the subagent**,
   the shape `long-running-workflow-protocol` names as the one that dies when the
   container pauses. Caught at its first hand-back (73 pages in); killed and
   relaunched from the parent as a tracked background job, resumed from the
   checkpoint, no pages lost. The worker then re-woke itself on a Monitor every
   few minutes to report progress nobody needed until told to stop.
3. **Unweighted cross-entropy over a 14:1 class imbalance never learned the
   small classes** (`platform_products` 0.00 F1 in all four fine-tuned arms,
   `assay_kits` 0.62–0.71), which is most of the 18-point gap to the probe on
   pages. The recipe was fixed in PLAN.md before the run and is reported as run;
   the class-weighted check arm is in the results table. Direction: the
   fine-tune-versus-probe comparison on pages overstates the probe's margin;
   the question-transfer finding does not depend on it.
4. **`platform_products` kept 20 of 94 sampled pages** because instrument pages
   are spec tables and images with under 60 words of prose. Its 3-row page test
   cell is reported with its n and carries no conclusion on its own.
5. **The whole-term scorer's "worst terms" list read the wrong counter.** After
   the loop over `domain` and `control`, the loop variable pointed at the
   control counters, so every domain term's per-type recovery was looked up in
   a table that did not contain it and printed 0.0. V-PLEX, U-PLEX and R-PLEX
   "scored zero in every arm" for two hours of this session; a spot check that
   masked one occurrence by hand recovered it exactly, which is what exposed the
   bug. The aggregate numbers were never affected. I first blamed the eight-
   targets-per-pass batching (co-listed family names masked together) and
   re-scored every model one target per pass: the aggregates moved by at most
   two points (dapt 0.678 → 0.688, dapt+term 0.817 → 0.832), so batching was a
   small real effect and the zero was the reporting bug. Fixed; per-type
   recovery is now written to the JSON so the claim can be checked without a
   rerun.
6. **Ordinary control words rose 45 points under adaptation** (0.39 → 0.84),
   which PLAN-vocab.md V1 predicted would stay under 5. Held-out product pages
   share their template with training pages, so the "domain-term" gain is
   partly the template. Reported as such; a split by page template does not
   exist in this corpus and would be the fix.
7. **The domain-term list contains extraction junk**: `XPlease`, `InSign`,
   `NowRegistration` are navigation text concatenated by the HTML extractor
   ("Sign In", "Register Now", "…X Please"). They pass the regex (two capitals,
   nine-plus letters) and are among the most frequent "terms". They inflate the
   term count and are recovered at 0.0 by every model. Left in the audit numbers
   (three of 1,624 types); excluded by name from any per-type claim.
8. **`pkill -f` matched its own shell.** Stopping the sequential crawl with
   `pkill -f "python3 fetch_and_extract.py"` killed the Bash tool call that
   issued it (exit 144) along with the target. Kill by PID.
