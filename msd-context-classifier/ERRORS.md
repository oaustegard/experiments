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
