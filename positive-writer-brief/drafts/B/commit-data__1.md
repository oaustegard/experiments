jev-prose-lint: add metrics.json and analyze.py, cite them in RESULTS

Add results/metrics.json and the script that produces it, analyze.py,
so every number in RESULTS.md traces to a checked-in file.

metrics.json holds:
- AUROC, Jev with the register.md entry: 0.808
- AUROC, Jev without the entry: 0.697
- AUROC, category prior: 0.769
- Settled fraction at 5% error: 24.5%
- Bootstrap 95% CI on (Jev with entry - prior): [-0.025, 0.106],
  2,000 resamples, seed 7

The CI includes zero. At this sample size, Jev with the entry has not
been shown to beat the category prior.

RESULTS.md cites each number with an @evidence marker.
`python3 _lib/evidence.py jev-prose-lint/RESULTS.md` passes.
