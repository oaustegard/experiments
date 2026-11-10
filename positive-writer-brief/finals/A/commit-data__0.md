jev-prose-lint: add metrics.json and analyze.py

Add results/metrics.json and the analysis script, analyze.py.

metrics.json holds:
- AUROC for Jev with the register.md entry: 0.808
- AUROC for Jev without the entry: 0.697
- AUROC for the category prior: 0.769
- Settled fraction at 5% error: 24.5%
- Bootstrap 95% CI on Jev-with-entry minus prior: [-0.025, 0.106]
  (2,000 resamples, seed 7)

RESULTS.md now cites each of these numbers with an @evidence marker.
`python3 _lib/evidence.py jev-prose-lint/RESULTS.md` passes.
