jev-prose-lint: add results/metrics.json and analyze.py

analyze.py computes the metrics and writes them to results/metrics.json:

- AUROC for Jev with the register.md entry: 0.808
- AUROC for Jev without the entry: 0.697
- AUROC for the category prior: 0.769
- Settled fraction at 5% error: 24.5%

The bootstrap 95% CI on Jev-with-entry minus the prior is
[-0.025, 0.106], from 2,000 resamples with seed 7. The interval
includes zero.

RESULTS.md now cites each of these numbers with an @evidence marker,
and `python3 _lib/evidence.py jev-prose-lint/RESULTS.md` passes.
