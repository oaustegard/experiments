jev-prose-lint: add metrics.json and analyze.py

Add results/metrics.json and analyze.py, the script that
produces it.

metrics.json holds:
- AUROC for Jev with the register.md entry: 0.808
- AUROC for Jev without the entry: 0.697
- AUROC for the category prior: 0.769
- Fraction settled at 5% error: 24.5%

The bootstrap 95% CI on Jev-with-entry minus the prior is
[-0.025, 0.106] (2,000 resamples, seed 7), so the interval
includes zero.

RESULTS.md now cites each of these numbers with an @evidence
marker, and `python3 _lib/evidence.py jev-prose-lint/RESULTS.md`
passes.
