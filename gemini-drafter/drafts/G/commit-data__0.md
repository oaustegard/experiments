jev-prose-lint: add analyze.py, metrics.json, and @evidence citations

Add `jev-prose-lint/analyze.py` and `jev-prose-lint/results/metrics.json` so `jev-prose-lint/RESULTS.md` cites each reported number with an `@evidence` marker. `python3 _lib/evidence.py jev-prose-lint/RESULTS.md` passes.

`metrics.json` records an AUROC of 0.808 for Jev with the `register.md` entry, 0.697 without it, and 0.769 for the category prior. The settled fraction at 5% error is 24.5%. The bootstrap 95% CI on Jev-with-entry minus the category prior is [-0.025, 0.106] across 2,000 resamples with seed 7.
