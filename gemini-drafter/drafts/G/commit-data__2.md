jev-prose-lint: add analyze.py, metrics.json, and @evidence citations

Adds `jev-prose-lint/analyze.py` and `jev-prose-lint/results/metrics.json`, and updates `jev-prose-lint/RESULTS.md` so each number cites `metrics.json` with an `@evidence` marker.

`metrics.json` records AUROC for Jev with the `register.md` entry (0.808), Jev without it (0.697), and the category prior (0.769). The bootstrap 95% CI on Jev-with-entry minus the category prior is [-0.025, 0.106] across 2,000 resamples with seed 7. The settled fraction at 5% error is 24.5%.

`python3 _lib/evidence.py jev-prose-lint/RESULTS.md` passes.
