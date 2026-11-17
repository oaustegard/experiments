jev-prose-lint: add analyze.py, metrics.json, and @evidence citations

Add `jev-prose-lint/analyze.py` and `jev-prose-lint/results/metrics.json`, and cite each number in `jev-prose-lint/RESULTS.md` with an `@evidence` marker so that `python3 _lib/evidence.py jev-prose-lint/RESULTS.md` passes.

`metrics.json` records the AUROC for Jev with the `register.md` entry (0.808), Jev without the entry (0.697), and the category prior (0.769). It also records the settled fraction at 5% error (24.5%) and the bootstrap 95% CI on Jev-with-entry minus prior ([-0.025, 0.106] from 2,000 resamples with seed 7).
