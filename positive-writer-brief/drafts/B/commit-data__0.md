jev-prose-lint: add metrics.json and analyze.py, cite results

Add results/metrics.json and the script that produces it, analyze.py.
RESULTS.md now cites every number it reports with an @evidence marker
that points into metrics.json.

metrics.json records:
- AUROC, Jev with the register.md entry: 0.808
- AUROC, Jev without the entry: 0.697
- AUROC, category prior: 0.769
- Settled fraction at 5% error: 24.5%

The 95% bootstrap CI on Jev-with-entry minus prior is [-0.025, 0.106]
(2,000 resamples, seed 7). The interval includes zero, so this data
does not separate Jev-with-entry from the prior.

Check: `python3 _lib/evidence.py jev-prose-lint/RESULTS.md` passes.
