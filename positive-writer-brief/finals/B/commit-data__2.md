jev-prose-lint: add metrics.json, analyze.py; cite from RESULTS.md

Add results/metrics.json and the analysis script analyze.py, so every
number in RESULTS.md traces back to a file.

metrics.json holds:
- AUROC, Jev with the register.md entry: 0.808
- AUROC, Jev without the entry: 0.697
- AUROC, category prior: 0.769
- Settled fraction at 5% error: 24.5%

The bootstrap 95% CI on (Jev with entry - prior) is [-0.025, 0.106],
from 2,000 resamples with seed 7. The CI includes zero, so at this
sample size Jev with the entry does not separate from the category
prior.

RESULTS.md now cites each number with an @evidence marker.
`python3 _lib/evidence.py jev-prose-lint/RESULTS.md` passes.
