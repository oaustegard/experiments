# ssuf-beta

β* engine for the single-source unsplittable flow (SSUF) cost-preserving
violation constant, per [claude-workspace#165](https://github.com/oaustegard/claude-workspace/issues/165).
See [`RESULTS.md`](RESULTS.md) for findings, scope, and what's not done.

## Repro commands

```bash
cd experiments/ssuf-beta

# LP solver sanity checks (5 independent geometric cases)
python3 test_lp.py

# self-hosted calibration gate (hand-derived instance, NOT the Rybin instance
# — see calibration.py docstring for why)
python3 calibration.py

# parametrized family sweep (exact-rational grid search)
python3 family.py
```

Dependencies: `sympy` (exact rational LP via `sympy.solvers.simplex.lpmin`),
Python 3 stdlib `fractions`. No other packages required.

## Files

- `engine.py` — core β* computation: `Instance`/`Terminal` dataclasses,
  routing enumeration, breakpoint list, exact rational membership-LP
  feasibility (`exact_lp_feasible`, `membership_lp_feasible`,
  `compute_beta_star`).
- `test_lp.py` — solver sanity tests, independent of SSUF semantics.
- `calibration.py` — self-hosted calibration gate (hand-derived instance).
- `family.py` — parametrized family sweep (demands (1,b,1), split (r,q,r)).
- `RESULTS.md` — findings, literature gate, honest scope cuts.
