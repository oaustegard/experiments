# discrepancy — Komlós + Beck–Fiala lower-bound records (issue #166)

Certified lower-bound records and small-parameter exact values for two
discrepancy conjectures. Companion to the Woodall τ=3 search (#163); same
max-min skeleton: adversary space (matrices / set systems), finite response
set (sign vectors / colorings), heuristic screening, **exact certification
for anything claimed**.

Literature gate findings (mandated first step) are logged on
[issue #166](https://github.com/oaustegard/claude-workspace/issues/166).
Headlines: Kunisky (arXiv:2111.02974) holds the Komlós record K ≥ 1+√2
asymptotically, so the honest Target-A deliverable is *per-size* K(n)
records; Beck–Fiala was resolved asymptotically for t ≥ log²n in 2025
(arXiv:2508.01937 + follow-up), so the open niche is *exact small-t values*
D(t, n).

## Files

| File | Role |
|---|---|
| `komlos.py` | Exact ℤ[√2] + Fraction discrepancy engines, Kunisky tree matrices, smoothed max-min search, rationalizer |
| `beck_fiala.py` | Exact set-system discrepancy, SAT disc≥k certificates, CEGAR ∃-system search |
| `run_komlos.py` | Target A pipeline (stage 1 + baseline search) → `komlos_results.json` |
| `deep_search.py` | Heavy-restart Komlós pass; updates `komlos_results.json` in place |
| `polish_small.py` | Earlier fine-rationalization pass for n=3..7 (superseded by deep_search) |
| `tree_dp.py` | Optimal Kunisky-style tree value δ*(n) for all n, by DP |
| `run_beck_fiala.py` | Target B pipeline → `bf_results.json` |
| `verify_certificates.py` | Independent re-verification of every claimed record (second code path) |
| `plot_growth.py` | D(t) growth vs √t / 2t−2 curves → `growth.png` |
| `tests/test_calibration.py` | Calibration gates G1–G9 (all must pass before records are claimed) |
| `RESULTS.md` | Findings writeup |

## Reproduce

```bash
pip install numpy python-sat matplotlib   # container has none of these by default

python3 tests/test_calibration.py         # gates: identity=1, Kunisky n=2=√2,
                                          # engine==brute force, SAT==enumeration,
                                          # Fano disc, D(2,n≤5)=2 via CEGAR

python3 run_komlos.py 16 400 6            # n≤16, 400 ascent iters, 6 seeds
python3 run_beck_fiala.py 2,3,4 10        # t∈{2,3,4}, n≤10
python3 plot_growth.py                    # growth.png from bf_results.json
```

Everything certified is decided in exact arithmetic: integer pairs (a+b√2)
for the Kunisky family, `Fraction` for rationalized search records, integer
enumeration + SAT UNSAT for set systems. Floats only ever *order* candidates.

## Method notes

- **Komlós per-size records.** disc(V) enumerates all 2^{n−1} sign classes.
  The outer max uses softmin/softmax-smoothed subgradient ascent with
  annealed sharpness, projected onto the product of unit ℓ₂ balls, from
  Kunisky / Hadamard / identity / random seeds. Candidates beating the
  Kunisky-family baseline at their size get rationalized (denominator ≤ 64,
  columns exactly re-normalized) and re-certified exactly.
- **Δ = δ on the Kunisky family** (one-line proof, verified in Stage 1):
  every row of Â^{T_k} has ℓ₁-norm exactly δ, so the all-ones coloring
  achieves ‖Â·1‖∞ ≤ δ, while unsatisfiability (Prop 2.3) forces ≥ δ. His
  published finite lower bounds are therefore already the exact values.
- **Beck–Fiala WLOG.** A degree-≤t system on n elements has ≤ t·n nonempty
  sets (with multiplicity), and duplicate/empty sets never change disc — so
  fixing m = t·n rows makes the CEGAR verdict exact for D(t,n), not just a
  bound at one m.
- **CEGAR loop.** SAT variables = incidence matrix (degree ≤ t per column,
  lex-ordered rows for symmetry breaking); counterexample colorings add
  "some row must break x by ≥ k" disjunctions (conditional sequential-counter
  cardinality encodings, fresh auxiliaries per constraint). Every witness is
  re-checked by exhaustive enumeration AND an independent SAT UNSAT
  certificate of the (k−1)-bounded coloring formula.
