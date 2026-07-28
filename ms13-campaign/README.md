# ms13-campaign — kill Morell–Skutella Conjecture 1.3 (issue #169)

Campaign to refute (or structurally explain) MS Conjecture 1.3 (STVZ
arXiv:2510.21287 numbering — see the collision table in WORKLOG.md): every
acyclic SSUF network + fractional flow x admits an unsplittable routing with
`x(a) − d_max ≤ f(a) ≤ x(a) + d_max` on all arcs.

Lineage: [#163](https://github.com/oaustegard/claude-workspace/issues/163)
(Woodall τ=3, negative, CEGAR verifier pattern) →
[#165](https://github.com/oaustegard/claude-workspace/issues/165) (SSUF β*
hunt, [PR #167](https://github.com/oaustegard/claude-workspace/pull/167)) →
[#166](https://github.com/oaustegard/claude-workspace/issues/166)/[PR #168](https://github.com/oaustegard/claude-workspace/pull/168)
(discrepancy records) → #169 (this campaign). PR #170 was a parallel
bootstrap session; its ledger content is merged into NOGOS.md (NG-1 note,
NG-5).

Two kill-paths (issue #169): **A** — direct implication-gadget counterexample
(floors force presence, presence breaks ceilings; see NOGOS.md Lemma 1 for
why ceilings alone cannot work); **B** — β* > 2 on a nonplanar acyclic
instance, refuting 1.3 through STVZ Theorem 1.6's contrapositive.

State lives in-repo: **WORKLOG.md** (session log, gates) and **NOGOS.md**
(killed families + killing arguments — read both before designing anything).

## Layout

| File | What |
|---|---|
| `core.py` | Exact-Fraction SSUF machinery: instances, routings, two-sided violation, counterexample verifier with certificate tables |
| `beta_star.py` | Breakpoint-LP β* engine (sup convention; corrects PR #167's off-by-one) |
| `rybin.py` / `test_rybin.py` | Constraint reconstruction of the Rybin instance from its verified anchors |
| `sweep.py` | Float screening: Dirichlet + polish maximization of t(x) over random layered DAGs (decisions always re-verified exactly) |
| `coverage.py` | Exact/MILP test for "does ANY counterexample x exist for fixed (G,d)" |
| `ladder.py` | Kill-path B mechanism ladder: conflict-graph families (C3/C5/C7/K4/K5), β* trend |
| `tests/test_calibration.py` | Verifier gate (issue gate 2): triangle hand-derivation, brute-force LP agreement, malformed-instance guards |
| `NOGOS.md` | The ledger — strategic product of the campaign |
| `WORKLOG.md` | Session state, literature-gate findings, naming table |

## Repro

```bash
cd experiments/ms13-campaign
python3 tests/test_calibration.py   # verifier gate — must be green before any search
python3 test_rybin.py               # Rybin anchor gate
python3 sweep.py                    # small screening demo
python3 ladder.py                   # kill-path B ladder run
```

Discipline (issue gate 3): floats only screen; Fractions decide; any positive
hit needs dual independent code paths + exhaustive routing table before the
word "counterexample" appears anywhere.
