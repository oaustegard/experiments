# Gate run — validating the optimizing-skills v0.2.0 patch against its own gate

**Date:** 2026-05-29
**Rule under test:** the two edits proposed in
[`proposed-patch-optimizing-skills.md`](proposed-patch-optimizing-skills.md).
Per the patch's own caveat, the edits must clear the gate before shipping to
`oaustegard/claude-skills`. This is that run.

## Method

The triggering failure is a **scoring-rule** failure (a collapsed pass/fail
masked a real win), not a model-behavior failure. So the check set is a set of
skill-revision **decision scenarios** with known-correct verdicts, evaluated by
hand — the skill permits "evaluate by hand for small sets," and dispatching to
Haiku cannot test a scoring rule.

- `best` rule = v0.1.0: "Score hard pass/fail per task. Accept only if candidate
  strictly beats best (more tasks pass)." A multi-criterion task collapses to one
  pass/fail (task passes iff all criteria pass).
- `candidate` rule = v0.2.0: accept iff the **triggering-failure criterion**
  strictly improves **and no regression-guard criterion gets worse**.

## Check set (verdict = correct accept/reject is known)

| # | Scenario | Correct verdict |
|---|---|---|
| T1 | **Triggering failure.** Primary criterion improves big (hallucination 60%→0%); secondary (length) flat-bad in both arms. | ACCEPT |
| T2 | No-op edit. No criterion improves. | REJECT |
| T3 | Primary improves (hallu 60%→0%) but a guard regresses (schema 100%→40%). | REJECT |
| T4 | Primary improves and secondary also improves. | ACCEPT |

## Scores

| | T1 | T2 | T3 | T4 | Correct |
|---|---|---|---|---|---:|
| **best rule** (collapse) | reject ✗ | reject ✓ | reject ✓ | accept ✓ | **3/4** |
| **candidate rule** (per-criterion) | accept ✓ | reject ✓ | reject ✓ | accept ✓ | **4/4** |

Candidate strictly beats best (4/4 vs 3/4); the sole difference is **T1, the
triggering failure**. No regression: T2–T4 stay correct. T3 confirms the new
rule still rejects a primary-improving edit that worsens a guard — edit 1 does
not make the gate "accept anything that improves the headline number."

## Verdict: SHIP

Edit 1 (per-criterion scoring) clears its own gate. Edit 2 (≥2 author samples
for Agent-compiled artifacts) is a methodology guard — it cannot produce a wrong
accept/reject, so it is not exercised by this decision-logic check set; it is
validated by the observed author variance in the retro (95%→0% vs 60%→0% across
two author pairs on the same edit). Both ship as optimizing-skills v0.2.0.

Caveat: hand-evaluated, n=4 constructed scenarios. The scenarios are the minimal
set that distinguishes the two rules (one triggering-failure case + three
regression guards covering no-op, guard-regression, and clean-win). Not a stress
test of pathological multi-criterion cases.
