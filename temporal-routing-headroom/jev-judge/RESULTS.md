# RESULTS: Jev as a completion judge for a model ladder

Question: can Jev (the Muninn `decide` model) sit on the decision path of a Haiku → Sonnet → Opus
ladder, judging a worker's finished attempt? Motivation: Hightouch's Jev-first hybrid matched
Opus accuracy at 0.23–0.55x cost because Jev's errors sat near 50% (Muninn 02050861).

## Answer

Not yet measured. The decision rule and the dev/test split below were written before any test-split
attempt was judged. Pilot so far: 14 Haiku 5.5 attempts, 10 pass the held-out suite, 4 fail, and the
visible tests were green on all 14, so "visible tests pass" carries no signal on this task family.
This section is rewritten when the test split has been read.

## Findings

1. Visible tests are green on every attempt in the pilot (14/14), including all 4 held-out failures.
   Workers stop when the visible suite passes, so that signal is constant by construction
   (Round 1, pilot; `data/attempts/`).

## Method

**Data.** Every worker attempt on the 14 `temporal-routing-headroom` seeded-bug tasks, with issue
text, final diff, worker summary, visible-test output, and the held-out suite result as ground truth.
Arms: Haiku 5.5, Sonnet 5.5 at low effort, Opus 5.5 at high effort, two replicates each, run as
headless `claude -p` children (effort settable, one background job), plus the 14-attempt Haiku pilot
run through the Agent tool (larger prefix, different harness; kept as its own arm, not pooled with
the headless Haiku arm).

**Split.** By task, never by attempt, so replicates and arms of one task do not straddle it.
`split.json`: sha1 of `jev-judge-split:<task>` ascending, first 4 tasks dev, remaining 10 test.
Prompts are fixed on dev only; no few-shot examples from test.

**Judge questions** (Jev via `decide`, one binary question each; inputs are issue text, diff,
worker summary and visible-test output, never the held-out result):
Q1 "The task is fully solved." Q2 "The summary mentions a defect, case or symptom left unaddressed."
Q3 "The change addresses every site/symptom in the issue text." Q4 "The visible tests exercise the
symptom described in the issue." Q2 is scored inverted against the pass label.

**Metrics.** Per question: TPR, TNR, AUROC, ECE and a 10-bin reliability diagram, each with a
bootstrap 95% CI resampled by task. Error placement: the fraction of misjudgements (P on the wrong
side of 0.5) with P in 0.3–0.9. Simulated ladder: accept if P > hi, escalate if P < lo, mid-band gets
an informed retry; sweep (lo, hi) on dev, report test. Costs from measured per-attempt usage.
Combination: logistic on [tests_green, P_Q1..Q4]; reported as degenerate if tests_green is constant.

**Decision rule (fixed before the test split was judged).** The primary question is the one with the
best dev AUROC. Evaluated on test:

- **GO** if at least 70% of that question's test-split misjudgements have P in 0.3–0.9, **and** the
  simulated ladder at the dev-chosen (lo, hi) solves at least as many test tasks as the
  visible-tests-only ladder, costs less than always-Opus, and stays within 5 percentage points of the
  held-out-gated ladder's solved rate.
- **NO-GO** if most misjudgements are confident (P outside 0.3–0.9, with the CI excluding 70%).
- **PARTIAL** otherwise; name which question carries signal.
- **INCONCLUSIVE**, whatever the point estimates say, if the test split holds fewer than 60 attempts
  or fewer than 20 held-out failures.

**Deviation from the brief, and why.** The brief's second GO condition was "beats the test-gated
ladder on cost at equal solved rate". The existing ladder is gated by the held-out suite, which the
orchestrator runs at no cost and which is exact by construction, so no judge can beat it on cost.
Jev's use case is work with no held-out suite. The gate it replaces is therefore the visible-tests
gate (which accepts every attempt here), and the held-out-gated ladder is the ceiling it is compared
against, not a baseline it must beat.

## Log

### Round 1, 2026-11-08: pilot

Haiku 5.5 rung 1 via the Agent tool, one replicate, 14 tasks, artifacts captured per attempt
(`data/attempts/`). 10 pass, 4 fail (`cron_next`, `interval_merge`, `lru_ttl`, `wrap_text`). Peak
context 80–86K tokens per attempt, almost all the ~55K spawn prefix; none reached 100K. The
`cron_next` worker named the AND-versus-OR second defect in its summary and left it. Harness note:
`git init` inside a worktree makes `grade_agentic` flag `tampered_tests` on every attempt, so diffs
are taken against the pristine task copy instead.
