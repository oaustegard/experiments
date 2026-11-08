# RESULTS: Jev as a completion judge for a model ladder

Question: can Jev (the Muninn `decide` model) sit on the decision path of a Haiku → Sonnet → Opus
ladder, judging a worker's finished attempt? Motivation: Hightouch's Jev-first hybrid matched
Opus accuracy at 0.23–0.55x cost because Jev's errors sat near 50% (Muninn 02050861).

## Answer

Jev is a small model, called through Muninn's `decide` tool, that returns a probability for a yes/no question. Can it judge a worker's finished attempt well enough to decide when to escalate up a Haiku →
Sonnet → Opus ladder? Not on this family. On 106 test attempts (10 tasks, 38 failing the held-out tests), its primary question Q3 ("the change addresses every site in the issue") ranks attempts with
AUROC 0.75 <!-- @evidence[[results/v2_test.json#questions.Q3.auroc]] -->, but within one task it rates failing
attempts as high as passing ones (0.45 <!-- @evidence[[results/v2_test.json#questions.Q3.within_task_auroc]] -->;
AUROC is the chance a random pass outranks a random fail, 0.5 being chance). The pre-registered rule says
PARTIAL, because its NO-GO trigger, confident errors, was not met: 84% of Q3's misjudgements fall in 0.3–0.9 <!-- @evidence[[results/v2_test.json#questions.Q3.frac_wrong_midband]] -->.
I call it NO-GO anyway: that share is compression, as 66% of all its scores sit there <!-- @evidence[[results/extras_test.json#Q3_band_base_rate]] -->.
The simulated ladder solves 0.82 <!-- @evidence[[results/v2_ladder_test.json#models.headless/jev_mid.policies.jev_ladder.0]] -->
against 0.87 for always-Opus, at 0.91 of its cost (Finding 2). A diff-size count beats every question within a task (Finding 3). Q1 and Q3 carry only a task-difficulty prior, the model advisor's job. A judge that can run
the code, or failures not defined by sites patched, would change this.

## Findings

Ordered by weight, each with its round and the number that carries it.

1. Jev has no attempt-level signal on this family. Within-task AUROC on test is 0.25 (Q1), 0.35 (Q2), 0.45 (Q3), 0.35 (Q4), all at or below chance; the pooled AUROC of 0.57–0.77 is task difficulty (Round 4, `results/v2_test.json`).
2. The simulated test ladder at the dev-chosen thresholds solves 0.82 at 0.91x the cost of always-Opus (bootstrap 0.57–1.21), which solves 0.87. Visible-gated solves 0.80 at 0.04x, and the free held-out-gated ladder 0.91 at 0.33x, so Jev's ladder is 9 points under the free gate and no cheaper than Opus (Round 4, `results/v2_ladder_test.json`).
3. A diff-size count beats every question within a task: larger diffs pass, AUROC 0.07 for smaller ones, i.e. 0.93 for larger <!-- @evidence[[results/extras_test.json#diff_size_within_task_auroc]] -->. Failures here are one-site fixes. Exploratory, run after the test split was read (Round 4).
4. Q1, Q3 and Q4 answer yes to nearly everything at 0.5 (TNR 0.16, 0.00, 0.00); Q2 is the only question that splits (TPR 0.82, TNR 0.58). Jev's probabilities for Q1 sit in 0.3–0.9 on 95% of attempts, so a 0.3–0.9 band cannot isolate its errors (Round 4).
5. The information is present and unused: on `lru_ttl` the failing diffs leave `__len__` unpurged in plain view of a 30-line context, and Jev scored them above the passing ones that add the `_purge()` call (Round 3, dev).
6. Wider diff context raised pooled dev AUROC (Q1 0.68 to 0.79, Q3 0.62 to 0.79) and left within-task AUROC at 0.4, so it taught Jev which tasks are hard, not which attempts are right (Round 3, `results/v1_dev.json`, `results/v2_dev.json`).
7. Combining Q1–Q4 by logistic regression does not help: leave-one-task-out AUROC 0.44 <!-- @evidence[[results/extras_test.json#logit_Q1-4_loto_auroc]] -->. `tests_green` is constant (14/14 pilot, 144/144 headless), so "beyond visible tests" cannot be fitted (Round 4).
8. The dev-chosen thresholds (lo 0.9, hi 0.95) escalate almost everything, because Jev's scores cluster below 0.9; the ladder then costs nearly as much as always-Opus and solves less. No (lo, hi) on dev came within 5 points of the held-out-gated ladder (Round 3, `results/v2_ladder_dev.json`).
9. Headless `claude -p` workers cost $0.002 (Haiku), $0.02 (Sonnet low) and $0.06 (Opus high) per attempt, with peak context of 11–15K tokens. The Agent-tool spawn adds a ~55K-token prefix: priced that way the same ladder is $0.30 against Opus's $0.34 (Round 2, Round 4).

## Method

**Data.** Every worker attempt on the 14 `temporal-routing-headroom` seeded-bug tasks, with issue
text, final diff, worker summary, visible-test output, and the held-out suite result as ground truth.
Arms: Haiku 5.5, Sonnet 5.5 at low effort, Opus 5.5 at high effort, run as
headless `claude -p` children (effort settable, one background job): 2 replicates on all 14 tasks, then
replicates 3–6 on the five failure-prone tasks (144 attempts, $4.58), plus the 14-attempt Haiku pilot
run through the Agent tool (larger prefix, different harness; kept as its own arm, not pooled with
the headless Haiku arm).

**Split.** By task, never by attempt, so replicates and arms of one task do not straddle it.
`split.json`, amended in Round 2 before any judging: tasks with at least one held-out failure in any
attempt (5 tasks) and tasks with none (9) are ordered separately by sha1 of `jev-judge-split:<task>`;
dev is the first 2 of each stratum (`interval_merge`, `lru_ttl`, `parse_range`, `toposort_lex`),
test is the other 10. Prompts are fixed on dev only; no few-shot examples from test. The ladder
simulation draws attempts from all replicates of a task and weights tasks equally, so the extra replicates
on the five failure-prone tasks sharpen their estimates without changing the task mix.

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

**Ladder simulation (fixed before the test split was judged).** Rungs: Haiku 5.5, Sonnet 5.5 low, Opus 5.5
high. At each rung one attempt is drawn uniformly from that arm's replicates of the task and judged by the
primary question (P = P(pass)). Accept if P > hi; escalate if P < lo; in between, one retry at the same
rung (a fresh attempt, not an informed one, so conservative against the informed retry that rescued 3/3
in `3e850450`), accepted unless its P < lo, in which case escalate. The last rung accepts whatever it
produced. Expected solved rate and cost are computed exactly by averaging over draws, tasks weighted
equally. Baselines: always-Haiku, always-Sonnet, always-Opus; visible-gated (accepts the first attempt,
because visible tests were green on every attempt); held-out-gated (escalates on held-out failure, free
and exact). (lo, hi) is swept on a 0.05 grid on dev: among points with solved rate within 5 points of the
held-out-gated ladder, the cheapest; if none qualifies, the highest solved rate. Costs: measured
`claude -p` cost per attempt, plus a second model that adds the ~55K-token Claude Code prefix a fresh
Agent-tool spawn writes to cache (`3e850450`, `harness/subagent_cost.py` prices) at each model's cache-write
rate. Jev is priced at an assumed $0.001 per judged attempt (the call is a fraction of a cent; usage logs
about 3.8K input tokens per question), with $0 and $0.003 as sensitivity.

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

### Round 2, 2026-11-08: headless run and split amendment

84 attempts (3 arms x 2 replicates x 14 tasks): Haiku 21/28, Sonnet low 22/28, Opus high 25/28 pass the
held-out suite; no errors, no tampering; $2.56. All 20 failures sat in five tasks (`cron_next`,
`wrap_text`, `lru_ttl`, `stack_vm`, `interval_merge`), and the hash split had put four of them in test,
leaving dev one failure in 28 attempts and no way to hit the INCONCLUSIVE floor of 20 test failures.
Before Jev saw any attempt I ran replicates 3–6 on those five tasks (60 attempts, $2.02) and replaced
the hash split with the stratified one above. Result: 144 headless + 14 pilot attempts; dev 52 attempts
with 13 failures, test 106 with 38. Peak context per attempt is 11–15K tokens headless (the Agent-tool
prefix is what made the pilot 80K+). Visible tests are green on all 144, as in the pilot.

### Round 3, 2026-11-08: prompts fixed on dev

Jev `jev-1.14.0` via `api.typesafe.ai` (the backend of the Muninn `decide` tool), one call per question
per attempt. Dev split, 52 attempts, 4 tasks, 13 failures. v1 (3-line diff context): pooled AUROC Q1 0.68,
Q2 0.64, Q3 0.62, Q4 0.78; TPR 1.00 and TNR 0.00 for Q1, Q3 and Q4 at 0.5 (Jev answers yes to everything);
within-task AUROC 0.38, 0.62, 0.33, 0.64. On `lru_ttl` Q1 and Q3 scored failures above passes. The
failing attempts leave `__len__` unpurged and the passing ones add `self._purge()` there; both are in the
diff context, so the information is present and Jev is not using it. v2 (30-line diff context, Q1 and Q3
told to check unchanged code in the context for the same faulty assumption): pooled AUROC Q1 0.79, Q2
0.63, Q3 0.79, Q4 0.68; within-task AUROC 0.42, 0.61, 0.41, 0.42. The pooled gain is between-task (Jev
learns which tasks are hard) and not attempt-level. One revision only, to limit overfitting 4 dev tasks.
v2 is frozen. Primary question by the pre-registered rule: Q3 (dev AUROC 0.789 against Q1's 0.787; a margin
this small is a tie, and all four questions are reported on test).

### Round 4, 2026-11-08: test split

Prompts frozen at v2 (hash in `prompts_frozen.json`), then the test split scored once: 106 attempts, 10
tasks (`cron_next`, `csv_line`, `expr_eval`, `glob_match`, `roman_strict`, `semver_cmp`, `stack_vm`,
`template_render`, `text_table`, `wrap_text`), 38 held-out failures. Primary question Q3 chosen on dev.
Decision rule, applied as written: (a) 84% of Q3's test misjudgements are mid-band, bootstrap 33–100%, so
the 70% threshold is met at the point estimate and not excluded below; (b) the dev-chosen ladder solves
0.82 test tasks against 0.80 for the visible-gated ladder, costs 0.91x always-Opus (bootstrap 0.57–1.21),
and is 9 points below the held-out-gated ladder's 0.91, outside the 5-point allowance. Not GO. Errors are
not confidently wrong either (the interval does not exclude 70%), so not NO-GO. Test size clears the
INCONCLUSIVE floor (106 attempts, 38 failures), though 10 tasks make every interval wide: Q3's AUROC
interval is 0.43–0.95. PARTIAL; Q1 and Q3 carry the only signal, and it is task-level.

Exploratory, run after the test split was read: band base rates (Q1 95% of all scores in 0.3–0.9, Q3 66%,
errors 84%, correct judgements 56%); diff-size baseline (within-task AUROC 0.07 for smaller diffs);
logistic combination (leave-one-task-out AUROC 0.44). Costs: Jev priced at $0.001 per judged attempt;
$0 and $0.003 change the ladder cost by under 10% and no conclusion (`results/v2_ladder_test.json`).
Limits: one judge version (`jev-1.14.0`), one prompt revision on 4 dev tasks, seeded-bug repair only,
mid-band retries uninformed (a lower bound for the informed retry), and the Opus and Sonnet arms ran at
the CLI's effort levels as headless children, not through the Agent tool.
