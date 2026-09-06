# harness-bench — pre-registration

**Date**: 2026-09-06. Written before any agent ran.

## Question

Holding the model fixed, what does an agentic harness loop (run the tests, read
the failure, try again) buy over a single-shot edit harness on Aider Polyglot?

Prompted by dollspace.gay's 2026-09-06 claim that a self-built harness (Peritus)
beat the leader on a harness benchmark. The published leaderboard entry to sit
next to is aider 0.86.0 + gpt-5.2 = 0.880 on aider-polyglot, 225 tasks
(TheLime1/harness-bench, `data/runs/aider-polyglot/2026-05/aider/gpt-5.2`).

## Arms

Model fixed at `claude-haiku-4-5` for every arm. Only the harness varies.

| arm | protocol |
|---|---|
| `oneshot` | agent reads instructions + stub, writes the solution file, stops. No test execution, no test file in context. Scored pass@1. |
| `agentic` | failures from `oneshot` are re-dispatched once with the pytest/go/cargo failure output. Scored pass@2. |

`agentic` reuses `oneshot`'s first attempt, so the retry is the only added cost
and the two arms are paired by construction.

## Metric

Pass rate against the exercise's own test suite, run by the driver from a
**pristine** copy: the graded tree is rebuilt from the upstream exercise and only
the files named in `.meta/config.json:files.solution` are overlaid from the
agent's work dir. Every file the agent touched outside that set is reported.
(`experiments` PR #76: a grader that compared against a restored tests/ dir saw
nothing when a run wrote its own `conftest.py`.)

Test commands: `python3 -m pytest`, `go test ./...`,
`cargo test -- --include-ignored` (Exercism marks all but the first Rust test
`#[ignore]`; without the flag a stub scores green on 1 passed / 11 ignored).

## Pre-registered null

If the harness loop does nothing, `oneshot` and `agentic` pass rates are equal
within binomial noise on n paired trials.

## Pre-registered ceiling stop

This repo has hit the same wall twice — `orchestrated-coding-pareto` (haiku-solo
14/14 = opus-solo 14/14 on authored Python specs) and the PR #76 stage-1 pilot
(sonnet-low 14/14, opus-high 13/14). Both designs went vacuous because the weak
arm never failed.

**Stop rule, fixed in advance:** if `oneshot` passes ≥ 11 of the 12 pilot tasks,
the task set does not discriminate. Report that and do not run the full sweep.
The reason to expect polyglot might clear it where authored specs did not: aider
selected these 225 from the Exercism tracks specifically because the models of
the day *failed* them, so the set is chosen for difficulty rather than authored
for it.

## Pilot

12 exercises, 4 each from python / go / rust, sampled with a fixed seed from the
practice sets (34 / 39 / 30 available). Batched 4 per subagent — a subagent costs
~32.5k tokens before it reads its prompt (`claude-workspace` docs/delegation.md).
