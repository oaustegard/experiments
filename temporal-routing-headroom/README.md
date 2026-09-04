# temporal-routing-headroom

Stage 1 of testing SWE-Router's routing claim (arXiv 2607.00053v1, Son et al., ICML 2026
DL4Code workshop) against the `agent-routing` skill.

SWE-Router runs a cheap model for K exploratory turns, reads the partial trajectory with
a learned value head, and then decides whether to continue cheaply or escalate. Our skill
routes on task shape declared up front and cascades only on a *verified* failure. Both
target the same waste; they disagree on what the escalation signal is.

**Status: see [RESULTS.md](RESULTS.md).** Two Stage-1 pilots, two ladder arms, and a
three-shape capability probe. Headline: the cascade `agent-routing` prescribes — Sonnet at
low effort, verify, **the same model one effort step up** carrying the failure output —
solves **13/14 at 0.31× the cost of always-Opus**, which solves 10/14. Escalating to Opus
instead reaches the same 13/14 for 0.76×, so the tier jump costs 2.5× and buys nothing.
Nothing in the probe separated the tiers; the concision lever cut rung-1 output 2.9%
against the 37% the skill measured on generation work; and handing rung 2 the first rung's
stated reasoning changed nothing (13/15 against 12/15, 1% fewer tokens), so the informed
retry's value sits in the patch and the failing assertions.

## Relation to orchestrated-coding-pareto

`orchestrated-coding-pareto` is single-shot: `spec.md` in, one Python module out, hidden
pytest. There are no turns and no observations, so a probe has nothing to read. The spec also fully determines difficulty, which is the case where the paper's own
theorem predicts zero gain.

So each of those 14 tasks becomes a small repository with a seeded bug and a bug report
that names the failing test and nothing else. Difficulty varies; the report does not say how. The paper's
premise is that a similar issue text can hide either a typo or a multi-module fix, and
building the tasks this way puts that confound in on purpose instead of assuming a
benchmark already contains it.

## The task set

`harness/build_tasks.py` generates `tasks/` from `seeds/*.json` plus the reference
modules and hidden suites next door. Each reference module is split across
`solution/core.py` and `solution/util.py`, a bug is applied as a single verified
find/replace, and the visible test file is sliced out of the hidden suite.

| class | n | bug lives in | what makes it hard |
|---|---|---|---|
| shallow | 4 | `core.py` | the function the failing test exercises directly |
| deep | 4 | `util.py` | a module no failing test names |
| paired | 6 | two sites | one broken assumption written twice |

The eight shallow and deep tasks are the control: the pilot solved all of them with
Sonnet 5 at effort low, so they are a measured ceiling rather than a guess.

A `paired` task writes one wrong assumption into two call sites. `cron_next` and
`expr_eval` span both files — util decides which cron fields count as restricted while
core decides what to do when two of them are; util's tokenizer emits the `**` token that
core's parser consumes. The other four sit in `core.py`: `lru_ttl`'s expiry predicate and
the `__len__` that has to agree with it, `stack_vm`'s binary-operand pop order and `SWAP`,
`text_table`'s padding width and its cell wrapper, `wrap_text`'s paragraph splitter and
its joiner.

`build_tasks.py` exits non-zero on a pair unless repairing **either** site alone leaves
the hidden suite red, and repairing the first turns the **visible** suite green. That is the trap: a run
that finds one cause, fixes it, sees its tests pass and stops has shipped a wrong patch.
Verified live per task rather than trusted from `meta.json`:

| task | sites | hidden tests still red after fixing one |
|---|---|---|
| cron_next | util + core | 2 |
| expr_eval | util + core | 2 |
| text_table | core + core | 8 |
| stack_vm | core + core | 1 |
| lru_ttl | core + core | 1 |
| wrap_text | core + core | 1 |

Every task is verified at build time and refuses to ship otherwise:

- the split repo, before any bug, passes the hidden suite
- the bug turns the hidden suite red
- the visible subset is also red, so the agent sees a failure, and fails rather than errors
- the visible slice is green on the unbugged reference, so a broken slice cannot pass as a seeded bug
- for trap tasks, the partial fix passes every visible test and leaves residue outside it

`build_tasks.py --check` rebuilds into a temp directory and diffs, so a seed edited
without a rebuild fails `tests/`.

## Stage-1 metrics

Two arms, neither of them routing: weak and strong both run every task to completion.
`harness/oracle.py` then reports

- **cost per completed task** per arm, which is what the routing decision turns on
- **the oracle router** — route to weak iff weak solves it — the upper bound on every
  router, learned or not
- **|W \ S|**, the tasks the weak arm solves while the strong arm fails them.
  SWE-Router's curve rises above its own all-strong marker only because this set is
  non-empty. `agent-routing` assumes escalation always helps, which requires it to be
  empty
- how many tasks flip pass/fail across replicates, since a task that flips carries no
  routing signal

The stop rule: if the oracle's cost per completed task is not materially below
all-strong, no router can pay and Stages 2 and 3 do not run.

Three replicates is the floor. `orchestrated-coding-pareto` measured disjoint failure
sets and a 23% token gap between two runs of the same model on the same tasks, so at
n=14 a one-or-two task difference is noise and token deltas are the signal to trust.

## Running it

```bash
python3 harness/build_tasks.py                 # regenerate tasks/ from seeds/
python3 harness/build_tasks.py --check         # confirm tasks/ matches seeds/
python3 -m pytest tests/ -q                    # 122 invariant checks over the registry
python3 harness/grade_agentic.py --self-test   # every task repo starts red

RUN=/tmp/stage1/r1
python3 harness/emit_prompts.py --arm weak   --run-root $RUN --stage
python3 harness/emit_prompts.py --arm strong --run-root $RUN --stage
python3 harness/emit_prompts.py --arm weak   --run-root $RUN --task parse_range
```

CCotw carries no `ANTHROPIC_API_KEY`, so `emit_prompts.py` prints prompts and the parent
session dispatches them through the Agent tool. Both arms get the same prompt text from the same
template; the model and the effort level are the only differences.

Grading restores `tests/` from the pristine task before running the hidden suite, and
records whether a run edited its tests. A run that rewrote its tests scores as failed.

```bash
python3 harness/emit_prompts.py --arm weak --run-root $RUN --manifest > runs.json
python3 harness/grade_agentic.py --runs runs.json --out data/results_stage1_r1.json
python3 harness/oracle.py
```

`oracle.py` also needs `data/tokens_stage1*.json` — measured output tokens per arm per
task, from `budget.spent()` deltas, same convention as `orchestrated-coding-pareto`.

## Known limits

- The bugs are planted, so their depth may be more legible to a probe than a real issue's
  would be. This biases toward the method under test.
- A seed says where each bug goes, so difficulty is authored rather than sampled from
  real issues.
- 14 tasks is too few to separate routers on pass rate. It is enough to measure the
  oracle bound and the disjointness, which is all Stage 1 claims.
- Prices in `params.json` are secondary provenance, copied from
  `orchestrated-coding-pareto` so the two experiments' costs compare.
