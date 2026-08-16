# orchestrated-coding-pareto — can a big orchestrator + cheap fleet beat frontier-solo on coding?

**Date**: 2026-08-16. **Harness**: Claude Code on the Web, Workflow tool, session model
Claude Fable 5; workers pinned per-arm to `claude-haiku-4-5`, `claude-sonnet-5`,
`claude-opus-5`, all at effort `medium`.
**Lineage**: continues [`luna-onprem-tco/`](../luna-onprem-tco/RESULTS.md) (PR #37);
the follow-up question was whether a large model orchestrating a fleet of
Luna-class/open-weight workers is Pareto-optimal for coding tasks.

## Summary

- **Quality saturated at every difficulty tier we could build.** Haiku 4.5 one-shot went
  14/14 against hidden test suites — the same score as Opus 5 — across three escalating
  tiers ending in a 20-opcode stack-VM interpreter and an exact-output table formatter.
  Sonnet 5 went 13/14 (it accepted `1.0.0-01` as a valid semver). On well-specified,
  self-contained Python modules, the cheap tier is not distinguishable from frontier
  on correctness.
- **The orchestration arms never activated.** Both retry arms (haiku + raw test
  feedback; opus-diagnoses → haiku-fixes) seed from haiku failures, and there were
  none. On this task class the measured value of an orchestrator is zero because
  there is nothing to orchestrate. The experiment bounds the thesis, not confirms it:
  whatever premium a big orchestrator earns must come from work this design
  deliberately excludes — under-specified asks, multi-file scope, large-context
  navigation, decomposition itself.
- **Token economics, not accuracy, decided the Pareto frontier — and inverted it.**
  Haiku emitted **280,717 output tokens** for its 14 tasks; Opus emitted **42,019**
  (ratio 6.7×). At Anthropic list prices the 5× per-token discount is more than
  cancelled: **haiku-solo $0.101/task vs opus-solo $0.079/task at equal quality**.
  In this harness, at these settings, opus-solo strictly dominates the
  Anthropic-priced fleet tier.
- **The fleet thesis survives only at Luna-class prices.** Repricing haiku's measured
  token profile at GPT-5.6-Luna direct rates ($0.20/$1.20) gives **$0.024/task** —
  4.2× cheaper than opus-solo; at DeepSeek V4 Flash rates, **$0.0057/task** (13.8×).
  The variable that decides Pareto-optimality is the fleet tier's sticker price and
  verbosity discipline, not the orchestration architecture.

## Design

Five arms over 14 tasks, each task a precise spec plus a hidden pytest suite
(9–35 tests) that the workers never see:

| arm | protocol |
|---|---|
| haiku-solo / sonnet-solo / opus-solo | one-shot: read spec, write module; no code execution, no test access |
| haiku-retry | seed from haiku-solo failures; re-attempt with raw pytest output, ≤2 rounds |
| orch-haiku | same seed; Opus 5 (effort high) reads spec+code+failures, writes guidance (no code); haiku re-attempts with it, ≤2 rounds |

Every hidden suite was validated against a reference implementation before any model
saw the task (this caught authoring bugs in 3 of 14 suites — see `ERRORS.md`).
Grading is local (`harness/grade.py`); generation ran as Workflow subagents with
per-phase output-token metering via sequential `budget.spent()` marks. Tool-use
counts averaged 3.0–3.3 per agent — consistent with exactly read-spec, write-module,
return-structured-output, i.e. no worker wandered into the test files.

**Task tiers, and why there are three.** Tier 1 (8 tasks: range parser, TTL-LRU cache,
lexicographic toposort with cycle extraction, strict Roman numerals, CSV state
machine, full SemVer 2.0 comparison, template engine with exact error contracts,
interval merge) was designed edge-case-dense on the theory that cheap models lose
points to spec compliance. Early grading showed haiku 6/6 on what had landed, so the
bank was extended mid-run — disclosed here deliberately — with tier 2 (expression
evaluator reproducing Python's `-2**2`/`2**-1` quirks, `**` path-aware glob,
5-field cron-next with the dom/dow OR rule, exact-output greedy text wrap) and,
after a second sweep, tier 3 (stack-VM with six-error taxonomy and call stack;
ASCII table formatter with wrapping/alignment rules tested to the character).
Haiku swept all three tiers.

## Results

| arm | pass | t1 | t2 | t3 | output tokens | $/run | $/task | $/task @Luna | @DS-Flash |
|---|---|---|---|---|---|---|---|---|---|
| haiku-solo | 14/14 | 8/8 | 4/4 | 2/2 | 280,717 | $1.41 | $0.101 | $0.024 | $0.0057 |
| sonnet-solo | 13/14 | 7/8 | 4/4 | 2/2 | 65,361 | $1.01 | $0.072 | — | — |
| opus-solo | 14/14 | 8/8 | 4/4 | 2/2 | 42,019 | $1.10 | $0.079 | — | — |
| haiku-retry | vacuous | | | | — | — | — | | |
| orch-haiku | vacuous | | | | — | — | — | | |

Costs use measured output tokens plus content-input estimated at chars/4 over
exactly what each worker read (params in `params.json`; the ~33k-token fixed harness
kernel each node carries is a CCotw artifact, cache-read in practice, excluded and
reported separately). The Luna/DS-Flash columns reprice haiku's token profile at
those vendors' list rates — a counterfactual, not a measurement: it assumes
Luna-class models match Haiku 4.5's quality and verbosity on this class, neither of
which this experiment tested.

The verbosity gradient is monotone and steep: haiku 20.1k output tokens/task, sonnet
4.7k, opus 3.0k. Haiku's ratio worsened as tasks got harder (tier-3: 25.5k/task vs
opus 2.3k — 11×). A caveat sits under it: the harness's `effort: medium` knob has no
documented mapping on Haiku 4.5 (the API-level effort parameter errors on that
model), so some of the volume is likely harness-configured thinking. Measured is
measured, but the ratio may be tunable rather than intrinsic.

## What this says about the orchestrator-fleet thesis

1. **On the task class where a fleet is easiest to run — well-specified, decomposed,
   self-contained units — the fleet needs no orchestrator brain and the frontier
   model needs no fleet.** Correctness is saturated both ways; only price moves.
   The architecture question dissolves into a procurement question.
2. **At current Anthropic pricing the cheap tier is not cheap.** Verbosity is a
   price multiplier: a fleet model that thinks 6.7× more tokens than the frontier
   model at 1/5 the price costs *more* per solved task. Any fleet procurement
   decision should measure tokens-per-task, not quote per-token prices.
3. **At Luna-class pricing ($0.20/$1.20) the fleet wins the priced tier by ~4× and
   open-weight Flash-class by ~14×** — if quality holds, which is the untested
   premise. This is where Oskar's Pareto intuition lives or dies, and it is a
   cross-vendor benchmarking question, not an architecture question.
4. **The orchestrator's residual value is confined to what produces the
   well-specified units in the first place** — decomposition, spec-writing,
   integration, and review of under-specified work. That is the regime the
   `omnigent` Polly pattern and Anthropic's own multiagent guidance target, and it
   is unmeasured here; measuring it needs tasks that are *not* precisely specified,
   which requires a fundamentally different (rubric- or integration-graded) eval
   than hidden unit tests.
5. **Tie-back to PR #37's self-hosting question:** an orchestrated fleet is
   round-based and batchable, so the Luna *batch* tier ($0.10/$0.60) applies,
   halving the fleet column again. Self-hosting the fleet still does not pencil:
   at ~20k output tokens/task, saturating PR #37's 7.64 B-token/night break-even
   volume would take ~380k tasks/night — orders of magnitude past any single team's
   coding workload. The fleet workload shape is friendlier to the box than office
   chat (parallel, batchable), but the volume is still missing.

## Follow-up: the effort knob (added same day, after PR #39 opened)

Oskar asked whether the haiku delegation ran at too high an effort setting. Re-ran
all 14 tasks at `effort: low` — the floor this harness exposes (there is no
thinking-off toggle on Workflow agents, and CCotw has no raw API key by design, so
fully-disabled thinking remains untested):

| arm | pass | output tokens | tok/task | $/task | $/task @Luna |
|---|---|---|---|---|---|
| haiku-low | 12/14 | 206,668 | 14.8k | $0.075 | — |
| haiku-low + test-feedback retry | 14/14 | 223,169 | 15.9k | $0.080 | $0.019 |
| haiku-low + opus-diagnose→haiku-fix | 14/14 | 232,374 | 16.6k | $0.097 | — |
| opus-solo (reference) | 14/14 | 42,019 | 3.0k | $0.079 | — |

Three answers fall out:

1. **The verbosity is mostly intrinsic, not the knob.** Dropping `medium` → `low`
   cut output tokens 26% (20.1k → 14.8k/task) — still 4.9× opus. If the 6.7× ratio
   were configured thinking, `low` should have collapsed it; it didn't.
2. **`low` finally broke the ceiling — and activated the orchestration arms.**
   haiku-low failed `parse_range` (its hyphen-splitter rejects even `"1-3"`) and
   `roman_strict` (one invalid form accepted). Both retry lanes ran on this seed,
   and **both went 2/2 in one round**: raw pytest output alone fixed everything the
   opus diagnosis fixed. Measured orchestrator premium over mechanical test
   feedback: zero quality, +$0.017/task cost (9,427 opus output tokens of
   diagnosis). n=2 failures — a first datapoint, not a verdict — but its direction
   matches the main result: the *verification loop* buys the quality back; the
   orchestrator's intelligence adds nothing a failing test didn't already say.
3. **The best cheap pipeline ties frontier-solo at Anthropic prices and wins only
   on Luna prices.** haiku-low + test-retry reaches 14/14 at $0.080/task vs
   opus-solo's $0.079 one-shot — a tie that additionally requires owning a grading
   harness. Repriced at Luna direct, the same pipeline is $0.019/task, 4.1× under
   opus-solo. The procurement conclusion from the main run stands unchanged.

## Confidence

Ranked by how much a wrong value would move the conclusion:

1. **Task-class generalization** (high impact, known limitation): all 14 tasks are
   single-module, fully-specified, stdlib-only. The saturation result should NOT be
   read as "Haiku ≡ Opus at coding"; it is "Haiku ≡ Opus at implementing precise
   specs ≤ ~40 rules". The tiers rule out the easy versions of this objection, not
   the hard one (multi-file, ambiguous, long-context work).
2. **Verbosity ratio** (high impact, medium confidence): measured through an
   undocumented effort mapping on Haiku; could compress under API-native settings.
   If haiku's output volume dropped to opus levels, haiku-solo would cost
   ~$0.03/task at Anthropic prices and the inversion would reverse.
3. **Training contamination** (medium impact, unquantified): Roman numerals, semver,
   glob, cron are classic exercises; all models have seen kin. The exact-corner
   contracts (error taxonomies, message texts, tie-breaks) are novel, and sonnet's
   single miss was exactly such a corner — but contamination inflates all arms
   roughly equally, which is why the *comparison* is more trustworthy than any
   absolute pass rate.
4. **Luna repricing** (medium impact): inherits every provenance caveat from
   PR #37's params (search-summary reads, secondary confidence).
5. **Input-token estimates** (low impact): outputs dominate cost at every price
   point tested; a 2× error in input estimation moves totals by <6%.

## Reproduce

```
python3 harness/grade.py --self-test          # validate hidden suites vs references
python3 harness/grade.py --solutions data/round0_solutions.json   # re-grade stored code
python3 harness/analyze.py                    # rebuild data/analysis.json
python3 recheck.py                            # prose-vs-data consistency (see below)
```

Generation itself requires the CCotw Workflow harness (scripts in `harness/*.js`,
run records in `data/marks.json`).
