# downskill-shots

Does Haiku 5.5 need the worked examples the `down-skilling` skill puts in every
distilled prompt?

## Answer

Haiku 5.5 does not need examples to apply a rule. It does need them to apply the
rules the way the prompt's author meant. On three of the skill's distilled prompts,
run as shipped, as rules only, and bare (task plus label names), every arm with
rules scored **40/40 on rule-determined items**; bare scored 38/40, missing only the
one default it was never told. On judgment calls an example settles, the prompt as
shipped scored **22/24**, bare **20/24**, and rules without examples **15/24**: taken
literally, the rules push Haiku off the reading both the examples and its own default
agree on. A convention the model would not guess transferred from its example in 1 of
2 runs, so state it as a rule.

The anti-invention examples are no longer needed against invented technical details:
the example set that made Haiku 4.5 invent them in 19 of 20 rewrites produced 0 of 8
on 5.5, as did every version. A no-invention rule still matters: without one, 3 of 8
rewrites invented facts.

This would change with gold labels written by someone other than a Claude model, or
on tasks whose labels the model has no prior for: both are untested here.

## Findings

1. Rule-determined items need no examples on Haiku 5.5: 40/40 with examples, 40/40
   without (rules arms, 2 replicates, 20 items).
2. Rules without their examples are the worst arm on judgment items, 15/24 against
   22/24 with examples and 20/24 with no rules at all. The misses are the rules'
   literal reading: moderation rule 6 ("surface content hostile → flag") flagged a
   harsh opinion about an article in both runs; feedback's MIXED definition read a
   blocked-adoption wish as NEGATIVE in both runs.
3. The bare prompt matched the shipped prompt on feedback (10/10 conventions each)
   and came within 1 of it on moderation (10/12 against 11/12). Haiku 5.5's default
   reading already agrees with most of the examples' calls.
4. A house convention the model would not infer barely transfers through an example:
   "top customers by spending" filtered to completed orders in 1 of 2 runs with the
   example, 0 of 4 without. State such conventions as rules.
5. Haiku 5.5 no longer invents technical details in a dry rewrite of an abstract
   announcement, with or without anti-invention examples: 0/8 in every version,
   including the example set that produced 19/20 on Haiku 4.5 (round 2).
6. A no-invention rule still earns its place on 5.5. Without one, 3 of 8 rewrites
   invented facts and 4 of 8 reversed the source's own claim ("not a paradigm shift");
   with the rule, 0 to 1 of 8 invented anything, and the calibrated examples on top
   took it from 1/8 to 0/8, which n=8 cannot distinguish (round 2).
7. Defaults stated only in rules (`LIMIT 10` for "some") are followed 4/4 when stated
   and 0/2 when not: a rule earns its place by stating something the model would not
   assume.

## Method

Fixture: the `data-extraction`, `content-moderation` and `sql-generation` prompts from
`claude-skills/down-skilling/examples/` (as of `66d1719`), extracted verbatim by
`build.py`. Test items in `fixture.py`: 12 feedback comments, 12 moderation comments,
8 SQL questions over a deterministic SQLite shop (`build.py`, seed 7). Each item is
`rule` (the prompt's rules alone fix the gold) or `convention` (the rules leave it
open and a named worked example settles it).

Arms: `full` (prompt as shipped), `rules` (`<examples>` block removed), `bare` (task,
label names and output schema only, no definitions or tie-breaks). Model: Haiku 5.5
as Agent-tool subagents with no conversation context, each reading its prompt from a
file and answering all items in one reply. Two replicates, 18 runs. Effort at the
session's level (the Agent tool cannot set it).

Grading (`grade.py`): labels match gold exactly (both fields, gold may list two
defensible answers); SQL executes in DuckDB (PostgreSQL dialect) and its result must
contain every gold column with the gold row count, gold executed in SQLite. Grader
self-tested on hand-written replies before any run was graded.

Caveat: gold for convention items was written by Opus 5.5 from the examples, which
were written by an earlier Opus. The bare arm's agreement with that gold may partly
be shared Claude priors and does not show agreement with a human labeller.

## Log

### 2026-10-07: three arms, two replicates

Asked by Oskar: *"Haiku 45 really needed a lot of examples to do the job right ...
haiku 55 I expect means less in the way of examples, so should our down skilling
skill be adjusted accordingly?"*

Prior art, two web searches: instruction-tuned models gain unevenly from
demonstrations and are sometimes hurt by them (InstructEval, BUFFET); DeepSeek's R1
report recommends zero-shot over few-shot for its reasoning model. The common
practical advice is to start zero-shot and add examples where labels, formats or edge
cases are unclear. Nothing found that separates calibrating examples from teaching
examples the way the convention/rule split here does. That comes from 2 searches and
does not show it is absent.

| task | arm | rule items | convention items |
|---|---|---|---|
| feedback | full | 14/14 | 10/10 |
| feedback | rules | 14/14 | 7/10 |
| feedback | bare | 14/14 | 10/10 |
| moderation | full | 12/12 | 11/12 |
| moderation | rules | 12/12 | 8/12 |
| moderation | bare | 12/12 | 10/12 |
| sql | full | 14/14 | 1/2 |
| sql | rules | 14/14 | 0/2 |
| sql | bare | 12/14 | 0/2 |

Misses by run: feedback-rules r1 items 4, 9 and r2 item 4; moderation-full r2 item 8
(REMOVED where the example says FLAGGED); moderation-rules r1 items 1, 6 and r2 items
1, 8; moderation-bare r1 item 1 and r2 item 8; sql Q1 (completed-only) everywhere
but sql-full r2; sql-bare Q4 (`LIMIT 10`) in both runs. Data: `results.json`,
`runs.json` (subagent transcript paths in session 7c199534; the transcripts themselves are not committed).

Prediction before the run: examples would matter for conventions and not for rules.
The first half held. The rules arm falling below bare was not predicted.

### 2026-10-07: the anti-invention examples, retested on Haiku 5.5

Asked by Oskar about the skill's claim that its "model the silence" examples were
still needed: *"Why not retest?"*

Fixture: the voice-rewrite task from the May Haiku 4.5 assessment
(`muninn.austegard.com/references/haiku-assessment/`): rewrite an 85-word promotional
announcement of a caching layer, which names no mechanism, in a dry voice at the same
length. Four versions, 8 Haiku 5.5 runs each, prompts verbatim from that archive
(`silence/prompts/`): the vanilla request (4/20 invented details on 4.5), the
original down-skilled prompt whose examples drove invention (19/20 on 4.5), the
calibrated prompt with anchored "model the silence" examples (0/5 on 4.5), and that
calibrated prompt with its `<examples>` removed.

Scoring: the archive's own term list of invented technical details (`HALLUCINATED_DETAILS`
in `n20/score.py`: LRU, TTL, Redis, p99, in-memory and so on), plus a read of every
output for unsupported claims (facts the source does not state) and for reversals of
the source's claims. Labels and texts in `silence/outputs.json`.

| version | invented details (term list) | unsupported claims | reverses the source | 60–90 words |
|---|---|---|---|---|
| vanilla | 0/8 | 3/8 | 4/8 | 7/8 |
| original down-skilled | 0/8 | 1/8 | 0/8 | 6/8 |
| calibrated | 0/8 | 0/8 | 0/8 | 7/8 |
| calibrated, no examples | 0/8 | 1/8 | 0/8 | 7/8 |

The unsupported claims: vanilla "We tested it across a range of workloads with
consistent results", "Read the docs to get started", "the gains hold up in practice";
original "consult the release documentation"; calibrated-no-examples "Future updates
will show measured results". The reversals are all one move: the source calls the
layer a paradigm shift and the vanilla rewrite says it is not one.

None of the calibrated runs copied the process claims its own examples model ("the
old path remains available", "adoption is opt-in"), which the term list would not
have caught either.

Reading: the failure the examples were built against is gone on 5.5 at this n. The
rule and the fact-listing step carry the remaining work; the examples are optional.
One task, 8 runs per version, labels by Opus 5.5.
