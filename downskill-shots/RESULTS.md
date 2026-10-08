# downskill-shots

Does Haiku 5.5 need the worked examples the `down-skilling` skill puts in every
distilled prompt?

## Answer

Haiku 5.5 does not need examples to apply a rule; it needs them to apply rules the
way the author meant. On three distilled prompts, every arm with rules scored
**40/40 on rule-determined items**. On judgment calls an example settles, the shipped
prompt scored **22/24**, bare (task plus labels) **20/24**, and rules without
examples **15/24**: taken literally, the rules push Haiku off the reading the
examples and its own default agree on. State unguessable conventions as rules.

Nothing tested here makes Haiku 5.5 invent technical details in a rewrite: 0 of 32
across round 2's versions, including the example set that drove Haiku 4.5 to 19/20,
and 0 of 64 across round 3's fact-listing variants. A no-invention rule still cuts
unsupported claims (3/8 without it). A fact list steers what gets carried over: one
Haiku writes itself counts the source's self-description as fact (8/8 rewrites
repeated it), and one supplied in the prompt is followed omissions and all.

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
8. A fact-listing step does not prevent invention on Haiku 5.5. Hidden in the
   process, visible as a `<facts>` block, supplied in the prompt, or absent: 0/16
   rewrites with an invented technical detail in each arm, two inputs (round 3).
9. A fact list in the prompt steers the rewrite's content more than any instruction
   does. Supplied lists that left out the source's promotional claims cut rewrites
   repeating them to 0/8 and 1/8, against 5/8 to 8/8 in every other arm; a supplied
   list that left out "it is launching" produced 2/8 rewrites calling the product
   "in beta" (round 3).
10. A list Haiku writes itself counts the source's claims about itself as facts
   ("the team describes it as more than an incremental improvement"), and the rewrite
   carries them: 8/8 with the visible list and 8/8 with the hidden step on both
   inputs, against 5/8 and 7/8 with no step (round 3; post hoc phrase count).

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

### 2026-10-08: does the fact-listing step do anything?

Asked by Oskar after a system-card figure showed Haiku 5.5 near 0% on chain-of-thought
controllability (following instructions about the content of its own thinking): does
`down-skilling` 1.6.0's "keep a step that lists the source's facts before writing"
mean anything when the prompt also says "output ONLY the rewritten paragraph"? In
round 2's calibrated prompt the step could only run in hidden reasoning, and the
prompt had already written the list.

Arms (`silence/steps/build.py`), all from round 2's calibrated prompt without
examples, differing only in step 1: **P** the step with the list supplied (round 2's
shape), **H** "List the factual claims in the input" with no list, reply is the
paragraph only, **N** no listing step, **V** the list as a visible `<facts>` block
before a `<rewrite>` block. Two inputs: round 2's caching announcement and a new
85-word semantic-search announcement, also naming no mechanism, with an invented-term
list (embedding, vector, HNSW, BM25 and so on) fixed in `score.py` before any run.
8 Haiku 5.5 Agent-tool runs per arm and input, 64 in all. Every output was read.

| arm | invented details (term list) | repeats a source promotional claim, cache / search | product status wrong |
|---|---|---|---|
| P, list supplied | 0/16 | 0/8 / 1/8 | 2/8 (search) |
| H, hidden step | 0/16 | 8/8 / 8/8 | 0/16 |
| N, no step | 0/16 | 5/8 / 7/8 | 1/8 (search) |
| V, visible list | 0/16 | 8/8 / 8/8 | 0/16 |

All 64 replies kept the format and the 60–90 word range; all 16 visible lists were
free of the invented-term list. No rewrite reversed a source claim.

The promotional-claim count is post hoc: the phrase list (`incremental`, `change how
teams/people`, `paradigm`, `upgrade`, `new way to search`, `state-of-the-art`,
`dramatic`/`far more` and the like) was written after reading the outputs, because
the difference was visible on reading. The status errors are "the feature is
currently in beta" against a source that says it is launching. The search arm's
supplied list was mine and omitted the launch; two of the eight P rewrites followed
the list rather than the source.

Reading at the time: the step is not what keeps Haiku 5.5 from inventing, since the
no-step arm is as clean as the rest. A list the model writes is faithful but
indiscriminate; it records the source's self-description as fact and the rewrite
follows. A list written by whoever prepares the input is followed closely, which
makes it a strong lever and a single point of failure. H against N (8/8 against 5/8
on cache) suggests the hidden step does reach the output, but n=8 on one input does
not separate it from noise. `down-skilling` 1.7.0 drops the step as an anti-invention
measure and says what each kind of list does.

