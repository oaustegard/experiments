# presearch-reask — plan

Two pieces of HyperSkill's execution protocol (arXiv 2611.15178, "C3"), tested on
Haiku 5.5 at `--effort low` through `claude -p`. The paper never ablates C3 option
by option, so nothing in it says which option earned its +10.4 F1. These are the two
that are not already measured here (voting and option-weighing were, in `dettin-68m`).

- **Question pre-search.** Before the model reasons, the question goes verbatim to
  the search tool once, outside the tool budget, and the results are prepended.
- **Final-answer turn (re-ask).** A rollout that ends without a closed `<answer>`
  tag (round cap reached) gets one more turn that asks for the answer.

## Fixture

MuSiQue-Answerable dev (2,417 questions, 2-4 hops, built so that every hop is
needed; HotpotQA was dropped because METHODS.md records it is mostly answered from
memory). Eval set: 300 questions, seed 0, stratified 150 2-hop / 100 3-hop / 50
4-hop. Knowledge base: every paragraph of the first 1,000 dev questions plus the
300 (deduplicated by title+text), searched by BM25 (k1 1.2, b 0.75), top 5 per
query, each result as `title: text`.

## Arms (all Haiku 5.5, `--effort low`, tools off, text-tag protocol)

| arm | retrieval |
|---|---|
| CB | closed book, one call |
| PS | pre-search only: question to BM25 once, top 5 in the prompt, one call, no tool |
| AG3 | `<search>` tool, budget 3, round cap 4, no pre-search |
| AG4 | same with budget 4, round cap 5 (equal total searches to PS+AG3) |
| PS+AG3 | pre-search outside the budget, then AG3 |

Each round is one stateless `claude -p` call carrying the transcript so far. A round
that emits `<search>` past the budget is told the budget is spent. At the round cap
without `<answer>`, the re-ask turn runs once. Both scores are kept: before
re-ask (no answer = 0) and after.

## Test 1b — decider failure rate

`scripts/decide.py eval tests/data/decider_eval.jsonl` (hub, 104 items, k=3) with
`--effort low` added to the CLI argv: count samples that fail to parse. The re-ask
is only worth building into decide.py if this rate is above noise.

## Metrics

MuSiQue EM and token F1 (SQuAD normalisation, max over answer + aliases). Paired
bootstrap (10k) for arm differences. Splits: hop count; questions CB gets wrong
(F1 < 0.5), where retrieval has to carry the answer. Cost and latency from the CLI.

## Predictions (written before any run)

- **P1** decide.py at low effort: ≤ 2% of 312 samples unparseable; accuracy within
  0.03 of the default-effort 0.846.
- **P2** AG3: ≥ 5% of questions reach the round cap with no answer; re-ask gets an
  answer for ≥ 60% of them; re-ask adds ≥ 1 F1 point on the arm.
- **P3** PS+AG3 beats AG3 by ≥ 3 F1, 95% CI excluding 0.
- **P4** PS+AG3 vs AG4 (same search count): |difference| < 2 F1, i.e. pre-search's
  gain over AG3 is mostly the extra search, not who writes the first query.
- **P5** PS (one-shot retrieval) trails AG3 by ≥ 5 F1 on 3-4 hop questions and is
  within 3 F1 on 2-hop.
- **P6** CB F1 ≤ 0.30.
- **P7** PS+AG3 issues ≥ 0.5 fewer model-written searches per question than AG4.
