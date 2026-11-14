# presearch-reask — results

HyperSkill (arXiv 2611.15178) credits most of its gain to an "execution protocol" of
prompt and decoding options that its baselines did not get, and never ablates those
options one at a time. Two of them had not been measured here before: **question
pre-search** (send the question verbatim to the search tool once before the model
starts, outside its tool budget) and a **final-answer turn** (ask once more when a
rollout ends without an answer). Both were tested on Haiku 5.5, Anthropic's smallest current model,
called at `--effort low` (the CLI's lowest reasoning-effort setting). The model searches by writing a
`<search>` tag; it gets a budget of searches and a round cap of one turn more than the budget, after which
the rollout ends. The "hub decider" is `scripts/decide.py` in `claude-workspace`, which asks Haiku
fixed-form classification questions and parses a JSON reply; it was checked because a re-ask would only
be worth adding there if Haiku's replies sometimes fail to parse.

## Answer

Pre-search is worth adopting; the re-ask is not. The test set was 300 MuSiQue multi-hop questions over a 14,661-paragraph BM25 index.
Pre-search plus a 3-search tool budget scored F1 0.570 <!-- @evidence[[results/scores.json#arms.PSAG3.all.f1]] -->
The tool alone scored 0.532 <!-- @evidence[[results/scores.json#arms.AG3.all.f1]] -->
Pre-search also beat a 4-search budget, which gives the model the same number of searches, by 0.045 F1 <!-- @evidence[[results/scores.json#pairs.PSAG3-AG4.all.diff]] -->
That margin has a 95% CI of 0.008 to 0.083, and pre-search did it with fewer searches and slightly lower cost per question.
So the gain is not the extra search: allowed 4 searches instead of 3, Haiku wrote only 0.23 more per question on average. Why the follow-up searches did better with pre-search is not measured directly; the likely reading is that results for the whole question let Haiku aim its next query at the missing hop.
The re-ask fired on 1 of 300 questions per arm, and the hub decider at low effort produced 0 unparseable samples out of 312 <!-- @evidence[[results/decider_low.json#unparseable]] -->
So the re-ask has nothing to recover. A different backbone or a tighter round cap could change that; the pre-search result rests on one dataset and one retriever.

## Findings

1. Pre-search plus 3 searches beat 3 searches alone by 0.037 F1 (CI 0.000 to 0.075). <!-- @evidence[[results/scores.json#pairs.PSAG3-AG3.all.diff]] -->
   It led 4 searches by 0.045 (CI 0.008 to 0.083). Over 4 searches the lead was +0.031 on 2-hop and +0.059 on 3-4 hop questions; the 3-4 hop CI excludes 0, the 2-hop one does not. (`run/score.py`)
2. Giving the tool a fourth search did nothing: 4 searches vs 3 differ by -0.008 F1. <!-- @evidence[[results/scores.json#pairs.AG4-AG3.all.diff]] -->
   Haiku used 1.97 searches per question when allowed 4 <!-- @evidence[[results/scores.json#arms.AG4.model_searches_per_q]] -->
   and answered with no search at all on 29 of 300. (`results/search_behaviour.json`)
3. With pre-search Haiku wrote 1.11 searches per question <!-- @evidence[[results/scores.json#arms.PSAG3.model_searches_per_q]] -->
   and still reached a gold-paragraph recall of 0.602 <!-- @evidence[[results/scores.json#arms.PSAG3.gold_paragraph_recall]] -->
   against 0.525 for the 4-search arm. <!-- @evidence[[results/scores.json#arms.AG4.gold_paragraph_recall]] -->
   Haiku's own first query found gold recall 0.407, <!-- @evidence[[results/search_behaviour.json#AG4.first_query_gold_recall]] -->
   slightly below 0.431 for the verbatim question, <!-- @evidence[[results/search_behaviour.json#PS_verbatim_gold_recall]] -->
   so the first query explains little of the gap; the rest arises in the follow-ups.
4. The final-answer turn had nothing to do: 1 of 300 questions reached the round cap without an answer in the 3-search arm, 0 in the 4-search arm, and no search past the budget was attempted in any arm. (`results/scores.json` `reask`)
5. The hub decider (`scripts/decide.py`, 104 items, 3 samples each) at low effort parsed all 312 samples; accuracy 0.808 <!-- @evidence[[results/decider_low.json#report.overall.accuracy]] -->
   against 0.846 at default effort; on 104 items one standard error is about 0.04, so the drop is not distinguishable from noise.
6. Retrieval of any kind carries most of the score. Closed book scored 0.263 F1, <!-- @evidence[[results/scores.json#arms.CB.all.f1]] -->
   low enough that most of the questions need the knowledge base; pre-search alone with no tool scored 0.485. <!-- @evidence[[results/scores.json#arms.PS.all.f1]] -->
7. The whole run (1,500 rollouts) cost $1.48 at list price <!-- @evidence[[results/search_behaviour.json#total]] --> and took 29 minutes at 8 parallel calls.

## Method

**Fixture.** MuSiQue-Answerable dev (HF `bdsaglam/musique`), 300 questions drawn with
seed 0: 150 2-hop, 100 3-hop, 50 4-hop. The knowledge base is every paragraph of the
first 1,000 dev questions plus the 300 (14,661 paragraphs after deduplication, so
each question's 18 MuSiQue distractors are in it). Search is plain BM25 (k1 1.2,
b 0.75) over `title text`, top 5 per query. HotpotQA was dropped because METHODS.md
records it is mostly answered from memory.

**Arms.** All Haiku 5.5 through `claude -p --model haiku --effort low --tools ""`,
with a text-tag protocol (`<search>…</search>`, `<answer>…</answer>`) and one
stateless call per round carrying the transcript. CB: closed book. PS: pre-search
results in the prompt, one call, no tool. AG3 / AG4: search tool with a budget of 3 /
4 searches and a round cap one higher. PSAG3: pre-search outside the budget, then
AG3. At the round cap without an answer, a re-ask turn runs once; both the before
and after answers are scored.

**Metrics.** MuSiQue EM and token F1 (SQuAD normalisation, max over the answer and
its aliases); paired bootstrap, 10,000 resamples, over the 300 questions. Splits by
hop count and on questions closed book gets wrong. Gold-paragraph recall is the
share of a question's supporting paragraphs that appeared in any result shown to
the model.

**Decider check.** `run/decider_low.py` wraps the hub's `decide.py` with `--effort
low` added to its CLI call and counts transport errors, unparseable replies and
replies missing a question.

## Log

### Round 1 — 2026-11-14

Asked: Oskar, after the HyperSkill review: *"Is there anything in the scaffolding
that may be a target for adoption?"*, then *"Ok try those two. Probably need to run
them against Haiku on Low"*.

| arm | F1 | EM | 2-hop F1 | 3-4 hop F1 | F1 where CB wrong | searches/q | gold recall | $/q | API s/q |
|---|---|---|---|---|---|---|---|---|---|
| CB | 0.263 | 0.177 | 0.330 | 0.196 | 0.054 | 0 | 0 | 0.0005 | 3.1 |
| PS | 0.485 | 0.367 | 0.561 | 0.409 | 0.369 | 0 | 0.431 | 0.0005 | 2.5 |
| AG3 | 0.532 | 0.413 | 0.611 | 0.454 | 0.413 | 1.74 | 0.489 | 0.0013 | 4.4 |
| AG4 | 0.524 | 0.420 | 0.604 | 0.445 | 0.412 | 1.97 | 0.525 | 0.0014 | 4.8 |
| PSAG3 | 0.570 | 0.453 | 0.635 | 0.504 | 0.468 | 1.11 | 0.602 | 0.0012 | 3.8 |

Searches per question in AG3: 0 on 33, 1 on 105, 2 on 69, 3 on 93. On the 33 it
answered without searching, AG3 scored 0.612 and PSAG3 0.656; on the 267 where it
searched, 0.522 and 0.559. The gain is spread over both groups, not confined to the
questions Haiku would otherwise have answered from memory.

Predictions (PLAN.md) against measurement:

- P1 (≤ 2% unparseable, accuracy within 0.03 of 0.846): held on parsing (0/312);
  missed on accuracy by 0.008 (0.808).
- P2 (≥ 5% no answer at the cap, re-ask adds ≥ 1 F1 point): missed. 1 of 300; the
  re-ask added 0.003 F1 to AG3 (the one recovered question; `results/scores.json` `reask`).
- P3 (PSAG3 beats AG3 by ≥ 0.03, CI excluding 0): held at the edge, +0.037 with a
  lower bound of 0.0002.
- P4 (PSAG3 ties AG4): missed. PSAG3 led by 0.045 with a CI excluding 0. I expected
  the extra search to explain the gain; Haiku does not spend the extra search.
- P5 (PS trails AG3 by ≥ 0.05 on 3-4 hop, within 0.03 on 2-hop): missed both ways.
  The gap was 0.045 on 3-4 hop and 0.050 on 2-hop.
- P6 (CB F1 ≤ 0.30): held, 0.263.
- P7 (PSAG3 writes ≥ 0.5 fewer searches than AG4): held, 1.11 vs 1.97.

Reading at the time: pre-search is a cheap default for a small model with a search
tool; the re-ask is not worth carrying into `decide.py`.
