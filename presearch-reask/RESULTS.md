# presearch-reask — results

HyperSkill (arXiv 2611.15178) credits most of its gain to an "execution protocol" of
prompt and decoding options that its baselines did not get, and never ablates those
options one at a time. Two of them had not been measured in this repo before: **question
pre-search** (send the question verbatim to the search tool once before the model
starts, outside its tool budget) and a **final-answer turn** (ask once more when a
rollout ends without an answer). Both were tested on Haiku 5.5, Anthropic's smallest current model,
called through the Claude Code CLI (`claude -p`); round 1 ran at `--effort low`, its lowest reasoning-effort setting, and round 2 at default effort. The model searches by writing a
`<search>` tag; it gets a budget of searches and at most budget + 1 model turns, after which the rollout ends. MuSiQue
questions chain 2 to 4 lookups ("hops"), each needing a different paragraph. The "hub decider" is `scripts/decide.py` in the separate `claude-workspace` repo, which asks Haiku
fixed-form classification questions and parses a JSON reply; it was checked because a re-ask would only
be worth adding there if Haiku's replies sometimes fail to parse.

## Answer

Pre-search helped Haiku with a fixed BM25 index, did not help Haiku research subagents on live web search, and the re-ask is not worth adopting.
On 300 MuSiQue questions over a 14,661-paragraph BM25 index, pre-search plus 3 searches scored F1 0.570 <!-- @evidence[[results/scores.json#arms.PSAG3.all.f1]] -->
It beat a 4-search budget, the same total of 4 searches, by 0.045 F1 (95% CI 0.008 to 0.083). <!-- @evidence[[results/scores.json#pairs.PSAG3-AG4.all.diff]] -->
Haiku barely used a fourth search (1.97 vs 1.74 per question), so the gain was not the extra search; the likely cause, not measured directly, is that seeing results for the whole question first improved the follow-up queries.
With live WebSearch on 100 of the questions, subagents scored F1 0.353 with pre-search and 0.365 without, a difference of -0.012 (CI -0.062 to 0.037). <!-- @evidence[[results/web_scores.json#diff.all.diff]] -->
It cut searches and run time, but the web tool already returns a written summary, so the pre-search hands Haiku a hedged guess more than evidence.
The re-ask fired on at most 1 of 300 questions in any arm, and the hub decider at low effort had 0 unparseable samples out of 312. <!-- @evidence[[results/decider_low.json#unparseable]] -->
A larger web sample could still show a small effect either way; 100 questions only resolve differences of about 0.05 F1.

## Findings

1. With live web search, pre-search changed how Haiku worked but not how often it was right: F1 0.353 vs 0.365 (CI of the difference -0.062 to 0.037), and gold contained in the answer 0.37 vs 0.39 (CI -0.09 to 0.05). It cut model-written searches from 1.77 to 1.31 per question <!-- @evidence[[results/web_scores.json#arms.PSWEB.searches_per_q]] -->
   and run time from 18.4 to 14.2 s; 34 of 100 answered from the pre-search alone, against 26 of 100 that answered from memory with no search at all. Run as a separate Haiku call it raised cost per question from $0.024 to $0.031. (`run/web_score.py`, round 2)
2. Pre-search plus 3 searches beat 3 searches alone by 0.037 F1 (CI 0.000 to 0.075). <!-- @evidence[[results/scores.json#pairs.PSAG3-AG3.all.diff]] -->
   It led 4 searches by 0.045 (CI 0.008 to 0.083). Over 4 searches the lead was +0.031 on 2-hop and +0.059 on 3-4 hop questions; the 3-4 hop CI excludes 0, the 2-hop one does not. (`run/score.py`)
3. Giving the tool a fourth search did nothing: 4 searches vs 3 differ by -0.008 F1. <!-- @evidence[[results/scores.json#pairs.AG4-AG3.all.diff]] -->
   Haiku used 1.97 searches per question when allowed 4 <!-- @evidence[[results/scores.json#arms.AG4.model_searches_per_q]] -->
   and answered with no search at all on 29 of 300. (`results/search_behaviour.json`)
4. With pre-search Haiku wrote 1.11 searches per question <!-- @evidence[[results/scores.json#arms.PSAG3.model_searches_per_q]] -->
   and still reached a gold-paragraph recall of 0.602 <!-- @evidence[[results/scores.json#arms.PSAG3.gold_paragraph_recall]] -->
   against 0.525 for the 4-search arm (recall counts every result shown to the model, pre-search included). <!-- @evidence[[results/scores.json#arms.AG4.gold_paragraph_recall]] -->
   Haiku's own first query found gold recall 0.407, <!-- @evidence[[results/search_behaviour.json#AG4.first_query_gold_recall]] -->
   slightly below 0.431 for the verbatim question, <!-- @evidence[[results/search_behaviour.json#PS_verbatim_gold_recall]] -->
   so the first query explains little of the gap; by elimination the rest arises in the follow-up searches, though what made them better was not measured.
5. The final-answer turn had nothing to do: 1 of 300 questions reached the round cap without an answer in the 3-search arm, 0 in the 4-search arm, and no search past the budget was attempted in any arm. (`results/scores.json` `reask`)
6. The hub decider (`scripts/decide.py`, 104 items, 3 samples each) at low effort parsed all 312 samples; accuracy 0.808 <!-- @evidence[[results/decider_low.json#report.overall.accuracy]] -->
   against 0.846 at default effort; on 104 items one standard error is about 0.04, so the drop is not distinguishable from noise.
7. Retrieval of any kind carries most of the score. Closed book scored 0.263 F1, <!-- @evidence[[results/scores.json#arms.CB.all.f1]] -->
   low enough that most of the questions need the knowledge base; pre-search alone with no tool scored 0.485. <!-- @evidence[[results/scores.json#arms.PS.all.f1]] -->
8. The whole run (1,500 rollouts) cost $1.48 at list price <!-- @evidence[[results/search_behaviour.json#total]] --> and took 29 minutes at 8 parallel calls.

## Method

**Fixture.** MuSiQue-Answerable dev (HF `bdsaglam/musique`), 300 questions drawn with
seed 0: 150 2-hop, 100 3-hop, 50 4-hop. The knowledge base is every paragraph of the
first 1,000 dev questions plus the 300 (14,661 paragraphs after deduplication; each
question's 20 MuSiQue paragraphs, 2 to 4 supporting and the rest distractors, are in it). Search is plain BM25 (k1 1.2,
b 0.75) over `title text`, top 5 per query. HotpotQA was not used because METHODS.md
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
hop count and on questions closed book gets wrong (F1 < 0.5). Gold-paragraph recall is the
share of a question's supporting paragraphs that appeared in any result shown to
the model.

**Round 2 (web).** 100 of the 300 questions (the first 50 2-hop, 33 3-hop and 17 4-hop in
`data/eval.jsonl`), run as `claude -p --model haiku` at default effort with WebSearch and WebFetch
allowed, no search limit and at most 16 turns (`run/web_arms.py`). WEB gets the question; PSWEB also
gets the raw tool result of one WebSearch on the question verbatim, made first by a separate Haiku call
that stands in for the orchestrator, whose cost is added to the arm. Tool results were scanned for
MuSiQue, Hugging Face dataset pages and the dataset's field names; 1 question matched, and dropping it
changes nothing. A second metric, gold contained in the answer, guards against F1 penalising longer
answers.

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

### Round 2 — 2026-11-14, research subagents with live web search

Asked: Oskar, *"So this seems worth reaching for especially when using haiku sub agents for research, etc?"*,
then *"Ok, run that test"*.

| arm | F1 | EM | gold in answer | 2-hop F1 | 3-4 hop F1 | searches/q | fetches/q | no tool use | $/q | wall s/q |
|---|---|---|---|---|---|---|---|---|---|---|
| WEB | 0.365 | 0.24 | 0.39 | 0.442 | 0.288 | 1.77 | 0.44 | 26% | 0.024 | 18.4 |
| PSWEB | 0.353 | 0.20 | 0.37 | 0.430 | 0.277 | 1.31 | 0.33 | 34% | 0.031 | 14.2 |

Paired difference PSWEB - WEB: F1 -0.012 (CI -0.062 to 0.037); 2-hop -0.012, 3-4 hop -0.012, both CIs
spanning about ±0.07. On the 34 questions PSWEB answered without further tool use it scored 0.478 vs 0.425
for WEB. Where the two arms differed by more than 0.3 F1, WEB won 7 and PSWEB 6; several PSWEB losses
were longer answers ("Stanley, North Dakota" for "Stanley"), and PSWEB answers averaged 6.5 tokens vs 5.6.
The WebSearch tool result is a list of links followed by a written summary that often names a likely
answer, so the pre-search block is closer to a hedged guess than to the BM25 paragraphs of round 1.

On these same 100 questions round 1 had pre-search plus 3 searches at 0.588 and 3 searches alone at
0.583, so this subset showed little of round 1's effect even with BM25; 100 questions cannot resolve a
difference of a few points. Web search scored lower than the closed index (0.365 vs 0.583 for the
plain-tool arms) because the closed index holds every needed paragraph and the web returns noisier pages.

Predictions: W1 (PSWEB ≥ +0.03) missed, -0.012. W2 (WEB ≤ 2.5 searches, ≥ 10% with none) held, 1.77 and
26%. W3 (fewer searches with pre-search) held, 1.31 vs 1.77. W4 (no extra cost) missed, $0.031 vs $0.024,
because the pre-search ran as its own Haiku call. W5 (≥ 5% leakage) missed: 0 pre-search results and 1
agent result matched. Total cost $5.47, 12.5 minutes at 6 parallel runs.

Reading at the time: do not add pre-search to Haiku research subagents on this evidence. It saves time
and searches but buys no accuracy, and the orchestrator pays for the extra search.

