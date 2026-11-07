# Jev as the judge of prose-lint hits

Muninn (Oskar Austegard's Claude-based assistant) publishes a blog at
muninn.austegard.com. Before a post goes out, `declaude_lint`, a regex lint
from the `declauding` skill, flags constructions typical of machine-written
prose: forced triads, participle tails, certainty adverbs. Each of its 56 tic
categories is defined in the skill's `register.md`. The lint calls its output
candidates, not verdicts, so a person or a Claude subagent reads every hit
before anything is rewritten. Jev is a commercial decision model from TypeSafe
that answers a yes/no question with a probability in about 0.2 s. This
experiment asks whether Jev can judge each hit (real tic, or legitimate use)
well enough to take that reading off the review.

## Answer

No. Two Claude judges (Opus and Sonnet) labelled 300 lint hits from published
posts and agreed on 87% of them; <!-- @evidence[[results/metrics.json#judges.agreement]] -->
the 261 hits they agree on are the reference labels. Given the tic's
definition, Jev ranks hits at AUROC 0.808. <!-- @evidence[[results/metrics.json#jev_B.auroc]] -->
Knowing only each hit's category, and how often hits in that category are real
tics, gives 0.769, <!-- @evidence[[results/metrics.json#baselines.category_prior_auroc]] -->
and the bootstrap 95% interval on the difference runs from -0.025 to 0.106, so
Jev's edge is not established. <!-- @evidence[[results/ci.json#auroc_diff_jevB_minus_prior.ci95.0]] -->
Jev's probabilities also cluster between 0.3 and 0.8. With at most 5% of its
calls wrong, it can settle only 24.5% of hits; <!-- @evidence[[results/metrics.json#jev_B.coverage@5%.coverage]] -->
the plan required 50% before wiring it in.

The category rates themselves are the useful result. On reviewed posts the
triad rule is right on 5 of 44 hits, <!-- @evidence[[results/ci.json#gold_tic_rate_by_category.triad]] -->
the participle rule on 5 of 43, <!-- @evidence[[results/ci.json#gold_tic_rate_by_category.participle]] -->
and the negation-first rule on 16 of 18. <!-- @evidence[[results/ci.json#gold_tic_rate_by_category.negation-first]] -->
Retuning or demoting the low-precision rules would cut the reading more than
Jev does. Labels from drafts, which carry more real tics than reviewed posts,
could change both answers.

## Findings

1. Jev with the tic's `register.md` entry in its input (arm B) ranks hits at
   AUROC 0.808 (`jev_judge.py`). <!-- @evidence[[results/metrics.json#jev_B.auroc]] -->
   Its accuracy at a 0.5 cut is 0.732, <!-- @evidence[[results/metrics.json#/jev_B/accuracy@0.5]] -->
   below the 0.751 that the category's tic rate alone reaches. <!-- @evidence[[results/metrics.json#baselines.category_prior_accuracy]] -->
   Jev's AUROC lead over that prior is 0.038, <!-- @evidence[[results/ci.json#auroc_diff_jevB_minus_prior.point]] -->
   with a bootstrap 95% interval from -0.025 to 0.106 (`ci.py`). <!-- @evidence[[results/ci.json#auroc_diff_jevB_minus_prior.ci95.1]] -->
2. With at most 5% of settled hits wrong, Jev settles 24.5% of hits <!-- @evidence[[results/metrics.json#jev_B.coverage@5%.coverage]] -->
   (`analyze.py`): P ≤ 0.205 counts as legitimate and P ≥ 0.795 as a tic,
   thresholds chosen on the same 261 hits. The pre-registered bar was 50%.
3. The lint's precision varies by rule (`ci.py`). Of the gold hits, real tics
   are 5 of 44 triads, 5 of 43 participle tails, 0 of 10 juridical verbs and
   0 of 9 verification compounds. They are 16 of 18 negation-first openers
   and 7 of 11 one-line paragraphs. <!-- @evidence[[results/ci.json#gold_tic_rate_by_category.juridical]] -->
   Overall, 35.2% of hits are real tics. <!-- @evidence[[results/metrics.json#gold.tic_rate]] -->
4. The definition is what makes Jev work (`jev_judge.py` arms A and B).
   Without the `register.md` entry (arm A), AUROC is 0.697 <!-- @evidence[[results/metrics.json#jev_A.auroc]] -->
   and recall of real tics is 0.207, <!-- @evidence[[results/metrics.json#jev_A.recall_tic]] -->
   against 0.641 with the entry. <!-- @evidence[[results/metrics.json#jev_B.recall_tic]] -->
5. Jev's errors follow the rule (`analyze.py`). On the 10 juridical hits, all
   legitimate in gold, Jev gives P between 0.43 and 0.79 and is right on 1. <!-- @evidence[[results/metrics.json#jev_B.accuracy_by_category.juridical]] -->
   It is right on 39 of 44 triads, where the right answer is almost always "legitimate". <!-- @evidence[[results/metrics.json#jev_B.accuracy_by_category.triad]] -->
6. A cross-validated logistic regression on Jev's P and the category prior
   (post hoc, `combo.py`) reaches AUROC 0.845 <!-- @evidence[[results/combo.json#combined.auroc]] -->
   but settles only 2.7% of hits at 5% error. <!-- @evidence[[results/combo.json#combined.coverage_at_5pct]] -->
7. Jev is stable and fast (`jev_judge.py`, two runs per arm). The mean change
   in P between identical calls is 0.018, <!-- @evidence[[results/metrics.json#jev_B.repeat_mean_abs_delta]] -->
   with 7 of 300 hits crossing 0.5, <!-- @evidence[[results/metrics.json#/jev_B/repeat_cross_0.5]] -->
   and the median call takes 161 ms. <!-- @evidence[[results/latency.json#B.median_ms]] -->
8. Git history gives no label signal (`build_items.py`). Hits that a later
   revision removed are real tics 12 of 36 times, <!-- @evidence[[results/metrics.json#gold.tic_rate_by_revision.removed]] -->
   against 80 of 225 for hits still in the final posts. <!-- @evidence[[results/metrics.json#gold.tic_rate_by_revision.kept]] -->

## Method

**Items.** `build_items.py` runs `declaude_lint` over the final version and
every earlier committed version of the 134 published blog posts on
muninn.austegard.com. It finds 1,029 hits, excluding the whole-document
density hits. 824 of them have a locatable paragraph and a category that maps
to a `register.md` entry. A seeded sample of 300 includes every hit that a
later revision removed (37 after a cap of 45 hits per category; 36 of them are in
gold) and kept hits
stratified by category. Each item carries the hit's category, the lint's note,
the flagged text, and the paragraph it sits in.

**Gold.** Four Opus and four Sonnet subagents, each with 75 items and the
prompt marker `[no-context]` (which stops a hook from appending this
session's transcript to the prompt), labelled every hit `tic` or `legitimate`. They
read the `register.md` entry for the hit's category. One Sonnet batch
reported that it skimmed the entry headings without reading the entries. Gold
is the 261 items where Opus and Sonnet agree; the 39 disagreements are left
out. Dropping them removes the hardest items, which flatters every arm, Jev
and baselines alike. The skimming batch's effect was not measured
separately.

**Jev.** One yes/no question per hit, asking for P(this is a real tic that a
careful technical writer would rewrite), sent to jev-1.14.0 through the direct
API. Arm A's state holds the category, the lint note, the flagged text and the
paragraph. Arm B adds the full `register.md` entry. Each arm ran twice, and
the reported P is the mean of the two runs.

**Metrics.** AUROC; accuracy and κ at 0.5; precision and recall of `tic`;
coverage at a 5% error budget, meaning the largest share of hits that two
thresholds settle with at most 5% of them wrong. Baselines: the majority
class, and the leave-one-out tic rate of the hit's category.

**Cost.** 1,200 Jev calls at about 500 to 800 input tokens each, with a
161 ms median. The eight judge subagents used about 90,000 to 100,000 tokens
each. Wall clock was about 40 minutes.

**Prior art.** Semgrep, Datadog and ZeroFalse put an LLM pass over
static-analysis alerts to drop false positives, and each uses a frontier
model. Two searches found no published evaluation of judging style-lint hits,
and none that uses a small decision model.

## Log

### 2026-11-07: first and only round

Asked by Oskar after `verifying-review-findings` shipped: should Jev join the
public-prose review? `PLAN.md` set eight predictions and the decision rule
(wire Jev in if P3 and P8 hold) before any label existed.

| Prediction | Measured | Verdict |
|---|---|---|
| P1 judges agree ≥ 75%, κ ≥ 0.5 | 87%, κ 0.723 | right |
| P2 gold tic rate ≤ 40% | 35.2% | right |
| P3 arm B AUROC ≥ 0.75 and accuracy ≥ majority + 10 points | 0.808; 0.732 against majority 0.648 (+8.4) | half |
| P4 arm B beats A by ≥ 0.03 AUROC | +0.111 | right |
| P5 worst on triad and cadence, best on lexical categories | best on triad (39/44), worst on juridical (1/10) | wrong |
| P6 repeat mean abs ΔP < 0.05, < 5% cross 0.5 | 0.018, 2.3% | right |
| P7 removed hits richer in tics | 33% against 36% | wrong |
| P8 arm B coverage at 5% error ≥ 50% | 24.5% | wrong |

P3 fell short and P8 failed, so under the plan's rule Jev is not wired into
the public-prose review. The category prior was not among the predictions and
turned out to be the comparison that matters: it ranks hits nearly as well as
Jev and beats Jev's accuracy at 0.5.

The P5 miss reads as Jev reacting to the surface of the flagged word. Every
`juridical` hit flags a legal-sounding verb, and Jev scores the word, where
the judges read whether the verb was the precise one in that sentence. On
`cadence` the flagged text is the whole one-line paragraph, which gives Jev
little surface to react to.
