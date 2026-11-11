# haiku-decider: can Claude Haiku stand in for a decision model, and how should it be prompted?

`deciding-with-confidence` is a skill in `oaustegard/claude-skills` that makes Claude Haiku 5.5 answer typed
questions (yes/no, pick one option, a level on a rubric) with a probability for each option, for people who
cannot use a purpose-built decision model such as TypeSafe's Jev. The questions here: which prompt and which
number of calls give the best answers, and how close does the result come to Jev on the same items and on a
public benchmark? This directory keeps the labelled sets, the runners and every raw distribution.

## Answer

Haiku is a usable stand-in, run as a stack of three calls when the confidence will decide anything. With every call quoting the words of the evidence that decide each question before
it answers (the "cue"), a three-call stack scored 0.7465 <!-- @evidence[[results/td/summary.json#arms.stack.accuracy]] -->
on the 2,000 Typed Decisions test decisions, zero-shot, where the benchmark's Hugging Face dataset card lists
Jev 1.13.0 at 0.727; one call
scored 0.741 <!-- @evidence[[results/td/summary.json#arms.one.accuracy]] -->. On the same
labelled article sections the stack scored 0.906 <!-- @evidence[[results/jev_same_items.json#sections.stack.acc]] -->
against Jev's 0.894 <!-- @evidence[[results/jev_same_items.json#sections.jev.acc]] --> (a tie within the
interval); one call trailed Jev on sections and was level on whole articles.

The cue was the largest prompt effect measured: on held-out article sections it moved the stack from 0.870 to
0.906, and on whole articles one call from 0.931 to 0.969. Haiku is slower: a stacked request (all questions
about one case) took a median 8.8 s through `claude -p`, process start-up included, against 0.71 s per case for
Jev on the card.

Labelled sets from more domains would change this before anything else: the sections and articles are
life-science text labelled by Claude models, and the benchmark's gold is one 4B-class teacher.

## Findings

1. **On Typed Decisions the stack scores 0.7465 and one call 0.741,
   zero-shot**, against 0.727 for Jev 1.13.0. The gold is a teacher model's averaged answer, and a fresh
   teacher sample agrees with it 0.735 of the time, so scores near that are at the benchmark's ceiling. Ranked
   by accuracy among the card's 43 general models scored zero-shot (models never fitted on the benchmark), the
   stack would be 10th and one call 13th; Jev is 18th. The 0.0055 between stack and one call is 11 decisions. Unanimous stack
   answers agree with gold 0.830 of the time, split ones 0.522
   (`results/td/summary.json`, Round 3).
2. **On the same labelled answers the stack is level with Jev; one call trails it on sections.** Sections
   (1,457 answers): stack 0.906, one call 0.870, Jev 0.894; stack minus Jev 90% interval −0.002 to +0.026, one
   call minus Jev −0.041 to −0.008. Articles (289): one call 0.969, stack
   0.962, Jev 0.972 (`results/jev_same_items.json`, Round 3).
3. **Quoting the deciding evidence first beat every other instruction tried, on accuracy.** On the dev half, cue
   alone averaged 0.902 against 0.868 for no added instruction (`plain`); literal, sceptic, contrast, strict and
   hybrid reached 0.865 to 0.889, though strict and cue + sceptic separated right from wrong better (AUROC 0.861
   and 0.856 against 0.829). On the
   held-out half: one call 0.808 / 0.931 / 0.857 → 0.827 / 0.969 / 0.870, stack 0.769 / 0.924 / 0.870 →
   0.865 / 0.962 / 0.906 (short items / articles / sections; the short-items test half is 52 items, so 0.019 is one item)
   (`results/prompts/compare_v030stack_test.json`, Round 2).
4. **A pool of three needs its own temperature.** A temperature T rescales a distribution as p^(1/T); T below 1
   sharpens it. Haiku states probabilities lower than its accuracy, and
   averaging three flattens them further: fitted temperatures 0.44 / 0.52 / 0.50 for pooled predicate / choice /
   score answers against 0.74 / 0.60 / 0.81 for one call; the skill's `assets/calibration.json` carries both
   sets. Fitted on one half of the items and scored on the other, both reach ECE 0.013 to 0.015 on articles and
   sections, Jev's range (`results/jev_same_items.json`, Round 2).
5. **On soft gold the calibration trades KL for ECE.** On Typed Decisions the shipped temperatures cut the
   stack's ECE from 0.113 to 0.035 and raise KL from gold from
   0.131 to 0.333; the raw pool's Brier,
   0.069, ties for third on the card (best 0.052), while the calibrated one is 0.121. Read `diagnostics.raw` when the
   target is a soft distribution; use the calibrated answer when it is a hard label (`results/td/summary.json`, Round 3).
6. **Agreement between the three calls marks the errors** on every set: unanimous answers 0.83 to 0.98 right,
   split ones 0.52 to 0.73, across the held-out halves and Typed Decisions; 7% to 27% of answers split. Escalating
   split answers and those under 0.7 sent 19% of section answers onward and left 3.0% wrong (Rounds 2 and 3).
7. **Low effort cost accuracy on article sections only.** One plain call at default against low effort (short
   items, articles, sections): 0.865 / 0.885, 0.908 / 0.908, 0.871 / 0.846 (`results/summary_v020.json`, Round 1).

## Method

Sets (the Findings call them short items, articles and sections). `bundled`: the skill's `assets/eval.jsonl`, 104 items (64 banking77 card intents, 24 guardrail yes/no
questions, 16 rubric scores). `papers` (`data/papers.jsonl.gz`): 150 CC BY Europe PMC articles, each read whole
(median 3,850 words), four questions, 554 labelled answers. `sections` (`data/sections.jsonl.gz`): 395 article
sections, eight questions, 2,834 labelled answers. A papers or sections question is labelled where Sonnet 5.5 and
Opus 5.5, labelling independently, agreed. Items split into dev and test halves by a hash of their id
(`run/prompts.py split_of`); prompts were chosen on dev and reported on test. Typed Decisions
(`LocalLLaMA/typed-decisions`, copied in `browser-decider/data/td/`): 400 synthetic cases from four workflows,
five questions each; gold is the mean of three samples from a teacher of roughly 4B-class capability. The
dataset card (its README on Hugging Face) carries a leaderboard; Jev is TypeSafe's hosted decision model, and its
API body, called System One, is `state` plus questions of type `noul` (yes/no, called `predicate` here),
`choice` and `score`. Several leaderboard models are open-weight imitations of Jev.

Arms. Every call is Haiku 5.5 through `claude -p` with no tools, at low effort unless stated. One call is the
first sample of a stacked request (plain angle, options in listed order). The stack is three parallel samples
under three angles (extra instructions: `plain` adds none, `literal` counts only what the evidence itself states,
`sceptic` starts from the least committal answer) with options shuffled, averaged; unanimous means all three
calls chose the same top option. Round 2 scored plain, literal and
sceptic as single runs on every item and the candidates on the dev half, and simulated stacks by averaging them
(`run/prompts.py`); the "before" stack on test is that simulation; the held-out test ran decide.py's own
`--stack` (`run/compare.py`). Typed Decisions ran as System One requests through decide.py (`run/td.py`) and is
scored with a copy of the dataset's own scorer. Jev's distributions for papers and sections come from
`dettin-68m/data/` (`run/jev_compare.py`).

Metrics. Accuracy of the top option; ECE in 10 bins of the top probability; AUROC of the top probability for
right against wrong; Brier over options; on Typed Decisions, KL(gold ‖ prediction). Intervals are
paired-bootstrap 90% intervals over answers; answers from one article or section are not independent, so they are
somewhat too narrow.

## Log

### Round 1 (2026-11-10)

Oskar asked for the skill to default to one low-effort call, with stacking as an option, and for the test data
to be kept for later prompt iteration. Two runs per set (`run/arms.py`, `results/summary_v020.json`): `old`, three
plain samples at the CLI's default effort (the 0.1 default), and `stack`, three samples at low effort under the
plain, literal and sceptic angles. The sections runs used a seeded sample of 150 (1,070 answers).

| accuracy | bundled | papers | sections |
|---|---|---|---|
| one call, low effort, plain | 0.885 | 0.908 | 0.846 |
| one call, low effort, sceptic | 0.865 | 0.931 | 0.858 |
| one call, default effort, plain | 0.865 | 0.908 | 0.871 |
| stack, low effort | 0.846 | 0.935 | 0.866 |
| three plain calls, default effort | 0.846 | 0.944 | 0.886 |

The rule set before the run, keep `plain` as the single-call angle unless `sceptic` was at least as accurate on
all three sets, kept `plain`. The stack's unanimous answers were right 0.912 / 0.964 / 0.939 and split ones
0.385 / 0.660 / 0.569. Cost per request at list price, stack then old: $0.0010 and $0.0012, $0.0055 and $0.0076,
$0.0027 and $0.0034. Shipped as deciding-with-confidence 0.2.0.

### Round 2 (2026-11-11): prompt iteration

Oskar asked to iterate on the prompting for the single and stacked modes. Each candidate angle ran as one
low-effort call per item over all items (baseline) or the dev half (candidates), checkpointed per item
(`results/prompts/<set>__<angle>.jsonl`); stacks were scored by averaging per-angle runs.

| dev half, mean over three sets | accuracy | AUROC | Brier |
|---|---|---|---|
| plain | 0.868 | 0.816 | 0.194 |
| contrast (decide between the two best options) | 0.886 | 0.832 | 0.180 |
| strict (each description is a test) | 0.865 | 0.861 | 0.188 |
| cue (quote the deciding words first) | 0.902 | 0.829 | 0.160 |
| cue + sceptic | 0.899 | 0.856 | 0.151 |
| stack plain + literal + sceptic | 0.881 | 0.854 | 0.177 |
| stack, each of those with the cue | 0.912 | 0.885 | 0.142 |

The cue went into the system prompt's answer format, so every sample writes it; the stack kept its angles. The
test half then ran decide.py's own `--stack` (`results/prompts/<set>__v030stack.jsonl`; a first launch was
stopped at 130 answers and discarded because the prompt wording changed mid-run). Held-out accuracy is in
finding 3. The stack's pools were further from calibrated (raw ECE 0.06 / 0.15 / 0.15 for predicate / choice / score
answers) than single calls, so
`calibration.json` gained a pooled temperature set, fitted on all 1,798 test answers. Shipped as 0.3.0.

### Round 3 (2026-11-11): Jev on the same items, Typed Decisions, System One requests

Jev's saved distributions from `dettin-68m` cover the same papers and sections, so `run/jev_compare.py` scores
them on the test-half answers the cue stack covered.

| test half | answers | Jev | one call | stack | old one call |
|---|---|---|---|---|---|
| papers | 289 | 0.972 | 0.969 | 0.962 | 0.931 |
| sections | 1,457 | 0.894 | 0.870 | 0.906 | 0.857 |

One call minus Jev on sections: −0.041 to −0.008. Uncalibrated ECE: Jev 0.023 / 0.013, stack 0.091 / 0.113;
cross-validated calibration brings the stack to 0.013 / 0.014. The bundled dev half was run with the cue stack
too, so all 104 bundled items have it: one call 0.856 <!-- @evidence[[results/jev_same_items.json#bundled.one.acc]] -->,
stack 0.894 <!-- @evidence[[results/jev_same_items.json#bundled.stack.acc]] -->, against Jev's
0.894 on the same 104 items from 2026-10-09. Half of these items chose the prompt in Round 2, so the tie leans
optimistic.

deciding-with-confidence 0.4.0 accepts a System One request (`state` plus `noul` / `choice` / `score`
questions keyed by id, with criteria), the body Jev and its clones take, and answers in that shape.
`run/td.py` sent all 400 Typed Decisions test cases through it unchanged, as `--stack` at low effort.

| Typed Decisions test, 2,000 decisions | accuracy | KL from gold | Brier | ECE |
|---|---|---|---|---|
| stack (shipped calibration) | 0.7465 | 0.333 | 0.121 | 0.035 |
| stack, uncalibrated | 0.7465 | 0.131 | 0.069 | 0.113 |
| one call (shipped calibration) | 0.7410 | 0.241 | 0.101 | 0.039 |
| one call, uncalibrated | 0.7410 | 0.163 | 0.079 | 0.096 |
| TypeSafe Jev 1.13.0 (card) | 0.727 | 1.442 | 0.148 | 0.144 |
| meraGPT Decider 1 (card) | 0.768 | 0.096 | 0.052 | 0.180 |
| Quyet-1.0-Large, 31B (card, first) | 0.791 | 0.281 | 0.114 | 0.093 |

By workflow (stack): agent trace observability 0.668, customer service 0.792, invoice processing 0.814, security incidents 0.712. By type (stack): choice 0.737, noul 0.810, score 0.706. Mean cost per
stacked case $0.0023 at list price; median latency 8.8 s.
The calibration temperatures were fitted on life-science and support text, never on this benchmark.
`results/td/leaderboard.png` places both arms among the card's 43 general zero-shot rows.
