# haiku-decider: how should a Haiku-based decision harness be run?

`deciding-with-confidence` is a skill in `oaustegard/claude-skills` that makes Claude Haiku 5.5 answer typed
questions (yes/no, pick one option, a level on a rubric) with a probability for each option, for people who have
no purpose-built decision model such as Jev. Until 0.2.0 it ran three calls per request at the CLI's default
effort and averaged them. The question here: what does one call at low effort cost in accuracy, and does a
stack of three calls with differently worded instructions earn its price? This directory also keeps the
labelled sets for the next round of prompt changes.

## Answer

One low-effort call is a third to a quarter of the cost and loses 0.02 to 0.04 accuracy on long scientific text.
The stack's worth is its agreement signal, not its accuracy.

Accuracy on three labelled sets (short support items, whole research articles, article sections): one
low-effort call 0.885 <!-- @evidence[[results/summary_v020.json#bundled.1 sample, low effort, plain.accuracy]] -->,
0.908 <!-- @evidence[[results/summary_v020.json#papers.1 sample, low effort, plain.accuracy]] --> and
0.846 <!-- @evidence[[results/summary_v020.json#sections.1 sample, low effort, plain.accuracy]] -->; the old
default 0.846, 0.944 and 0.886 <!-- @evidence[[results/summary_v020.json#sections.old: 3 samples, default effort.accuracy]] -->.
A stack of three angles at low effort scores 0.846, 0.935 and 0.866 <!-- @evidence[[results/summary_v020.json#sections.stack: plain + literal + sceptic, low effort.accuracy]] -->.
Where its three calls agree the answer is right 0.939 <!-- @evidence[[results/summary_v020.json#sections.stack, unanimous.accuracy]] -->
of the time on sections; where they split, 0.569 <!-- @evidence[[results/summary_v020.json#sections.stack, split.accuracy]] -->.

More labelled sets from other domains would change the default before anything else would: two of the three
here are life-science text.

## Findings

1. **One low-effort call beats the old default on short items and loses on long text.** 0.885 against 0.846 on
   104 short items; 0.908 against 0.944 on 554 article questions; 0.846 against 0.886 on 1,070 section
   questions (`results/summary_v020.json`).
2. **Agreement between stacked calls marks the errors.** Unanimous against split: 0.912 and 0.385 on the short
   items, 0.964 and 0.660 on articles, 0.939 and 0.569 on sections. Splits are 13%, 10% and 20% of answers.
3. **A single call's stated probability is a weaker signal.** Its AUROC for telling right from wrong is 0.81,
   0.88 and 0.79 on the three sets, against 0.89, 0.87 and 0.85 for the stack.
4. **The "sceptic" instruction helps on long text and not on short items.** One low-effort call: 0.931 against
   0.908 on articles and 0.858 against 0.846 on sections, but 0.865 against 0.885 on the short items (two
   items). `plain` stays the default angle.
5. **Low effort costs accuracy only on sections.** One plain call at default against low effort: 0.865 and
   0.885, 0.908 and 0.908, 0.871 and 0.846.
6. **The stack at low effort is slightly below three plain calls at default effort** on long text (0.935
   against 0.944, 0.866 against 0.886) at about 75% of the cost.

## Method

Sets, in `decide.py eval` format. `bundled`: the skill's `assets/eval.jsonl`, 104 items. `papers`
(`data/papers.jsonl.gz`): 150 CC BY Europe PMC articles, each rebuilt whole from its parsed sections (median
3,850 words), four questions (study type, field, human samples, immunoassay), 554 labelled answers. `sections`
(`data/sections.jsonl.gz`): 395 article sections with eight questions, 2,834 labelled answers; the runs used a
seeded sample of 150 (1,070 answers). A question is labelled where Sonnet 5.5 and Opus 5.5, labelling
independently, agreed; a section's article-part label comes from its heading. Both sets come from `dettin-68m`.

Runs (`run/arms.py`), through `claude -p` with no tools: `old` is three plain samples at the CLI's default
effort; `stack` is three samples at low effort, one each under the plain, literal and sceptic angles. Single-call
rows are the individual samples of those runs, so the single calls differ from a real one-sample run only in
option order. Metrics are on the uncalibrated distributions: accuracy of the top option, ECE in 10 bins of the
top probability, and AUROC of the top probability for right against wrong.

Cost per request at list price, stack then old: $0.0010 and $0.0012 (short items), $0.0055 and $0.0076
(articles), $0.0027 and $0.0034 (sections). The whole comparison cost about $3.

## Log

### Round 1 (2026-11-10)

Oskar asked for the skill to default to one low-effort call, with stacking as an option, and for the test data
to be kept for later prompt iteration. The rule set before the run: keep `plain` as the single-call angle unless
`sceptic` was at least as accurate on all three sets. It was not (finding 4).

The shipped temperature for choice questions (0.621, fitted on three-sample pools in October) also fits single
calls: choice ECE falls from 0.128 to 0.072 on the short items, 0.084 to 0.050 on articles and 0.083 to 0.073 on
sections. Predicates fit near 1 on both new sets.

An earlier hard-label test in `dettin-68m` (`results/paper.json`), one call per article with a JSON schema and no
probabilities, scored plain 0.933 and sceptic 0.963 on the same articles, and Jev 0.965. The numbers here are
lower for the plain call because the questions are answered as distributions and scored on agreed items only.
