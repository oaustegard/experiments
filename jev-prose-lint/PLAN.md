# Plan: Jev as the judge of declaude_lint hits

Written 2026-11-07, before any label or Jev answer existed.

## Question

`declaude_lint` flags candidate LLM-prose tics by regex and says of its own
output "candidates, not verdicts". Today every hit is read by a person or a
Claude subagent. Can Jev (TypeSafe's decision model, ~0.3 s a call) judge each
hit (real tic, or legitimate use) well enough to sit between the lint and the
rewriting pass in the public-prose review, the way Jev judges code findings in
`verifying-review-findings`?

## Items

`build_items.py`: every non-density lint hit on the 134 published posts of
muninn.austegard.com, final versions and every earlier committed version.
1,029 hits, 824 with a locatable paragraph and a mapped register.md entry. A
seeded sample of 300: all hits removed by a later revision (37 after the
per-category cap) and kept hits stratified by category, capped at 45 per
category. The paragraph the hit sits in is its context.

Bias stated up front: published posts went through review before their first
commit, so most surviving hits were already accepted by a reviewer. Expect a
low tic rate. Drafts, where the step would run, carry more real tics.

## Reference labels

Two independent Claude judges (one Opus, one Sonnet subagent, `[no-context]`)
label every item `tic` or `legitimate` with a confidence, reading the
register.md entry for the hit's category. Gold = items where both agree.
Disagreements are reported, not resolved.

The revision signal (removed vs kept) is a second, weaker reference.

## Jev arms

One noul per item, P(this is the tic, worth rewriting). Model jev-1.14.0,
direct API.

- **A**: state = category, lint note, flagged text, paragraph.
- **B**: A plus the full register.md entry for the category.

Each arm runs twice to measure call-to-call variation (METHODS.md: Jev noul
scores vary between identical calls enough to reorder a ranking near a cut).

## Metrics

Against gold: AUROC of P(tic); accuracy and Cohen's κ at 0.5; precision and
recall of `tic`; accuracy of the majority-class baseline and of a
per-category prior (leave-one-out tic rate of the hit's category). Coverage at
a 5% error budget: the share of hits Jev can settle (P below t_low or above
t_high) with at most 5% of those wrong.

## Predictions

- **P1** The two judges agree on at least 75% of items, κ ≥ 0.5.
- **P2** Gold tic rate on these published-post hits is at most 40%.
- **P3** Arm B AUROC ≥ 0.75, and its accuracy at 0.5 beats the
  majority-class baseline by at least 10 points.
- **P4** Arm B beats arm A by at least 0.03 AUROC.
- **P5** Jev agrees with gold least on `triad` and `cadence` (structural) and
  most on lexical categories (`slop`, `flat-certainty`, `editorializing`).
- **P6** Between repeat calls, mean |ΔP| < 0.05 and fewer than 5% of items
  cross 0.5.
- **P7** Gold tic rate is higher on revision-`removed` hits than on `kept` ones.
- **P8** Coverage at a 5% error budget for arm B is at least 50%.

## Decision rule

Wire Jev into the public-prose review as the lint-hit judge if P3 and P8 hold.
Otherwise report where it fails and leave the review as it is.
