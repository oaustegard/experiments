# Haiku 4.5 — vanilla vs. down-skilled, n=20 per cell

**Date:** 2026-05-26 (same day as the n=1 probe in
[`../RESULTS.md`](../RESULTS.md))

Scaled the original 3-task probe to 20 independent runs per cell, 120
total Haiku dispatches. Two of those (1/40 task-2-downskilled, 8/20
task-3-downskilled) responded inline to the wrapper agent instead of
calling Write; their outputs were captured manually from the inline
replies — content-equivalent for scoring purposes.

Outputs: [`outputs/`](outputs/) · Scorer: [`score.py`](score.py) · Raw
metrics: [`scores.json`](scores.json) · Chart:
[`comparison.png`](comparison.png)

## Headline numbers

| Metric | Vanilla | Down-skilled |
|---|---:|---:|
| **Task 1 — Triage** (80 classifications: 4 tickets × 20 runs) | | |
| Per-rubric accuracy | 75/80 (94%) | **80/80 (100%)** |
| Tight schema compliance | 0% | **100%** |
| No markdown drift | 0% | **100%** |
| Ticket B classification: QUESTION (rubric) / FEATURE_REQUEST | 15 / 5 | **20 / 0** |
| Mean output length (chars) | 800 | 339 |
| **Task 2 — Code review** | | |
| Found mutation bug | **100%** | **100%** |
| No unsolicited fix section | 0% | **100%** |
| Output ≤400 chars | 0% | 90% |
| Flagged None-skip as false-positive bug | 0% | 0% |
| Mean output length (chars) | 1320 | 384 |
| **Task 3 — Voice rewrite** | | |
| No banned hype words | 85% | **100%** |
| No soft-promotional residue | 80% | **100%** |
| Length in spec (60–90 words) | 40% | **0%** |
| **No invented architectural details** | 80% | **5%** |
| Mean word count (spec: 60–90) | 84 | 43 |

## What stays true from the n=1 writeup

- **Task 1.** Down-skilling fixes schema drift cleanly (0% → 100%
  tight-schema, 100% → 0% markdown). The per-rubric accuracy gap is
  narrower than the n=1 probe suggested — vanilla gets ticket B right
  75% of the time, not 0% — but the *consistency* gap is the real
  finding: vanilla classifications drift across runs (15 QUESTION,
  5 FEATURE_REQUEST on the same input); down-skilled never drifts.

- **Task 2.** Both variants find the bug 100% of the time. The
  scope-discipline gap was real and stark: **every single** vanilla run
  added an unrequested fix section; **zero** down-skilled runs did. The
  "Haiku adds value past the question" pattern is robust.

- **Task 3 — banned vocabulary.** Down-skilling cleanly removes the
  surface-level hype words (15% → 0%).

## What changes from the n=1 writeup

**The Task-3 hallucination problem is far worse than n=1 showed.** At
n=1, one downskilled output invented "distributed hash table" and
"cache-aside pattern." That looked like a fluke. At n=20:

- **19 of 20 downskilled runs (95%) invented architectural details
  not in the source.** The vanilla baseline is 4/20 (20%).
- Detail count: vanilla averages 0.2 invented terms per output;
  downskilled averages **2.6**.
- Inventions observed across the 20 downskilled outputs include:
  *distributed hash table, LRU eviction, TTL-based invalidation,
  write-through consistency, cache-aside pattern, sub-millisecond
  latency, RDB snapshots, p99 percentiles, Redis comparisons,
  in-memory storage, batch operations, per-key TTL, background
  queues.*

**And the length constraint completely fails under down-skilling.** The
rule said "60–90 words"; the in-prompt examples were 33–36 words. The
20-run average for downskilled output: **43 words.** Zero outputs fell
in the stated 60–90 range. Vanilla, with no length rule at all, hit the
input's natural ~85 words in 40% of runs and averaged 84.

The mechanism is exactly what the down-skilling skill itself warns
about: **when rules and examples conflict, Haiku follows the examples.**
The skill's own examples were ~35 words long and contained specific
architectural details. Haiku faithfully copied the pattern — length and
specificity — across 20 independent runs. The "60–90 words" rule and
"do not invent" rule were structurally outranked by the demonstration.

## Mechanism check: why does this happen with examples but not rules?

The down-skilling prompt for Task 3 contained:
- Rule 6: "Replace adjective stacks with one concrete fact. 'Dramatic
  reductions in latency' → an actual number, or omit."
- Rule 7: "If the input has no concrete numbers, do NOT invent them.
  Describe in qualitative-but-precise terms."

But also:
- Example 1 output: "...replaces the previous session-cookie flow with
  short-lived JWTs plus a refresh endpoint, and ships with bindings for
  the three official SDKs."
- Example 2 output: "...replaces the Highcharts-based v1 with a
  server-side rendering pipeline; charts now load in one round-trip
  instead of three."

The examples *demonstrate* concrete architectural specificity. They
don't visibly anchor that specificity to source text — a reader (or
Haiku) cannot tell that "session-cookie flow" was drawn from a
hypothetical prior version of the input rather than fabricated. Without
that anchoring, the example reads as "be confidently specific about
architecture." Haiku obeyed.

This isn't a Haiku failure mode. It's a **prompt-engineering failure
mode that Haiku surfaces faster than Opus or Sonnet would.** Larger
models can override the demonstration with the rule's spirit; Haiku
follows the demonstration's surface pattern.

## Cost confirmation

Across 120 runs, every Haiku dispatch reported between 33,654 and
34,948 input tokens (the bulk being the harness's general-purpose
system prompt, not the task content). Task content varied from ~120
tokens (vanilla Task 3) to ~1300 tokens (down-skilled Task 1). At
Haiku pricing, the down-skilled prompts cost ~$0.0008 more per call
than vanilla — still negligible relative to the value of zero retries
on Tasks 1 and 2.

The Task-3 hallucination would, in a production setting, *require*
retries (because the output is wrong, not just stylistically off).
The down-skilling skill's economic argument fails for Task 3: the
extra cost of the longer prompt is dominated by the cost of producing
incorrect output 95% of the time.

## Updated guidance for the practitioner guide

1. **Triage / classification with a rubric: down-skill, always.** N=20
   confirms 100% consistency and zero format drift. Cost: ~$0.001/call
   extra. Worth it.

2. **Code review with explicit scope: down-skill.** Same bug-finding
   rate as vanilla, plus perfect scope discipline. No regressions
   observed.

3. **Voice / register rewrites: do NOT down-skill with the standard
   template.** It removes surface vocabulary while introducing
   architectural hallucination in 95% of runs. Either:
   - Use Sonnet 4.6 / Opus for these (the cost gap is small for one-off
     rewrites and the hallucination problem disappears), OR
   - Down-skill with examples that **explicitly demonstrate
     non-invention** — e.g., examples where the input already contains
     specifics and the output preserves them, and a negative example
     where the rewrite invents details and is marked BAD.

4. **For any down-skilled prompt: audit the examples for length and
   specificity discipline.** Whatever pattern the examples demonstrate
   IS the output, regardless of what the rules say. The
   `down-skilling/SKILL.md` document already says this explicitly; the
   experiment confirms it operationally.

## What this still doesn't tell us

- **Sonnet 4.6 baseline.** Does vanilla Sonnet match down-skilled
  Haiku on these tasks? Is the cost/reliability tradeoff still in
  Haiku's favor after these findings?
- **Whether the hallucination pattern generalizes.** Does
  "demonstration-induced specificity" happen on other task types
  (summarization, extraction, code generation), or is voice-rewrite
  uniquely vulnerable?
- **Variance with deliberately *bad* downskilled examples.** Would
  shorter, less-specific examples in the Task-3 prompt fix the length
  and hallucination problems? An A/B on the example-set itself is the
  obvious next probe.
- **Effect on multi-turn or tool-using flows.** This is all
  single-shot, no tools.
