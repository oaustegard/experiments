# Haiku 4.5 — broader probe across 6 task archetypes (n=5)

**Date:** 2026-05-26 (follow-up to [`../RESULTS-n20.md`](../RESULTS-n20.md))

After landing PR #108, Oskar asked for more task coverage at lower n, with
the goal of identifying down-skilling failure cases and patching the skill.
This probe runs 5 new archetypes plus T3b — a direct A/B test of the
calibrated-examples hypothesis from the prior probe.

**Method:** 6 task pairs × 2 conditions × 5 runs = 60 Haiku dispatches.
All single-turn, single-shot, no tools beyond Write for delivery.

Outputs: [`outputs/`](outputs/) · Scorer: [`score.py`](score.py) · Raw
metrics: [`scores.json`](scores.json)

## Task inventory

| ID | Task | Hypothesized failure mode |
|---|---|---|
| T4 | Extract meeting fields as JSON | DS likely clean (format-bound) |
| T5 | One-line changelog from a diff | Possible length-collapse like T3 |
| T6 | Filter/sort/select top-N with tie-break | DS clean (concrete output) |
| T7 | NL → gh CLI command | DS may induce flag-invention |
| T8 | Copyedit a paragraph (open judgment) | DS over-flagging or rewriting |
| T3b | T3 voice rewrite, **calibrated examples** | A/B: does fixing examples fix the regression? |

## Headline numbers

| Task | Metric | Vanilla | Down-skilled |
|---|---|---:|---:|
| **T4** (extraction) | parseable JSON | 5/5 | 5/5 |
| | distinct schemas across runs | **5** | **1** |
| | uses only required keys | 0/5 | **5/5** |
| **T5** (changelog) | in 8–18 word range | 4/5 | **5/5** |
| | starts with action verb | 2/5 | **5/5** |
| | leaks impl detail (`ZoneInfo`) | 4/5 | **0/5** |
| | "refactor" framing (wrong) | 1/5 | 0/5 |
| **T6** (filter/sort) | top 3 set correct | 5/5 | 5/5 |
| | **correct tie-break order** | 2/5 | **5/5** |
| | tight 2-line schema | 0/5 | **5/5** |
| | markdown drift | 5/5 | 0/5 |
| **T7** (gh CLI) | **uses invented flags** | **5/5** | **0/5** |
| | fully correct command | 0/5 | **5/5** |
| | wrapped in markdown | 5/5 | 0/5 |
| **T8** (copyedit) | all 8 expected fixes present | 5/5 | 5/5 |
| | numbered-list schema | 4/5 | **5/5** |
| | markdown section headers | 5/5 | 0/5 |
| **T3b** (voice, **calibrated**) | hallucinated arch detail | 0/5 | **0/5** ✓ |
| | banned hype words | 3/5 | **0/5** |
| | in 60–90 word range | 0/5 | **0/5** ✗ |
| | avg word count (spec: 60–90) | 119 | 52 |

## Findings

### Down-skilling wins, cleanly, in 5 of 6 archetypes

No archetype in the broader probe showed down-skilling REGRESSING vs
vanilla. Every cell where down-skilling did something different did
something better:

- **T4** — vanilla produced 5 different JSON schemas across 5 runs
  (key names: `title`/`meeting_title`/`meeting_type`/`event_type`;
  `date`+`time`/`datetime`/`day`+`time`; `notes`/`description`/`requirements`).
  Down-skilled produced byte-identical output 5/5 times. This is the
  canonical use case: any production pipeline that consumes the JSON
  needs predictable keys.

- **T5** — vanilla leaked `ZoneInfo` (a Python module name) into 4 of
  5 changelog lines. Users reading a changelog don't care about Python
  module names; they care about behavior. Down-skilled stayed at the
  behavior layer in 5/5.

- **T6** — vanilla used the WRONG tie-break in 3/5 runs (listed
  Piranesi before Atomic Habits when they tie at rating 4.4; alphabetical
  break gives Atomic Habits first). Down-skilled, with the rule "ties
  broken alphabetically" stated explicitly, got it right 5/5. This is
  a subtle correctness category Haiku will silently miss without a rule.

- **T7** — **most actionable finding of this probe.** Vanilla
  hallucinated non-existent `--sort created --order asc` flags in 5/5
  runs. These flags do not exist on `gh pr list` — running the command
  produces an error. Down-skilled (with explicit rule "do not invent
  flags" and a tagged BAD example showing the wrong syntax) produced
  the correct `--search "sort:created-asc"` in 5/5 runs.

  This is the kind of failure that's *worse than wrong* because a
  user might believe the command worked. Haiku in vanilla mode here is
  not "less reliable than Sonnet" — it's *confidently outputting
  broken commands*. Down-skilling fixes it categorically.

- **T8** — both variants found all 8 grammar errors. The difference
  was format: vanilla wrapped every output in `# Copyedit Result` /
  `## Cleaned Paragraph` / `## Corrections Made` markdown structure
  (5/5); down-skilled obeyed the numbered-list-then-cleaned-paragraph
  schema (5/5). No false positives, no over-flagging in either.

### T3b — calibrated examples fix hallucination cleanly; length is partially fixed

The headline result: **0/5 down-skilled outputs invented architectural
details** vs **19/20 (95%) in the original T3 probe**. The calibrated
examples — see [`prompts/T3b_downskilled.md`](prompts/T3b_downskilled.md) —
worked exactly as hypothesized.

What made the difference, comparing T3 vs T3b examples:

| | Original T3 examples | T3b calibrated examples |
|---|---|---|
| Source specificity in Example 1 | Source is vague | Source has explicit specifics ("Stripe Connect", "idempotency keys", "2 vs 8-step flow") |
| Output specificity in Example 1 | Invented JWTs and refresh endpoints (not in source) | Preserves source specifics, adds nothing |
| Abstract-source example | None | Example 2: abstract input → output that *names what the source omitted* |
| BAD example | None | Example 3: abstract input → BAD invented output, explicitly tagged |
| Length range demonstrated | 33–36 words | 60–79 words |

The down-skilled T3b outputs picked up the **"acknowledge silence"
pattern** demonstrated in Example 2. Multiple runs explicitly said
things like *"the announcement does not specify the underlying
architecture, eviction strategy, or operational characteristics"* —
that pattern was modeled in Example 2 and Haiku adopted it.

**Length is still imperfect:**
- Original T3: avg 43 words, 0/20 in 60–90 range.
- T3b: avg 52 words, 0/5 in 60–90 range.
- The example-set average length moved from ~35 to ~70 words; output
  average moved from 43 to 52 (about 1/3 of the example shift).
- Haiku still anchors below the stated lower bound. The "Length 60–90"
  rule, even when reinforced with example calibration, doesn't fully
  override the example-set's central tendency.

**Practical reading:** if length-target is mission-critical, you need
something stronger than a single rule. Possible reinforcements (each
unmeasured here): adding a word-count gate in `<process>`, providing
*shorter* examples paired with explicit "your output should be ~2×
the length of these examples" instructions, or chaining a second
post-processing call that expands too-short outputs.

### Cross-task: when does down-skilling fail Haiku?

The original T3 result (95% hallucination) was specific to one prompt
construction failure mode, not a property of down-skilling or of
Haiku. Across this broader probe:

- **Examples drive the failure mode.** When examples model
  source-grounded outputs, down-skilling helps universally. When
  examples model un-grounded specificity, it hurts.
- **Concrete inputs prevent hallucination regardless of prompt
  structure.** T5 (diff input) and T6 (table input) had grounded
  outputs in both vanilla and down-skilled — the source IS the truth,
  and Haiku stuck close to it.
- **Abstract inputs are where down-skilling can backfire.** T3
  (abstract marketing paragraph) and to a lesser extent T7 (English
  prose requesting a CLI) leave room for invention. The fix is the
  same in both: a BAD example that explicitly demonstrates the
  invention you don't want.

## Proposed down-skilling skill update

See [`proposed-skill-patch.md`](proposed-skill-patch.md) for the
specific edits. The three additions, in priority order:

1. **Source-anchoring rule in Example Quality Criteria.** Every
   concrete fact in an example *output* must trace back to its example
   *input*. If an example has invented facts (even plausibly), Haiku
   will copy the invention pattern.

2. **Abstract-source example pattern.** For any task whose input
   could be vague (rewrites, summarization, voice tasks), include at
   least one example where the input is abstract and the output
   acknowledges the abstraction explicitly. The T3b Example 2 is the
   template.

3. **Mandatory tagged BAD example for confabulation-prone tasks.**
   Specifically: rewriting, summarization, NL→CLI conversion, any
   task where Haiku could output something plausible-but-fabricated.
   The BAD example should demonstrate the *specific* invention you
   want to prevent, not a generic "bad output."

## What this doesn't tell us

- **n=5 per cell** — robust enough to spot the dominant patterns; not
  enough to compare two close interventions.
- **Sonnet 4.6 still not measured.** The "is down-skilled Haiku worth
  it over vanilla Sonnet" question remains open. T7 in particular —
  where vanilla Haiku is *confidently broken* — would be the cleanest
  cell to test that gap.
- **No tool-using or multi-turn flows yet.**
- **The length-target problem in T3b is real but partially solved.**
  We don't have a clean fix yet beyond "make examples longer."
