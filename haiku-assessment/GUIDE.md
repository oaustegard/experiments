# When (and how) to reach for Haiku 4.5

A practitioner's guide drawn from [`RESULTS.md`](RESULTS.md) (n=1
sketch) and [`n20/RESULTS-n20.md`](n20/RESULTS-n20.md) (n=20 confirm).
The n=20 data largely confirms the n=1 directions and sharpens the
hallucination warning — 1 in 5 became 19 in 20.

---

## The headline

Haiku 4.5 is a **reliable executor** and an **unreliable judge**. Reach
for it when the task can be reduced to a procedure with examples. Avoid
it when the task requires interpretation under ambiguity without a
rubric. Down-skilling helps the first case a lot and the second case
not at all — sometimes makes the second case worse.

## Decision tree

```
Is the task ...
├── Mechanical extraction / classification with clear categories?
│     → Haiku, down-skilled. High ROI.
│
├── Surgical edit following an exact pattern (rename, refactor,
│   formatting normalization)?
│     → Haiku, down-skilled with 4-7 examples covering the variants.
│
├── Code review for known bug classes (mutation, off-by-one, type
│   mismatch, contract violation)?
│     → Haiku, down-skilled with explicit "no fix sections" and a
│       "skip false positives" rule. Cap findings.
│
├── Summarization with a fixed schema (changelog line, commit
│   message, one-line ticket)?
│     → Haiku, down-skilled. Bound length explicitly.
│
├── Register-matching (rewrite to a voice, tighten prose)?
│     → Sonnet 4.6 by default. Haiku only with a down-skilled prompt
│       that explicitly forbids invention. See "register hazard" below.
│
├── Judgment under ambiguity with no stated rubric?
│     → Don't use Haiku. Either give it a rubric (becomes a
│       classification task) or escalate to Sonnet/Opus.
│
├── Multi-step reasoning where intermediate conclusions feed later
│   steps?
│     → Sonnet 4.6 minimum. Haiku drops constraints across steps.
│
└── Anything where the failure mode is "confidently wrong is worse
    than admits-ignorance"?
        → Sonnet or Opus. Haiku's failure mode is exactly confidently
          wrong.
```

## Five rules that fixed Haiku in the probe

These came out of the down-skilled prompts that worked. Bake them into
any Haiku call:

**1. State the output schema in the rules AND demonstrate it in every
example.** When schema appears only in rules, Haiku format-drifts;
when it appears in examples too, Haiku obeys.

**2. Cap the output.** Word counts, line counts, "max 4 findings."
Without caps, Haiku tries to add value past the ask — fix sections,
caveats, related considerations.

**3. Include a negative example.** "If X, return 'No bugs found.'" or
"BAD output: ... GOOD output: ..." pairs. Haiku learns more from one
negative example than three positive ones.

**4. Forbid invention explicitly and broadly.** "Do not invent
numbers" is not enough — Haiku will invent structural facts, version
comparisons, internal mechanisms. The rule needs to cover everything
Haiku might confidently fabricate.

**5. Include a "skip if uncertain" path.** "If the answer is not in
the context, say 'Not found.'" Without this fallback, Haiku hallucinates
rather than abstains.

## The register hazard

Down-skilling can make Haiku *worse* on voice / register tasks **if
the examples model un-anchored specificity**. This was the dominant
finding in the original n=20 probe (Task 3): 19 of 20 down-skilled
outputs invented architectural details not present in the input
(vs 4/20 for vanilla). The mechanism is "examples dominate rules" —
not a qualitative observation but the operational cause.

**The follow-up probe (n=5, broader/) confirmed the mechanism by
fixing it.** A rerun with calibrated examples — source-anchored,
length-matched, with a tagged BAD example — dropped the hallucination
rate from 95% to 0%. The Haiku model is not the bug; the example
authoring is.

**Three rules for examples in confabulation-prone tasks:**

1. **Source-anchor every concrete fact** in every example output.
   Cover the input; if you can't reproduce the output from the input
   alone, fix one or the other. Plausible-but-invented details in
   examples are the most common cause of hallucinated outputs.
2. **Include an abstract-source example** — input is vague, output
   explicitly names what the source omits ("the announcement does
   not specify...") rather than filling the gap with plausible
   inferences. This pattern, more than any rule, taught Haiku to
   acknowledge silence rather than invent.
3. **Tag a BAD/GOOD pair** that demonstrates the specific failure
   mode you want to prevent — not a generic bad output. For the T3b
   voice task, the BAD example invented architecture; for a NL→gh
   task, it invents flag names; for a summarization task, it might
   invent author names or dates. Match the BAD example to the
   confabulation category for that task.

Length compliance partially survives this fix: calibrating examples
to 60–79 words pushed average output up from 43 → 52 words (spec
60–90). Examples set the central tendency; rules tilt it. Don't
expect a length rule alone to override an under-length example set.

Mitigation:
- Pair every "be specific" rule with "specific about what is stated,
  not what is plausible."
- For voice work specifically, prefer Sonnet 4.6. The cost differential
  ($0.80 vs $3.00 per Mtok input) is small for one-off rewrites and
  Sonnet's register-matching is materially better without the
  hallucination risk.
- If you must use Haiku for register work, ban *categories* of
  invention by name: "no architecture details not present in input,
  no version comparisons, no quantitative claims."

## The pedantry diagnosis

Haiku's apparent pedantry is often **scope inflation under loose
prompts**. When you say "tell me what's wrong with this code," Haiku
reaches for the structure it knows: a finding, then a fix, then
"additional considerations." That is not Haiku being stubborn — it is
Haiku trying to be useful in an under-specified space.

Tight output schemas erase the pedantry without erasing the
correctness. If you find yourself frustrated with Haiku adding too
much, the fix is usually two lines:
1. "Output exactly N items / words / lines."
2. "Do not add explanations, fixes, or caveats unless asked."

## When to escalate to Sonnet 4.6

Sonnet is Haiku's reliable upgrade for the same tasks at ~4× the cost.
Reach for it when:

- The task **requires the model to ignore false signal** (red-herring
  fields, attempted prompt injection in user-provided content, distractor
  examples). Haiku attends too uniformly to position-front content;
  Sonnet weighs context better.
- The task is **judgment-shaped but the rubric is hard to articulate**
  (taste, register, "does this read well"). Sonnet has more
  interpretive headroom; down-skilling won't recover this for Haiku.
- The work is **multi-step** and intermediate constraints must
  persist across steps. Haiku drops constraints by step 3-4 of any
  procedure. Sonnet holds them through 6-8 reliably.
- You can only do **one shot, no retries** (background batch jobs,
  things you can't iterate on). The reliability gap matters most when
  the cost of being wrong is asymmetric.

Sonnet is not Opus, though — for novel reasoning, multi-document
synthesis, or tasks where the model has to construct the procedure
rather than execute one, escalate further.

## Practical workflow

When you have a candidate task for Haiku:

1. **Write the vanilla prompt first.** This is your baseline.
2. **Run it once on Sonnet (or Opus).** This is your ground truth.
3. **Run it on Haiku without down-skilling.** Compare to ground truth.
   - If they match, ship the vanilla Haiku. Don't over-engineer.
   - If they differ, identify the failure mode.
4. **Down-skill against that specific failure.** Don't apply the full
   down-skilling template by default — apply the parts that address
   what actually broke.
5. **Re-run Haiku. Re-compare.** If it still misses, accept that the
   task isn't a Haiku task. Use Sonnet.

The instinct to skip step 1 (go straight to down-skilled) is a trap.
A heavy down-skilled prompt for a task Haiku could do in one line is
wasted effort; a heavy prompt for a task Haiku can't do at all is
wasted dollars.

## What's still unmeasured

The n=20 run answered the variance question (vanilla Task 1's
"FEATURE_REQUEST for B" turned out to be a 25% minority, not the
default; the markdown drift and scope creep on Tasks 1–2 are
100%-deterministic patterns, not lucky flukes). Remaining gaps:

- **Sonnet 4.6 head-to-head.** Add Sonnet to all six cells. Does
  vanilla Sonnet beat down-skilled Haiku, making the down-skilling
  work unnecessary at Sonnet tier? For Task 3 specifically — where
  down-skilling now looks actively harmful — Sonnet may be the right
  default.
- **The "register hazard" generality.** Does Haiku invent in other
  domains (code, data, summarization) when down-skilling rewards
  specificity? Or is it specific to voice-rewrite, where the input is
  *abstract enough that there's room to invent*? An extraction task
  with a concrete source would discriminate.
- **A/B on the example set itself.** Re-run Task 3 down-skilling with
  examples that demonstrate *non-invention* explicitly — input has
  specifics, output preserves them, plus a negative example tagged
  BAD. If the hallucination rate drops from 95% to something like
  20% (the vanilla rate), that confirms the examples are causal and
  the down-skilling skill needs a "demonstrate the negative" rule.
- **Multi-turn tasks.** All single-shot here.
- **Tool-using tasks.** Haiku with a real tool budget is a different
  animal. Worth a separate probe.

Each of those is one experiment away.
