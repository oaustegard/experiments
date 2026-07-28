# Haiku 4.5 with and without down-skilling — a probe

**Date:** 2026-05-26  
**Question:** When does Haiku 4.5 earn its place in a delegation chain, and how
much does down-skilling actually change its behavior?  
**Method:** Three small tasks chosen to stress the failure modes the
operator named — context mis-reading, pedantry under ambiguity, weak
register-matching. Each task run twice on Haiku via parallel sub-agents:
once with a vanilla prompt (what a normal practitioner would type), once
with a down-skilled prompt built per the `down-skilling` skill (XML
sections, rubric, 4–6 examples).  
**Sample size:** n=1 per condition. This is the initial probe.
Findings here are qualitative.

> **Update (same day):** Scaled to n=20 per cell — see
> [`n20/RESULTS-n20.md`](n20/RESULTS-n20.md) for the larger sample and a
> [comparison chart](n20/comparison.png). The Task-3 hallucination
> finding strengthens dramatically at scale (1/5 → 19/20). The Task-2
> scope-creep pattern holds at 100% in both directions
> (vanilla always adds a fix; down-skilled never does). Task-1
> per-rubric accuracy at n=20: vanilla 94%, down-skilled 100%.

Prompts: [`prompts/`](prompts/) · Outputs: [`outputs/`](outputs/)

---

## Task 1 — Support-ticket triage (4 tickets, 4 categories)

**Vanilla output** ([file](outputs/task1_vanilla.md)): 4 entries, each as a
bolded markdown header + a paragraph of reasoning. Length: ~80 words.
Classifications: A=BUG, **B=FEATURE_REQUEST**, C=SPAM, D=BUG.

**Down-skilled output** ([file](outputs/task1_downskilled.md)): 4 lines,
exact format. ~40 words total. Classifications: A=BUG, **B=QUESTION**,
C=SPAM, D=BUG.

The split is on ticket B: *"How can I export... I see CSV and PNG but no
PDF. Is this on the roadmap?"* Both classifications defend themselves
locally; what the down-skilled prompt got was **policy consistency** —
its examples covered an SSO-roadmap case identically, so B fell into the
same bucket. Vanilla Haiku picked a reasonable interpretation in
isolation but you cannot predict which interpretation it will pick next
time.

What down-skilling fixed:
- Schema adherence (one line per ticket, no markdown drift)
- Length discipline (≤15 words per reason)
- Policy consistency under ambiguous categories

What it did not need to fix:
- Recognition of A as a regression, C as spam, D as a "minor" bug

---

## Task 2 — Code review for a contract bug

The function `merge_user_prefs` has one real correctness bug: assignment
aliasing causes the function to mutate `default_prefs`, violating its
own "Does not mutate inputs" docstring.

**Vanilla output** ([file](outputs/task2_vanilla.md)): Found the bug.
Then added an unsolicited fix block (`result = default_prefs.copy()`),
then added an "Additional consideration" paragraph about `copy.deepcopy`.
Length: ~140 words plus a code block.

**Down-skilled output** ([file](outputs/task2_downskilled.md)): Found the
bug. Stated the divergence, gave a concrete demonstration. ~60 words. No
fix, no caveats, no expansion.

Both found the bug. The down-skilled prompt explicitly banned "suggested
fix" sections, and Haiku obeyed. This is the pedantry-and-scope-creep
failure mode that Oskar named, except it's not quite pedantry — it's
**helpfulness inflation**. Vanilla Haiku tries to add value past the
question; the downskill rule "Findings only." stops it cleanly.

Note: neither variant flagged a second potential issue (the
`if value is not None` skip), correctly, because the docstring doesn't
promise None-passthrough semantics. Both stayed within contract — but
the down-skilled prompt's rule 6 ("If a 'bug' requires assuming the
caller misuses the function, it is not a bug — skip it") was doing
load-bearing work to keep Haiku from false-positiving.

---

## Task 3 — Rewrite a promotional paragraph in dry voice

This is where the comparison gets interesting.

**Vanilla output** ([file](outputs/task3_vanilla.md)):

> "We've developed a new caching layer designed to improve team performance.
> The system reduces latency and provides a streamlined developer experience.
> It's available now for you to use."

Strips the worst adjectives. Still has soft promotional residue:
"designed to," "streamlined developer experience," "for you to use."
Length: ~30 words — much shorter than the requested "same length."

**Down-skilled output** ([file](outputs/task3_downskilled.md)):

> "Caching layer released. It intercepts reads at the application level
> and stores results in a distributed hash table with automatic invalidation
> on write. Early users report lower p99 latency on cached reads and
> simplified integration compared to the previous manual cache-aside
> pattern."

Hits the requested register. Hits the requested length. **But it
invented architectural details** — "distributed hash table," "cache-aside
pattern" — that were not in the source. Rule 7 said "If the input has no
concrete numbers, do NOT invent them." Haiku obeyed the letter (no
fabricated numbers) but not the spirit (fabricated structural facts).

This is the most useful finding from the probe: **down-skilling pushed
Haiku from vague-helpful into confidently-specific, and confident
specificity is exactly where hallucination lives.** A rule that says "be
concrete" without a corresponding rule that says "concrete about what is
actually given, not what is plausible" rewards invention.

To fix the down-skilled prompt: rule 7 should read "Do not invent
numbers, system architecture, internal mechanisms, or comparisons to
prior versions. If the input does not state how the thing works,
describe what it does in qualitative terms."

---

## Cross-task observations

| Failure mode | Vanilla Haiku | Down-skilled Haiku |
|---|---|---|
| Format drift | Heavy (markdown bold, sections, expansion) | None — schemas held |
| Scope creep | "Fix" blocks, "additional considerations" | None — stayed inside the rules |
| Policy consistency under ambiguity | Local-reasonable, globally unpredictable | Reproducible via rubric |
| Length adherence | Loose | Tight |
| Hallucination | Vague-but-grounded | **Specific-but-invented** (when rules reward specificity) |
| Pedantry (over-flagging) | Present but mild here | Absent — guarded by "skip caller-misuse" rule |

The pedantry the operator perceives in Haiku may be partly a
**format-and-scope artifact**: when you ask "tell me what's wrong" with
no schema, Haiku reaches for the structure it knows — headers, fixes,
caveats, "additional considerations." Constrain the output and the
pedantry disappears with it.

The remaining real Haiku failure mode after down-skilling is the
**confident-specificity drift** seen in Task 3 — and it is induced, not
intrinsic. The down-skilling skill should grow a "do not invent facts
not present in input" guardrail by default.

---

## Cost notes

The six sub-agents reported 33,808–34,730 total tokens each. The
down-skilled prompts are 3–10× longer than the vanilla ones (Task 3:
~700 vs ~120 tokens). At Haiku pricing, that delta is ≈$0.0005 per call
— negligible. The down-skilling skill's economics argument
("examples pay for themselves at 1-in-5 retry prevention") holds; on
these three tasks, down-skilling would have prevented at least one
retry on Task 1 (B mis-categorized) and one on Task 2 (scope drift),
ROI positive in every case.

---

## What this does not tell us

- **n=1 per cell.** Variance across runs is unmeasured. Haiku might pick
  QUESTION on a re-run of vanilla Task 1.
- **Task selection is curated**, not random — the tasks were chosen to
  exercise specific failure modes.
- **Sonnet 4.6 was not tested.** The "by less extension" question
  remains a priori.
- **No multi-turn or tool-using tasks.** The pedantry mode Oskar named
  may live more in those settings than in single-turn classification.

See [`GUIDE.md`](GUIDE.md) for practitioner takeaways.
