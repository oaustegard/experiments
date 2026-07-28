# Proposed patch to `down-skilling/SKILL.md`

Based on findings in [`RESULTS-broader.md`](RESULTS-broader.md). The
skill lives in `oaustegard/claude-skills` (not in this repo); these
edits should be applied there.

The skill already names "examples dominate rules" but doesn't give
the example author a procedure to keep examples aligned with rules.
The edits below close that gap.

---

## Edit 1 — Add to `### Example Quality Criteria` (section starting at SKILL.md line 233)

Existing list of bullets ends with:
```
- Order examples from simplest to most complex — this progressive
  difficulty helps Haiku build up its understanding
```

Add two new bullets:

```
- **Source-anchor every concrete fact.** Each specific noun, number,
  or claim in an example *output* must be inferrable from that
  example's *input*. If your example output invents a plausible
  architectural detail, Haiku will copy that invention pattern even
  when its real input doesn't support it. The single fastest way to
  audit: cover the input and ask "could I have produced this output
  knowing only the input?" If not, either add the detail to the input
  or remove it from the output.
- **Match example length to the stated output range.** Examples
  outweigh rules on length. If your rules say "output 60–90 words"
  but your examples average 35 words, Haiku will produce ~35 words.
  Calibrate example lengths to sit inside (or slightly above) the
  stated output range.
```

---

## Edit 2 — Add a new subsection between `### Example Format` (line 200) and `### Example Sizing Guidance` (line 221)

Insert:

```markdown
### When the input could be abstract: model the silence

Some tasks (rewriting, summarization, voice/register editing,
explanation-from-source) take inputs where the source content is
*itself* abstract — marketing copy, vague descriptions, high-level
summaries. These tasks are uniquely prone to a failure mode where
Haiku invents concrete details that *sound right for the domain* but
aren't in the source.

**Symptom:** an output that confidently mentions specifics (algorithm
names, technology choices, percentile numbers, version comparisons)
not present in the input.

**Fix:** Include an example pair where the input is genuinely abstract
and the output **explicitly acknowledges what is unspecified**, rather
than filling the gap with plausible-sounding inferences. Pattern:

```xml
<example>
<input>
"We're excited to release our new search backend. We think it's a
massive upgrade over the old one."
</input>
<output>
New search backend released. The announcement does not specify the
underlying changes — whether they affect indexing, ranking, query
parsing, or operational characteristics is not stated. Migration
guidance will follow. The old search path remains available during
transition.
</output>
<reasoning>
The source provides only one factual claim ("released") and one
unverifiable claim ("upgrade"). The output stays at the source's
level of abstraction and explicitly names the categories of
information the source omits.
</reasoning>
</example>
```

This pattern, plus a paired tagged-BAD example that demonstrates the
exact invention you want to prevent, reduced architectural
hallucination from 95% to 0% on a voice-rewrite task. See
`oaustegard/claude-workspace/experiments/haiku-assessment/n20/broader/`
for the data.
```

---

## Edit 3 — Strengthen the Negative/rejection-case row in the example distribution table (line 192)

Existing:
```
| 4 | **Negative/rejection case** | Input that should be rejected, handled differently, or produce an empty/default output. Shows Haiku what NOT to do. |
```

Replace with:
```
| 4 | **Negative case (tagged BAD/GOOD pair)** | For tasks where Haiku could output something plausible-but-wrong (rewriting, summarization, NL→command, anything that rewards confident specificity), include an example pair tagged "BAD output:" and "GOOD output:" that demonstrates the *specific* failure mode you want to prevent. A generic "bad output" example is much less effective than one that names the failure category Haiku is most likely to fall into for this task. |
```

---

## Edit 4 — Add a step to `## Activation` (section at line 55)

After step 4 ("Generate 4-7 diverse examples..."), insert a new step:

```
5. **Audit your example set before delivering.** Two checks, both
   should pass:
   (a) **Source-anchoring**: for each example output, every concrete
       fact traces to a source in that example's input.
   (b) **Length calibration**: example output lengths sit inside the
       stated output range. If your rule says "60–90 words" and your
       examples average 35, the rule will not hold at runtime.
   If either check fails, regenerate the offending examples. The
   evidence on this is unambiguous: examples beat rules.
```

Renumber the existing step 5 (Deliver) to step 6.

---

## CHANGELOG entry

Suggested addition to `down-skilling/CHANGELOG.md`:

```
## 1.2.0 - 2026-05-26

- Added source-anchoring requirement to Example Quality Criteria.
  Examples whose outputs contain invented facts cause Haiku to
  copy the invention pattern (95% rate on a voice-rewrite task in
  haiku-assessment/n20/RESULTS-n20.md).
- Added "model the silence" subsection for abstract-source tasks
  with a concrete example pattern. Validated to drop hallucination
  rate from 95% → 0% on the same task with calibrated examples
  (haiku-assessment/n20/broader/).
- Tightened negative-example guidance: prefer tagged BAD/GOOD pairs
  that name the specific failure mode for the task.
- Added an "audit your examples" step to the activation procedure.
```
