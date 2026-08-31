# ERRORS — hypothetical-classification

What was wrong, how it was caught, and which direction it pushed the conclusion.
The base rate is the useful number: 5 errors, 2 of which reached a published
artifact before being caught.

## 1. Implemented the wrong paper for the first half of the session

**Wrong.** Given a PDF and the framing "cheap possibly-hallucinating classifier,
better model as the judge", I identified the PDF as HyDE (arXiv 2212.10496) and
built HyDE — generative query expansion for document retrieval — measuring it over
Muninn's FTS5 memory corpus at n=80. The actual request was Doug Turnbull's
hypothetical-classification pattern: a *closed label vocabulary*, a cheap model, and
a cheap embedder. Different problem. HyDE has no label set and its "better model" is
an encoder.

**Caught.** By Oskar, who had not read the PDF either and said so. The two are
genuinely adjacent — both hallucinate a text artifact and resolve it against real
data — which is why the misread survived a careful reading of the paper.

**Direction.** No effect on anything reported here; the HyDE work is a separate
negative result in `../hyde-recall/`. Cost: roughly half the session. The
generalisable bit is that "user attaches a paper + describes a pattern" does not
establish that the paper *is* the pattern, and one clarifying question would have
been cheaper than the measurement.

## 2. Published a boundary that was a prompt bug — twice, into two repos

**Wrong.** Measured the pattern on Muninn's tag corpus using the source post's
novelty-anchored prompt, got 0.200 against a 0.400 no-model control, and concluded
that the pattern fails when item and label do not share a register. Shipped that
conclusion as a finding in `muninn-utilities#127` and `claude-skills#782`, in the
module docstring, the SKILL.md, the CHANGELOG, and both PR bodies.

**Caught.** By running the register prompt — already established as better on WANDS
— against the same 250 memories. The arm moved 0.200 → 0.500, past the control, and
the union 0.496 → 0.676. The "boundary" was the prompt.

**Direction.** Understated the pattern badly, and in the most damaging place: a
documented limit is exactly the thing a future reader will not re-test. Both PRs
were corrected in a follow-up commit rather than a force-push, so the withdrawn
claim stays in the history. The lesson is narrower than "test more": **an arm that
loses under configuration X is not a finding about the pattern until X is the best
known configuration.** The register prompt was already measured as better on WANDS
when the tag conclusion was written, and was not carried over.

## 3. Asserted a subagent's reasoning-effort control without verifying it

**Wrong.** Wrote `.claude/agents/*.md` probe definitions with `reasoning_effort:
low`/`high` to test whether a Haiku subagent's thinking is controllable.

**Caught.** The Agent tool rejected both new agent types — the registry loads at
session start, so definitions written mid-session are invisible.

**Direction.** None on any number here; the subagent cost floor was measured with
the built-in `general-purpose` agent and does not depend on the frontmatter. The key
is reported as unverified rather than as working.

## 4. First eval was 6 hand-written queries

**Wrong.** The initial retrieval comparison ran on 6 queries. Every arm scored 3/6.

**Caught.** By noticing one query was worth 0.167 — the trap `METHODS.md` already
records under "a 34-row eval can report the opposite sign of a real effect".

**Direction.** Uninformative in both directions. Regenerated at n=80 (and n=468 for
the classification work) before any conclusion.

## 5. Did not grep METHODS.md before starting

**Wrong.** Built and measured from scratch without checking the repo's ledger.

**Caught.** By grepping it after the fact, which surfaced two directly relevant
prior negatives (`nl2sh-dense`, `muninn-rm3`) and a named open question. `METHODS.md`
says "**Grep this file before starting a new experiment**" in bold at the top.

**Direction.** None on the conclusions — the prior work agrees. Cost was duplicated
effort on the HyDE half.
