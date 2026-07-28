# Retroactively running `optimizing-skills` on the down-skilling v1.2.0 edit

**Date:** 2026-05-29
**Question:** If you run the `optimizing-skills` validation gate (v0.1.0,
shipped #677) retroactively against the `down-skilling` v1.2.0 edit (#674,
the example-calibration update), does it reproduce the ship decision — and
what does the exercise expose about `optimizing-skills` itself?
**Source case:** `../haiku-assessment/` — the probe behind the blog post
*When Down-Skilling Makes Haiku Worse*.

---

## Setup

The v1.2.0 edit was four operations (exactly the optimizing-skills default
budget) added to `down-skilling`:

1. Source-anchoring bullet in Example Quality Criteria
2. Length-calibration bullet in Example Quality Criteria
3. "When the input could be abstract: model the silence" subsection
4. Tagged BAD/GOOD negative example as the default for confab-prone tasks
   (+ Activation step 5: audit the example set)

To run the gate properly I held two SKILL versions:

- `best` = `down-skilling` **v1.1.0** — current SKILL with the four edits
  reverse-applied (`skill-versions/down-skilling-v1.1.0.SKILL.md`)
- `candidate` = `down-skilling` **v1.2.0** — current SKILL as shipped
  (`skill-versions/down-skilling-v1.2.0.SKILL.md`)

**Check task:** the single task that *triggered* the revision — rewrite the
caching-layer promo paragraph in dry voice (Task 3 from haiku-assessment).
The optimizing-skills gate requires the check set "must include the failure
that prompted this revision"; this is that failure.

**Controlled dispatch.** The down-skilling artifact is a *prompt an author
compiles from the SKILL*. So the only variable between arms must be the SKILL
version. Two Sonnet author-agents each read one SKILL version (and nothing
else — the haiku-assessment outputs were walled off) and compiled a
Haiku-ready prompt. Each compiled prompt was then run on Haiku 4.5 ×5
(independent dispatches). 2 authors + 10 Haiku runs.

This is the gate's "dispatch each check task to the Agent tool" recipe, run
on `best` and `candidate` against a fixed check task.

---

## Result — the gate reproduces SHIP on the triggering failure

| Metric | best (v1.1.0) | candidate (v1.2.0) |
|---|---:|---:|
| **No invented architecture** (the triggering failure) | **2/5** (60% invention) | **5/5** (0% invention) |
| Length in stated 60–90 range | 0/5 | 0/5 |
| Mean words | 47 | 35 |

Per-run scores and rationale in [`scores.json`](scores.json); raw outputs in
[`arms/best/`](arms/best/) and [`arms/candidate/`](arms/candidate/); the two
compiled prompts are the `haiku_prompt.md` in each arm.

**Candidate strictly beats best on the triggering failure** (architectural
hallucination 0% vs 60%). Under the optimizing-skills rule — accept only on a
strict improvement on the held-out failure — the gate ships v1.2.0. That is
the decision #674 made. The retro reproduces it.

The original n=20 corroborates and strengthens this: the original v1.1.0-style
prompt (un-anchored examples) hallucinated in **19/20** runs; our fresh v1.1.0
author hallucinated in 3/5. Across two different v1.1.0 authors the candidate's
0% holds — so the win tracks the source-anchoring discipline, not author luck.

---

## What the retro exposed — about `down-skilling`

**Three of the four edits transmitted; one did not.**

- **Source-anchoring + BAD/GOOD + model-the-silence: transmitted and worked.**
  The candidate author produced a BAD/GOOD pair that named the exact invention
  ("inverted index", "60% latency reduction") and a GOOD output that *states
  what the source leaves unspecified*. Haiku copied that pattern: 5/5 clean,
  including at 56–59 words where there was room to invent and it didn't.

- **Length-calibration: did NOT transmit.** Both arms scored 0/5 in the
  60–90 range; candidate ran *shorter* (mean 35 vs 47). The candidate author
  wrote "60–90 words" into its rules but authored examples averaging ~37 words
  — the precise rules-vs-examples conflict the skill warns about, left in place
  despite Activation step 5 telling the author to audit for it. The audit step
  is a verbal instruction to the author, and the author blew past it. The
  "model the silence" pattern compounds this: abstain-style outputs are
  naturally terse, pulling length further below the floor.

So the v1.2.0 edit is a genuine improvement on its primary target (invention)
and a non-improvement on its secondary target (length). The ship was correct;
the length sub-claim in the v1.2.0 changelog ("rules don't override the example
central tendency" — true, but the fix doesn't hold) is not yet earned.

---

## What the retro exposed — about `optimizing-skills` (v0.1.0)

Two gaps the gate has, both surfaced by actually running it:

1. **Scoring collapses multi-criterion tasks.** The gate says "Score hard
   pass/fail per task." This task has two criteria (no-invention, in-range).
   If "pass" means *both*, then best 0/5 and candidate 0/5 → tie → **reject** —
   the length failure masks a real, large win on invention. The gate must score
   on the **triggering-failure criterion**, per-criterion, not a single
   collapsed pass/fail. The skill already says the check set must include the
   triggering failure; it does not say the *scoring* centers it. That is the
   gap.

2. **Single author-sample per version is underpowered.** The measured benefit
   moved from 95%→0% (original authors) to 60%→0% (our authors) purely on
   author variance. When the artifact under test is itself produced by an Agent
   (down-skilling, creating-skill outputs), n=1 author per arm lets author
   capability dominate the edit effect. The gate should sample ≥2 authors per
   version, or fix the author across arms.

Proposed edits to `optimizing-skills` from these findings:
[`proposed-patch-optimizing-skills.md`](proposed-patch-optimizing-skills.md).
Note the recursion: those edits should themselves clear the gate before
shipping — this retro is their motivating failure, not their validation.

---

## Caveats

- n=5 per arm on one check task. Enough to reproduce a 60-point gap; not enough
  for the length question (which is a flat 0/0 anyway).
- Hallucination scored by Muninn (Opus) on the experiment's own rubric — not an
  independent scorer. Per-run rationale is in `scores.json` for audit.
- Confound on length: candidate outputs are shorter, and shorter outputs have
  less surface to invent on. Countered by candidate staying clean at 56–59
  words (comparable to best's invented 54-word run), but not eliminated.
- `best` was reconstructed by reverse-applying the four edits to the live
  v1.2.0 SKILL, not pulled from git history of `oaustegard/claude-skills`.
