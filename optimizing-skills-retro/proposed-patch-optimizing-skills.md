# Proposed patch to `optimizing-skills/SKILL.md` (v0.1.0 → v0.2.0)

Motivated by `experiments/optimizing-skills-retro/RESULTS.md`. Two edits,
both inside the textual-learning-rate budget. Language follows
skill-language-compliance: imperative, diagnosed-failure cited, no
liking/reciprocity. **These edits should themselves clear the gate they
describe before shipping** — this retro is their motivating failure, not
their validation.

The skill lives in `oaustegard/claude-skills`; apply there.

---

## Edit 1 — score on the triggering-failure criterion, not a collapsed pass/fail

In `## The gate — run it every revision`, replace gate step 3:

Existing:
```
3. **Score both on the check set.** "Run" here = dispatch each check task to the
   **Agent tool** (`subagent_type=general-purpose`) with the skill version in
   context, or evaluate by hand for small sets. Score hard pass/fail per task.
```

Replace with:
```
3. **Score both on the check set.** "Run" here = dispatch each check task to the
   **Agent tool** (`subagent_type=general-purpose`) with the skill version in
   context, or evaluate by hand for small sets. Score **per criterion**, not one
   collapsed pass/fail. When a task carries several criteria, the criterion that
   decides accept/reject is **the failure that prompted this revision** — the
   others are regression guards (they must not get worse). Collapsing criteria
   masks the win: in the down-skilling-v1.2.0 retro, the edit drove architectural
   hallucination 60%→0% while an unrelated length criterion stayed 0/5 in both
   arms; a single combined pass/fail scored that as a 0–0 tie and would have
   rejected a large, real improvement.
```

## Edit 2 — sample ≥2 authors per version when the artifact is Agent-produced

Append to `## The gate — run it every revision`, after step 4:

```
**When the skill's output is itself compiled by an Agent** (e.g. down-skilling
and creating-skill produce a prompt an author writes from the SKILL), score
**≥2 author samples per version**, or fix one author across both arms. n=1
author per arm lets author capability dominate the edit effect: the same
down-skilling edit measured 95%→0% with one author pair and 60%→0% with another.
The edit was real either way — but a single sample cannot tell a real edit from
a lucky author.
```

---

## CHANGELOG entry

```
## [0.2.0] - 2026-05-29

### Changed
- Gate scoring is now per-criterion; accept/reject is decided by the
  triggering-failure criterion, with other criteria as regression guards.
  A collapsed pass/fail masked a 60%→0% win behind an unrelated tie
  (experiments/optimizing-skills-retro).

### Added
- Require >=2 author samples per version (or a fixed author) when the skill's
  artifact is compiled by an Agent, to separate edit effect from author variance.
```
