# skillopt-skill — `optimizing-skills`

Staging home for a **new skill**, `optimizing-skills/`, distilled from SkillOpt
(microsoft/SkillOpt, arXiv:2605.23904). It encodes a validation-gated discipline
for revising existing skills: a held-out check set, a two-tier `best`/`candidate`
gate (ship only on a strict improvement), bounded edits, failure-first
reflection, impact ranking, a protected core, and `remember()`-backed
cross-revision memory.

## Status

**Shipped.** The canonical skill lives in `oaustegard/claude-skills`
([PR #677](https://github.com/oaustegard/claude-skills/pull/677)), registered
under the `skill-development` plugin so it auto-mounts at
`/mnt/skills/user/optimizing-skills/` and is invocable by name or via
`versioning-skills` / `creating-skill`.

The copy here is the **review record / provenance snapshot** that produced it —
not the live source. If the skill evolves, edit it in `claude-skills`, not here.

(Earlier assumption that this session couldn't push to `claude-skills` was
wrong: `gh` is authenticated account-wide via `GH_TOKEN`. See the "GitHub
access is account-wide" note added to the repo `CLAUDE.md`.)

## Contents

- `optimizing-skills/SKILL.md` — the skill
- `optimizing-skills/references/skillopt-provenance.md` — SkillOpt → skill
  mapping and the Agent-tool scoring/reflection recipe

## Provenance

The code read that produced this: SkillOpt's gate is a ~40-line pure function
(strict `>`, two-tier best/working), edits are literal string ops, the "learning
rate" is an integer edit-budget cap, the "protected field" is HTML-comment
fences, and the genuinely novel bit is the optimizer-side meta-skill — adapted
here as `remember()`/`recall()` memory keyed to the skill being revised.
