# gemini-drafter

Muninn is the Claude-based agent with persistent memory that runs this repo's experiments and writes under its own byline. Its public prose is drafted by Opus 5.5 from a one-page writer brief, then edited by a second Opus pass that runs the `declauding` skill (the pipeline measured in [`../positive-writer-brief/`](../positive-writer-brief/RESULTS.md), where the brief is arm B and brief-plus-editor is arm C).

This experiment asks whether drafting with a different model family gives better prose than Opus does from the same brief. Two things prompted it. Alexander Khakhalin [described](https://bsky.app/profile/khakhalin.bsky.social/post/3n2gntncefk2d) on 2026-11-16 a writing setup where Claude does research and prompting, Gemini writes first drafts, and he rewrites. And the stock constructions the editor pass keeps removing from Muninn's drafts may be Opus's defaults, which no brief removes.

## Arms

| arm | drafter | editor pass | source |
|---|---|---|---|
| B | Opus 5.5, effort low | none | `../positive-writer-brief/drafts/B/` (2026-11-10) |
| G | `gemini-4-carbon`, thinking level low | none | new, `drafts/G/` |
| C | as B | Opus declauding pass | `../positive-writer-brief/finals/B/` |
| G-final | as G | the same Opus declauding pass | new, `finals/G/` |

- **Same inputs.** G's system prompt is B's, byte for byte (`pwb.system_prompt("B")`: the Muninn persona line plus [`brief_b.md`](../positive-writer-brief/brief_b.md)), sent as Gemini's `systemInstruction`; the task prompt is `pwb.draft_prompt(t)`; output comes back through a JSON schema with one `draft` field, as B's did. Twelve tasks across six surfaces (README, PR body, blog section, results Answer, commit message, Bluesky post), four drafts per task per arm, from [`tasks.json`](../positive-writer-brief/tasks.json).
- **Drafter choice.** `gemini-4-carbon` is the most capable Gemini model the Cloudflare gateway serves on 2026-11-17 (third-party listings put it above `gemini-4-argon` at the same price). It answers through the gateway with no special access.
- **Editor.** G-final uses the editor system prompt and edit prompt of positive-writer-brief unchanged (Opus 5.5, effort low, declauding SKILL.md, register.md, exceptions.md, voice signature, plus the lint output on the draft).

## Metrics

1. **Blind judges.** Each judge call holds all eight texts for one task and one stage (four B and four G, or four C and four G-final), shuffled under random ids, with the rubric from positive-writer-brief (`JUDGE_SYSTEM`): register 1–7, focus 1–7, invented-claim count. B's texts are re-judged here alongside G's; ratings from the earlier experiment are not reused, because each call rates its eight texts against each other.
2. **Judge families.** Opus 5.5 and Sonnet 5.5 (effort medium, as before), and `gemini-4-carbon` on the identical prompt. LLM judges favour their own family's text ([Panickssery et al., NeurIPS 2024](https://papers.nips.cc/paper_files/paper/2024/hash/7f1f0218e45f5414c79c0679633e47bc-Abstract-Conference.html)). Claude judges would therefore lean towards B, so a G advantage under Claude judges is the conservative result. The Gemini judge is reported as a check, and the gap between the two families is reported as an estimate of self-preference.
3. **Lint rate** from `declaude_lint.py`, per 1,000 words. The lint catalogues Claude's tics, so a lower rate for G partly follows from how the lint was built. It is reported and carries no prediction.
4. **Newcomer read.** A Sonnet 5.5 reader, as in positive-writer-brief, counts stumbles for every G and G-final text. B's and C's counts are reused from the earlier run (same reader prompt and model, one text per call).

Differences are G − B on raw drafts and G-final − C after editing, averaged over the twelve tasks, with a two-level bootstrap (tasks, then drafts within a task; 10,000 resamples, seed 7).

## Pre-registered predictions

Committed and pushed before any G draft used in the data was generated. One G draft was generated as a plumbing probe and is not in the data.

- **P1** (the question): G's raw register beats B's under the Claude judges. Holds if the 95% CI on G − B lies above 0.
- **P2**: the advantage survives the editor pass. Holds if the 95% CI on G-final − C register (Claude judges) lies above 0.
- **P3**: G's focus is no worse than B's. Holds if the lower bound on G − B focus is above −0.5.
- **P4**: G invents no more than B. Holds if the upper bound on G − B invented claims per text is below +0.5.
- **P5** (exploratory): lint rate, stumbles, the Gemini judge's verdicts and the self-preference gap.

Muninn's prior, stated before the run: P1 fails (about 70%). A secondary write-up of the creative-writing leaderboards (not checked against the leaderboards themselves) puts Opus 5.5 first and Gemini 4 Argon far down, with about three times the slop score of a Claude model. B's register under the earlier judges was already 5.7 of 7. P3 holds. P4 is a coin flip.

## Reproducing

```bash
python3 score.py     # recomputes every number in RESULTS.md from drafts/, finals/, scores/ and the B/C texts
```

Regenerating needs `CF_ACCOUNT_ID`, `CF_GATEWAY_ID` and `CF_API_TOKEN` for the gateway, and the `dispatching-subagents` skill for the Claude stages; `run.sh` runs every stage in order and resumes after an interruption.
