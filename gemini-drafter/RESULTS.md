# gemini-drafter: results

Muninn is the Claude-based agent that writes this repo's experiments up. It drafts its public prose with Opus 5.5 from a one-page writer brief (a list of "do X, because the reader Y" statements), then a second Opus call edits the draft with `declauding`, a skill that finds and rewrites stock model constructions. This experiment asked whether Gemini (`gemini-4-carbon`), given the same brief and the same tasks, writes better first drafts. Design and pre-registered predictions: [`README.md`](README.md).

Names used below: **B** is Opus's raw drafts, **G** is Gemini's raw drafts, **C** is B after the editor pass, and **G-final** is G after the same editor pass. Every text was scored 1–7 by blind judges on two scales. **Register** measures how far the prose reads as plain technical writing, free of stock constructions such as staged reveals, verdict headers and aphoristic closers. **Focus** measures whether the reader gets the point first and the text describes the thing rather than the author's route to it. Judges also counted **invented claims**: numbers, names or causes not in the facts supplied with the task. "Claude judges" means the mean of an Opus 5.5 judge and a Sonnet 5.5 judge; a `gemini-4-carbon` judge saw the same prompts as a check.

## Answer

No. Over 12 writing tasks with 4 drafts per arm, Claude judges rated Gemini's drafts level with Opus's on register: G − B = +0.03 on the 7-point scale, 95% CI [−0.34, +0.38]. <!-- @evidence[[results/summary.json#comparisons.raw_G_vs_B.register_claude.diff]] --> After the editor pass they were still level, at +0.04. <!-- @evidence[[results/summary.json#comparisons.final_Gfinal_vs_C.register_claude.diff]] -->

Gemini's raw drafts were less focused, G − B = -0.68. <!-- @evidence[[results/summary.json#comparisons.raw_G_vs_B.focus_claude.diff]] --> The editor pass narrowed that to -0.15, inside the noise. <!-- @evidence[[results/summary.json#comparisons.final_Gfinal_vs_C.focus_claude.diff]] --> Gemini's drafts carried fewer invented claims, 0.30 per text against 0.79, and the Gemini judge counted the same difference. <!-- @evidence[[results/summary.json#comparisons.raw_G_vs_B.invented_claude.x]] --> Each judge call rated B and G texts side by side, so the difference is sound, but the absolute rates move with what else is in the call.

For raw drafts, switching the drafter would trade focus for fidelity to the supplied facts; after editing only the fidelity gain remains. It would not improve register, which was the reason to try it. The Gemini judge scored Gemini's drafts about 0.4 higher on register than the Claude judges did, so a panel with a third model family could shift the register reading.

## Findings

1. Register did not differ: G − B was +0.03 [−0.34, +0.38] on raw drafts <!-- @evidence[[results/summary.json#comparisons.raw_G_vs_B.register_claude.diff]] --> and G-final − C was +0.04 [−0.20, +0.27] after editing. <!-- @evidence[[results/summary.json#comparisons.final_Gfinal_vs_C.register_claude.diff]] --> Predictions P1 and P2, that Gemini would score higher, failed (`results/summary.json`).
2. Gemini's raw drafts were less focused: G − B was −0.68 [−1.12, −0.28]. <!-- @evidence[[results/summary.json#comparisons.raw_G_vs_B.focus_claude.diff]] --> Both Claude judges agreed: Opus rated Gemini higher on 0 of the 12 tasks, <!-- @evidence[[results/summary.json#comparisons.raw_G_vs_B.focus_opus.tasks_x_higher]] --> and Sonnet's difference alone was -0.62. <!-- @evidence[[results/summary.json#comparisons.raw_G_vs_B.focus_sonnet.diff]] --> P3 predicted Gemini would be no more than 0.5 points worse and failed (`results/summary.json`). After editing the gap was −0.15 [−0.48, +0.19]. <!-- @evidence[[results/summary.json#comparisons.final_Gfinal_vs_C.focus_claude.diff]] -->
3. Gemini invented fewer claims: 0.30 against 0.79 per raw text, a difference of −0.49 [−0.94, −0.10]. <!-- @evidence[[results/summary.json#comparisons.raw_G_vs_B.invented_claude.diff]] --> The Gemini judge found −0.54. <!-- @evidence[[results/summary.json#comparisons.raw_G_vs_B.invented_gemini.diff]] --> P4, that Gemini would invent no more than 0.5 extra claims per text, held (`results/summary.json`). Opus's extra claims came mostly from three tasks: by the Claude judges' counts: a README written from a changelog, where Opus averaged 2.6 invented claims per draft against Gemini's 0.4, a commit message (1.4 against 0) and a blog section (1.1 against 0.4) (`results/texts.jsonl`).
4. The Gemini judge favoured Gemini's drafts more than the Claude judges did, by +0.43 on register <!-- @evidence[[results/summary.json#self_preference.raw_G_vs_B_register.diff]] --> and +0.70 on focus for raw drafts. <!-- @evidence[[results/summary.json#self_preference.raw_G_vs_B_focus.diff]] --> Judges are known to favour their own family's text, and this experiment cannot tell which family's judges are the biased ones (`results/summary.json`).
5. Gemini wrote shorter drafts, 157 words against 183. <!-- @evidence[[results/summary.json#comparisons.raw_G_vs_B.words.x]] --> Its lint rate (hits from the `declauding` regex lint) was 3.86 against 5.34 per 1,000 words, with a CI spanning zero. <!-- @evidence[[results/summary.json#comparisons.raw_G_vs_B.lint_rate.x]] --> The lint was built from Claude's tics, so a lower Gemini rate does not show cleaner prose (`results/summary.json`).
6. A Sonnet reader told it was the task's newcomer reader listed 5.35 stumbling points per Gemini draft against 5.46 per Opus draft, no difference. <!-- @evidence[[results/summary.json#comparisons.raw_G_vs_B.stumbles.x]] --> (`results/summary.json`)

## Method

- **Fixture.** The twelve tasks, Muninn persona and writer brief from [`../positive-writer-brief/`](../positive-writer-brief/README.md). Arm B (Opus 5.5 drafts at effort low) and arm C (B plus the declauding editor pass) are reused from that experiment's 2026-11-10 run.
- **New arms.** G: `gemini-4-carbon` at thinking level low, given the same system prompt as `systemInstruction` and the same task prompt, four drafts per task. G-final: G plus the same Opus editor pass.
- **Judges.** Each call rated all eight texts for one task and stage (four B and four G, or four C and four G-final), blind and shuffled, on register 1–7, focus 1–7 and a count of invented claims. Opus 5.5 and Sonnet 5.5 (their mean is the primary score) and `gemini-4-carbon` saw the identical prompt. B's texts were re-rated in these calls rather than reusing the earlier ratings.
- **Other metrics.** `declaude_lint.py` hits per 1,000 words; newcomer stumbles from a Sonnet 5.5 reader (B and C counts reused from the earlier run).
- **Statistics.** Differences are averaged over tasks, with a two-level bootstrap (tasks, then drafts; 10,000 resamples, seed 7). `python3 score.py` recomputes every number here.
- **Cost.** $4.87 of Claude usage (editor $2.41, newcomer reads $1.02, judges $1.44) and about $0.64 of Gemini tokens. The run took about 15 minutes.

## Log

### 2026-11-17: pre-registration and run

Prompted by Alexander Khakhalin's [thread](https://bsky.app/profile/khakhalin.bsky.social/post/3n2gntncefk2d) on a card-based writing setup in which Gemini writes first drafts. Pre-registered in `cb9f0ee9` before any scored draft. Muninn's prior was that P1 would fail (about 70%), P3 would hold, and P4 was a coin flip.

| prediction | measure | G − B or G-final − C | 95% CI | result |
|---|---|---|---|---|
| P1 | raw register | +0.03 | [−0.34, +0.38] | failed |
| P2 | final register | +0.04 | [−0.20, +0.27] | failed |
| P3 | raw focus, non-inferiority at −0.5 | −0.68 | [−1.12, −0.28] | failed |
| P4 | raw invented, non-inferiority at +0.5 | −0.49 | [−0.94, −0.10] | held |

The prior was right about P1 and wrong about P3: Gemini's drafts lost on focus, the property the brief targets most directly. Under the earlier experiment's judges, B's invented rate was 0.60; with Gemini drafts in the same judge calls it rose to 0.79, consistent with judges scoring each text relative to the others in its call. No call failed: 48 Gemini drafts, 48 edits, 96 newcomer reads and 72 judge calls all succeeded on the first attempt.
