# positive-writer-brief: results

Arms, metrics and the pre-registered predictions P1 to P5 are defined in [`README.md`](README.md). Muninn is the agent that writes this repo's prose; arm A drafts with today's context, arm B with the new brief, and A-final and C are those drafts after the same declauding editor pass.

## Answer

Does Muninn draft better from a brief of "do X, because the reader Y" statements with no banned phrases in it than from today's context, the voice signature plus the declauding skill, which name every banned phrase? Opus 5.5 wrote four drafts per arm per task, twelve tasks.

The brief did not reduce lint hits. Raw drafts from the brief had 5.34 hits per 1,000 words against 4.88 for today's setup, a difference of +0.46 with a 95% CI of [-1.60, +2.93]. <!-- @evidence[[results/summary.json#predictions.P1.diff]] -->
Focus held up: the blind judges scored the brief's drafts 0.24 higher on a 7-point scale, CI [-0.15, +0.65], which passes the pre-registered non-inferiority margin of -0.5. <!-- @evidence[[results/summary.json#predictions.P2.diff]] -->
Register went the other way, at -0.31 [-0.62, +0.02]. <!-- @evidence[[results/summary.json#comparisons.raw_B_vs_A.all.register.diff]] -->
After the same declauding editor pass on both arms, the lint difference was gone (+0.08), and the focus edge stayed at +0.28 [-0.02, +0.64]. <!-- @evidence[[results/summary.json#predictions.P4.diff]] -->

The CI is wide enough to hide an effect of two hits per 1,000 words either way, so the data show no priming at this resolution and cannot rule a small one out. The brief helps focus a little; it does not cut tics or replace the editor.

## Findings

1. P1 failed. Brief-B raw drafts had +0.46 lint hits per 1,000 words over arm A, CI [-1.60, +2.93], and were lower on only 5 of 12 tasks. (`results/summary.json`) <!-- @evidence[[results/summary.json#predictions.P1.diff]] -->
2. P3 failed. After the editor pass, C against A-final was +0.08 hits per 1,000 words, CI [-3.45, +3.22]. (`results/summary.json`) <!-- @evidence[[results/summary.json#predictions.P3.diff]] -->
3. P2 held. Brief-B focus on raw drafts was +0.24, CI [-0.15, +0.65]; Opus rated it +0.35 and Sonnet +0.13. (`results/summary.json`) <!-- @evidence[[results/summary.json#predictions.P2.diff]] -->
4. P4 held. C focus was +0.28 over A-final, CI [-0.02, +0.64], and C was not lower on 10 of 12 tasks. The largest gap was on the two blog tasks where the facts came as the author's own story: +0.75. (`results/summary.json`) <!-- @evidence[[results/summary.json#comparisons.final_C_vs_Afinal.author_traps.focus.diff]] -->
5. Brief-B drafts read slightly worse in register: -0.31 on raw drafts, CI [-0.62, +0.02], and lower on 9 of 12 tasks for both judges. The editor pass shrank this to -0.10. (`results/summary.json`) <!-- @evidence[[results/summary.json#comparisons.final_C_vs_Afinal.all.register.diff]] -->
6. Brief-B drafts ran 21 words longer on average (183 against 162), CI [+8, +36]. (`results/summary.json`) <!-- @evidence[[results/summary.json#comparisons.raw_B_vs_A.all.words.b]] -->
7. P5, exploratory: newcomer stumbles did not differ. Raw drafts gave +0.19 stumbles, CI [-0.48, +0.92], and the finals gave 0.00. (`results/summary.json`) <!-- @evidence[[results/summary.json#predictions.P5.diff]] -->
8. Most of brief B's extra lint hits are header hits: 17 against 11 in raw drafts, out of about 40 hits per arm in all. Brief B tells post headers to be declarative sentences, which it takes from the voice signature, and lint entry 7 (verdict headers) flags declarative headers. Some brief-B drafts also used headers in a README section where arm A used none. (`results/summary.json`) <!-- @evidence[[results/summary.json#by_category.raw_B.hits.header]] -->
9. The two judges agreed moderately, with Pearson r 0.671 on register and 0.646 on focus. (`results/summary.json`) <!-- @evidence[[results/summary.json#judges.register.pearson]] -->

## Method

The question is whether a positive drafting brief beats today's drafting context. The fixture is twelve tasks in `tasks.json`, two per surface: README section, PR body, blog section, results Answer, commit message and Bluesky post. Each task supplies its facts in the prompt. Six tasks invite focus failures. Four deliver the facts as a changelog or round log and ask for text that describes the thing. Two deliver the facts as the author's sequence of attempts and ask for a blog section.

| arm | drafter context | editor |
|---|---|---|
| A | voice signature + declauding `SKILL.md` | — |
| B | `brief_b.md` | — |
| A-final | A | declauding pass |
| C | B | declauding pass |

Drafter and editor were `claude-opus-5-5` at effort low, run as headless `claude -p` children with no tools. The editor had the skill, register.md, exceptions.md, the voice signature and the lint output, and ran workflow steps 1 to 4 and 6 by eye. Brief B passes `check_brief.py` with 0 hits against 160 lint regexes and 49 voice-signature tell strings.

Three metrics were scored:

- **Lint rate.** declaude_lint hits per 1,000 words, at distinct positions, without the density notes.
- **Blind judges.** Opus 5.5 and Sonnet 5.5 at effort medium rated register and focus from 1 to 7, with all eight texts for a task in one shuffled, unlabelled call. Each text's score is the mean of the two judges.
- **Newcomer read.** A Sonnet 5.5 reader counted stumbles in each text.

Differences are averaged per task and then over tasks. CIs come from a two-level bootstrap (tasks, then drafts) with 10,000 resamples and seed 7. Predictions were pushed in commit `c0ddaf07` before the run.

Limitations:

- The drafter saw no user preferences, memory or session context.
- Brief B mentions CCotw (Claude Code on the web, where Muninn's heavy work runs) and memory without defining either for a drafter that has no tools.
- The editor lacked the skill's scripts and the model-profile files.
- Four of the six focus-trap fixtures are partly invented.

Cost was $11.57 in Claude Code usage, about 10 minutes wall clock: drafts $1.91, edits $6.28, newcomer reads $2.00, judges $1.39.

## Log

### 2026-11-10: the single run

The README, brief, check and tasks were pushed as `c0ddaf07`. A two-draft pilot checked the plumbing and was discarded. All 432 children succeeded on their first attempt.

| comparison | metric | A side | B side | diff | 95% CI |
|---|---|---|---|---|---|
| raw B − A | lint /1k | 4.88 | 5.34 | +0.46 | [-1.60, +2.93] |
| raw B − A | focus | 5.62 | 5.87 | +0.24 | [-0.15, +0.65] |
| raw B − A | register | 6.03 | 5.72 | -0.31 | [-0.62, +0.02] |
| raw B − A | stumbles | 5.27 | 5.46 | +0.19 | [-0.48, +0.92] |
| C − A-final | lint /1k | 3.93 | 4.00 | +0.08 | [-3.45, +3.22] |
| C − A-final | focus | 5.62 | 5.90 | +0.28 | [-0.02, +0.64] |
| C − A-final | register | 6.02 | 5.92 | -0.10 | [-0.44, +0.24] |

Against the predictions: P1 failed, P2 held, P3 failed, P4 held, and P5 showed no direction. `python3 score.py` prints the full table. It also covers the trap subsets.

The reading at the time was that the priming hypothesis from caps-emphasis does not show up as lint hits in Opus drafts. The do-because form may help focus. It costs a little register and some length, and the editor recovers most of the register.

What this implies for the two documents, neither of which was changed here:

- Moving declauding's "load before drafting" instruction to "run after drafting" would cost nothing measurable in tics.
- A short do-because brief could replace the Scan-for lists in the drafting context for the focus gain. The editor pass still has to run.
- The voice signature's declarative-header rule and lint entry 7 disagree. Either the lint should exempt post headers or the rule needs reconciling.
