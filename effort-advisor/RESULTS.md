# Would a real-time effort / model / delegation advisor for Claude Code be worth it?

Study run 2026-11-15 on Oskar's own transcripts. Short answer: **no for the advisor, no for the NPU spike; one cheap setting change is worth testing instead.**

## Data
- 366 transcripts (41 on this Mac, 325 from the `transcripts` branch of claude-workspace), 939 turns, **656 typed by a human** in 191 sessions. 76% of human turns ran in cloud sessions (416 from mobile), where nothing on the Mac can help.
- 63% of human turns ran on Opus 5.5 at medium effort. Model was switched by hand between turns 4 times; `/model` typed 3 times.

## What was measured
| Question | Method | Result |
|---|---|---|
| How often did a turn run at too low an effort? | Sonnet 5.5 and Opus 5.5 each judged all 656 turns with the outcome in view | Both said "too low" on **11 turns (1.7%)**. Of the 66 turns that ran one level below the judged need, 0 were corrected or failed |
| How often too high? | same | Both said "too high" on 97 (14.8%). Those turns are cheap: turns needing only low effort hold 2% of output tokens |
| Where do the tokens go? | output tokens by judged need | Turns needing high or xhigh are 28% of turns and **79% of output tokens** |
| Can Haiku call the effort from the prompt alone? | Haiku 5.5, low effort, sees what a hook sees; scored on the 458 turns where both judges agree | exact 0.61 (always-medium 0.44), within one level 0.97. "Needs high or more": precision 0.59, recall 0.76. Among turns run at medium or below: precision 0.48 |
| Can a tiny model? | TF-IDF + logistic regression, 5-fold | exact 0.50; "needs high or more" precision 0.57, recall 0.42 |
| Could Opus sessions have run on Sonnet? | stricter of the two judges, per session | 46% of Opus turns needed Sonnet or less, but they hold 10% of Opus output tokens; whole sessions that never needed Opus hold **1%**. Haiku's first-prompt call under-rates the session's hardest turn 29% of the time |
| Missed delegation? | both judges | 8 turns (1.2%). Haiku's delegate call: precision 0.20 |
| Stuck mid-turn? | rule signals over every tool call, 25 episodes read | signals fire on 3.9% of turns, about 20% of fires are real: roughly 1% of turns |
| Does higher effort change the answer? | 35 text-answer turns replayed with the same history at low and xhigh on the original model, and on Haiku at medium; Opus judged pairs blind | xhigh beat low on 27, **materially on 10 of 35 (29%, 95% interval 16-45%)**, never materially worse. Haiku was materially worse than xhigh on 23 of 35 (66%). xhigh replies were 25% longer |
| Does anything predict which turns gain? | cross the replay with the labels | Haiku foresight: gain on 5 of 13 it called high, 5 of 22 it called low or medium. Opus hindsight: 3 of 4 high, 6 of 17 medium, 1 of 14 low. Too few to separate |

## Reading
1. **The two methods disagree, and the disagreement is the finding.** Judges looking at a finished turn almost never say it needed more effort. A replay at higher effort is judged materially better about three times in ten. A judge cannot see the better answer that was not written.
2. **A per-turn advisor adds little over a setting.** Haiku's call does not pick out the turns that gain (38% against 23%, on 35 turns). Raising the default for conversational turns captures all the gains with no classifier.
3. **Routing down saves almost nothing.** Easy turns are cheap whatever they run on. The tokens are in the long hard turns, where nobody proposes less effort.
4. **Delegation and stuck-loop monitors have nothing to catch** at about 1% each.

## Is step 3 (a trained spike on the Neural Engine) warranted? No.
- Speed is not the constraint: advice can be computed in the background and delivered at the first tool call, which is what `turn-advisor/` does.
- There are no trustworthy labels to train on. The two judges agree on effort 70% of the time, and the replay contradicts them on what "needed" means.
- The decision it would make is worth little (points 2 and 3).

## What to do instead
- **Test `high` as the default effort for interactive sessions** (currently medium on the desktop app, low in claude-workspace with the escalating skill). The replay supports it for question-and-answer turns. It did not cover tool-heavy turns, which hold most of the tokens, so check cost there before keeping it.
- **Do not send conversational turns to Haiku.**
- Leave the first-prompt model advisor off rather than porting it from Jev to Haiku: the sessions it could move hold 1% of tokens.

## Limits
- Hindsight labels are two models' opinions; gold is their agreement (458 of 656 on effort).
- Replays were no-tool, from a rendered conversation (19 of 35 with the original turn's tool results pasted in), run as subagents inside Claude Code, not through the API. The judge saw no ground truth and may favour fuller answers. One judge, one run, 35 turns. The Sonnet arm was not run.
- Foresight was labelled in batches of 25, not one call per prompt as a hook would.
- The standalone `claude` CLI on this Mac was logged out, so no hook was exercised against a live model and Haiku latency is unmeasured. `hooktest/run_tests.sh all` runs those tests after login.

## Files
- `extract_turns.py`: transcripts to one record per turn (`data/turns_trace.jsonl`)
- `analyze.py`: joins `data/fore`, `data/hind_sonnet`, `data/hind_opus`; run with `uv run --no-project --with scikit-learn --with numpy python analyze.py`
- `replay/`: `items.jsonl`, `prompts/`, `results.json`
- `data/stuck_report.md`, `hooktest/FINDINGS.md`
- `turn-advisor/turn_advisor.py`: the hook. Background Haiku call on each prompt, advice delivered at the next tool boundary, every decision logged. Ships in `shadow` mode (logs, injects nothing); `TURN_ADVISOR_MODE=act` turns delivery on. 6 offline tests pass; not yet run against a live session.
