# model-advisor-replay: the first-prompt model advisor on archived sessions

The model advisor is a Claude Code hook that sends a session's first prompt to
TypeSafe's Jev classifier, asks which model tier the task needs, and suggests a
stronger model when the session is on a weaker one. Its source is
`scripts/model_advisor.py` in the private claude-workspace repo; the same rules
ship publicly as the `model-advisor` mod in
[oaustegard/claude-code-mods](https://github.com/oaustegard/claude-code-mods/tree/main/model-advisor).
How it decides is written up at
[muninn.austegard.com/blog/how-the-model-advisor-decides.html](https://muninn.austegard.com/blog/how-the-model-advisor-decides.html).

Two questions, asked on 2026-10-08:

1. Did the advisor see which model each recent session was on?
2. What did the 2026-10-07 rubric change do to its calls? That change redrew
   the Sonnet/Opus line on whether the prompt names its deliverable (Sonnet) or
   sets a goal whose work has to be discovered (Opus), after Sonnet 5.5 stopped
   to ask permission on an audit-and-update task Jev had called Sonnet at 72%.

## Design

The archive is the claude-workspace `transcripts` branch, one JSONL file per
session. Only interactive sessions count: scheduled Routines and
`remote_trigger` runs are left out, as the advisor skips them. That leaves 74
sessions from 2026-09-10 to 2026-10-08.

**Part 1** reads the advisor's own one-line verdict from each transcript since
2026-09-28 (the hook prints one on every first prompt it classifies). A line
that says "session on unknown model" means the hook could not tell which model
the session was on.

**Part 2** replays each session's first prompt through the advisor twice: once
with the 2026-09-28 rubric (claude-workspace `bb35fbb`) and once with the
2026-10-07 rubric (`72b146d`). Both texts are in [`rubrics.json`](rubrics.json).
Every session is scored as if it had started on Sonnet (`replay --from
sonnet`), with the same thresholds: suggest Opus when Jev puts at least 0.75 on
Opus-or-stronger, and stay silent when Jev puts less than 0.5 on "the prompt
describes the work". That is 148 Jev calls, one replicate.

There are no correctness labels. The grader labels behind the September
calibration were never committed. The one hindsight signal in the archive is a
session that switched to a stronger model mid-task, and there is exactly one:
session `7c199534`, the session that prompted the rubric change.

## Results

### Part 1: whether the advisor saw the session's model

| first prompts | model known | model unknown | classifier error | no verdict line |
|---|---|---|---|---|
| 2026-09-28 to 10-02 | 12 | 0 | 0 | 4 |
| 2026-10-03 to 10-08 | 1 | 21 | 3 | 8 |

From 2026-10-03, Claude Code on the Web starts a preloaded CLI process
(`claude --preload <socket>`) with no `--model` argument, and the SessionStart
hook input has no model field either. The hook read the model from that argument,
so it saw the model only when a model switch fired before the first prompt: 1
of 22 classified sessions. An unknown model rules out every upgrade, so the
advisor could not suggest Opus in those 21 sessions. Session `7c199534` was one
of them.

The three errors are one HTTP 503 from the classifier and two sessions with no
Jev credentials in the environment.

claude-workspace `4f8f278` fixes the blind case for the web and mobile clients,
where the advice already goes to the model instead of blocking the prompt. With
the model unknown, the hook scores the upgrade as if the session were on Sonnet
and words the note as a condition. The model reading it knows which model it is
and hands the heavy work to a stronger subagent only when it is below the
suggested tier. In the terminal an unknown model still never blocks a prompt.

### Part 2: calls under the two rubrics

| decision, all 74 sessions scored from Sonnet | 2026-09-28 rubric | 2026-10-07 rubric |
|---|---|---|
| suggests Opus | 13 | 22 |
| silent: prompt only points at the work | 36 | 37 |
| silent: leans Opus, below 0.75 | 8 | 8 |
| silent: Sonnet is enough | 13 | 4 |
| silent: Haiku looks enough | 4 | 3 |
| Jev's most likely tier is Opus | 30 | 52 |
| Jev's most likely tier is Sonnet | 30 | 11 |

Jev's probability of Opus-or-stronger rose by 0.21 on average (median 0.15)
per prompt. The rise is broad rather than targeted at goal-shaped prompts: a
one-line request to install a language runtime went from 0.07 to 0.40.

On session `7c199534` the probability went from 0.05 to 0.41. That is still
below 0.75, so the new rubric would not have suggested Opus for the prompt that
motivated it.

Oskar started 49 of the 74 sessions on Opus, 16 on Sonnet, 8 on Fable and 1 on
Haiku, so "sessions start on Sonnet" describes the advisor's policy more than
his practice. Of the 22 suggestions under the new rubric, 14 land on sessions
he had already started on Opus; 5 land on Sonnet sessions (4 under the old
rubric), 2 on Fable sessions and 1 on the Haiku session.

Half the first prompts (37 of 74) only point at the work, such as "Address
<issue link>" or a memory id, and the advisor stays silent on them under either
rubric.

## Limits

- No correctness labels, so neither rubric can be scored as right or wrong.
  The counts say only where the calls moved.
- One replicate. Jev reruns flip a few values near any threshold.
- The new rubric was written with `7c199534` in view, so that session is not a
  held-out test.

## Reproduce

Inside a claude-workspace clone (the archive is private):

```bash
git fetch --shallow-since=2026-09-10 origin transcripts
git log --format='%H %s' FETCH_HEAD | grep -v scheduled | awk '!seen[$4]++ {print $1}' > commits.txt
python3 notes.py commits.txt > notes.tsv
python3 extract.py commits.txt > rows.jsonl
git show bb35fbb:scripts/model_advisor.py > advisor_0928.py
python3 advisor_0928.py replay rows.jsonl --from sonnet > replay_0928.jsonl
git show 72b146d:scripts/model_advisor.py > advisor_1007.py
python3 advisor_1007.py replay rows.jsonl --from sonnet > replay_1007.jsonl
python3 compare.py notes.tsv rows.jsonl replay_0928.jsonl replay_1007.jsonl > results/summary.json
python3 recheck.py
```

## Files

- `notes.py`: the advisor's verdict line per archived session
- `extract.py`: first prompt, entrypoint and model sequence per archived session
- `compare.py`: aggregates both into `results/summary.json` (counts only; the prompts stay private)
- `rubrics.json`: the two rubric texts and thresholds
- `recheck.py`: checks the numbers in this file against `results/summary.json`
