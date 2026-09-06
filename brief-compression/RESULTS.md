# Brief compression: telegraphic briefs to Sonnet and Haiku subagents

**Date:** 2026-09-06
**Question (Oskar):** GPT Astra writes its Luna sub-agents in telegraphese
("Need yourfinal audit tests+bothrealprepsrunandruff ASAP, then
second/thirdcommits"). Would Muninn (Fable/Opus) gain by briefing Sonnet and
Haiku subagents the same way?
**Answer:** No. On the task where the styles separate, the telegraphic brief
saved 106–110 brief tokens per run and the receiving subagent then spent
1,500–6,300 *more* tokens, 2–3x the tool calls, and 2–2.6x the wall time,
with a lower success rate. Plain prose was the only style with zero failures
on both models. The token count of the brief is the wrong quantity to optimise.

Pre-registered predictions are in [`PREDICTIONS.md`](PREDICTIONS.md); what
went wrong along the way is in [`ERRORS.md`](ERRORS.md); `recheck.py` checks
every number below against `data/`.

## Setup

Two file-manipulation tasks, three brief styles carrying identical
information, two receivers (Haiku 4.5 and Sonnet 5 as Claude Code
`general-purpose` subagents, dispatched from this session's Agent tool with
the brief as the entire prompt). Each run has its own directory and its own
opaque id; the subagent's reply and the files it wrote are scored by script.

| style | what it is | Task A brief | Task B brief |
|---|---|---|---|
| prose | full sentences, articles, "please" | 184 tok | 287 tok |
| structured | one bullet per rule, no articles, no fused words | 119 tok | 218 tok |
| telegraphic | Astra's register: dropped copulas, fused nouns, `w/`, `excl`, `lc`, `casesens` | 99 tok | 178 tok |

Token counts are `o200k_base` as a proxy; the Claude tokenizer is not
reachable from this session. Stripping the spaces between words, which the
Astra screenshots appear to do, was measured first: 53 tokens with spaces, 53
without. BPE folds the leading space into the word token, so that habit saves
nothing. The telegraphic arm here keeps spaces and fuses only where the
screenshots did ("exactly3keys", "Append1line", "Inputreadonly").

**Task A** (9 checks): count lines containing `ERROR` but not `DEBUG`, write a
3-key JSON with the count and the input's SHA-256, append one log line, touch
nothing else, reply `DONE`. 6 runs per cell, 36 runs.

**Task B** (13 checks): from a 300-row CSV keep rows with `status == 200`,
action exactly `GET` or `PUT`, and user not starting with lowercase `svc_`
(the data carries a `Svc_ops` user and a lowercase `get` action as decoys);
sum bytes per user; top 5 with a tie-break rule; write a TSV with an exact
header, a 3-key JSON, a log line carrying the kept-row count, and reply
`OK <count>`. 10 runs per cell, 60 runs.

## Results

Task A hit ceiling: 33 of 36 runs passed all 9 checks and the three misses
were Sonnet leaking a sentence of self-talk before `DONE` (prose 1, structured
2). Task A was capped at 6 per cell once the first 22 runs came back 22/22
(PREDICTIONS.md, amendment 1) and Task B was added.

Task B, success = 13/13 checks, n = 10 per cell:

| receiver | prose | structured | telegraphic |
|---|---|---|---|
| Haiku 4.5 | **10/10** | 9/10 | 8/10 |
| Sonnet 5 | **10/10** | 2/10 † | 6/10 |

† 8 of the 10 Sonnet-structured runs failed exactly one check, `top_user`:
they wrote `{"user": "jo", "bytes": 30545}` where a string was wanted. The
structured brief said `top_user (first report row)`; the prose brief said
"the first user in the report". That is my wording, not the receiver's
reading — see ERRORS.md #1. With that check set aside, Sonnet-structured is
10/10 and the structured arm matches prose on both models.

Fisher exact, two-sided, against prose on the same receiver:

| comparison | p |
|---|---|
| Sonnet telegraphic 6/10 | 0.087 |
| Sonnet structured 2/10 (as scored) | 0.0007 |
| Haiku telegraphic 8/10 | 0.47 |
| Haiku structured 9/10 | 1.0 |
| telegraphic pooled over both receivers, 14/20 vs 20/20 | 0.020 |

### The telegraphic failures, run by run

All four Sonnet-telegraphic misses are the same miss. `user !startswith lc
"svc_"` was read as a case-insensitive prefix, so the `Svc_ops` rows were
dropped. In each of the four runs the reported `rows_kept` equals, to the row,
the count under case-insensitive exclusion (43 vs gold 47, 55 vs 61, 41 vs 45,
and one more), and the top-5 report was still correct because `Svc_ops` never
reached the top 5. The prose brief said "the lowercase prefix", the structured
brief said `lowercase "svc_"`; neither was misread once in 20 runs. The
abbreviation `lc` is where the information was lost, which is what P6
predicted, except P6 predicted it for Haiku.

Haiku's two telegraphic misses are different. One run wrote `rows_kept: 5`
(the "exactly5rows" number, not the filter count), replied `OK 5`, and
rewrote the CSV with `\n` line endings where the generator had written
`\r\n` (ERRORS.md #2). The other run appended `ok 0` and then `ok 58` to the
log. Its Haiku-structured miss wrote `rows_kept: 11`, the number of distinct
users. These are not abbreviation failures; they are Haiku picking the wrong
integer to call N under any phrasing that does not spell N out twice.

### Task B cost per run

`sub_tokens` is the subagent's total token usage as reported by the Agent
tool (a Claude Code subagent starts at roughly 32.5k before it reads the
brief; see METHODS.md). `brief` is the o200k count of the prompt.

| receiver | style | brief | sub_tokens | tool calls | wall ms |
|---|---|---|---|---|---|
| Haiku | prose | 289 | 38,959 | 2.0 | 14,976 |
| Haiku | structured | 218 | 42,933 | 3.6 | 22,739 |
| Haiku | telegraphic | 177 | 45,225 | 4.1 | 39,531 |
| Sonnet | prose | 284 | 50,217 | 1.0 | 6,938 |
| Sonnet | structured | 219 | 50,253 | 1.4 | 7,482 |
| Sonnet | telegraphic | 178 | 51,736 | 2.9 | 13,653 |

Telegraphic saved 106 (Sonnet) and 112 (Haiku) tokens on the brief. The
receiver then spent 1,519 (Sonnet) and 6,266 (Haiku) more tokens than under
the prose brief, made 2–3x the tool calls, and took 2–2.6x as long. The extra
tool calls are the receiver re-reading the data or the brief to resolve what
"lc", "casesens", "postfilter" and "row1" mean. P8 (telegraphic runs use more
tool calls) held on both receivers and on both tasks.

The one place the brief's own token count matters is the orchestrator's
output, which is billed at the orchestrator's rate. On this session that is
Fable output against Haiku input. Even at that exchange rate, 110 tokens of
Fable output does not cover 6,266 tokens of Haiku, and it does not cover the
retry when the run fails.

## Predictions scored

| | prediction | outcome |
|---|---|---|
| P1 | Sonnet Task A: no style difference | held (6/6, 4/6, 6/6; the 2 misses are self-talk before DONE, not task failures) |
| P2 | Haiku Task A: telegraphic loses ≥3 | not observed; ceiling, 6/6 everywhere |
| P3 | Haiku Task A: structured ties prose at fewer tokens | held, at ceiling |
| P4 | no arm modifies the input | **failed** once: Haiku telegraphic ba9aff3 rewrote the CSV's line endings |
| P5 | Sonnet Task B: every cell ≥ 8/10 | **failed**: structured 2/10 (brief wording), telegraphic 6/10 |
| P6 | Haiku Task B: telegraphic loses ≥3, on `lc`/`casesens` and `reply` | direction held, size did not (−2, not −3); the `lc` mechanism showed up on Sonnet, 4 of 10 |
| P7 | Haiku Task B: structured within 1 of prose | held (9 vs 10) |
| P8 | telegraphic uses more tool calls on both receivers | held: 4.1 vs 2.0 (Haiku), 2.9 vs 1.0 (Sonnet) |

## Dropped referents in the Astra screenshots

Astra's briefs also lean on referents that only resolve inside a persistent
channel with the same sub-agent ("Checkpoint4saved", "E179docmain owns",
"finalim_end label=100"). A fresh Agent-tool subagent has no prior turn, so
those would have to be spelled out or put in a file and referenced by path,
which is the rule Oskar set this morning for orchestrator context anyway
(memory 16d9e7fb). This experiment did not test dropped referents; every
style here carried the full information. It tested only the grammar, and the
grammar alone was enough to lose the case-sensitivity rule on 4 of 10 Sonnet
runs.

## Caveats

- n = 10 per cell. The pooled telegraphic-vs-prose result (14/20 vs 20/20,
  p = 0.02) is the only comparison that clears 0.05; the per-receiver ones
  do not.
- One task family. A brief with more numeric parameters and fewer
  natural-language rules might compress better. A brief with more rules would
  likely compress worse.
- The structured arm carried a wording defect on `top_user` that the other
  two arms did not, so its Sonnet row is not a clean measurement of structure.
  Its Haiku row, and both rows with that check excluded, are.
- Token counts on the briefs are o200k, not Claude's tokenizer. The ordering
  prose > structured > telegraphic will not change; the exact numbers will.
- `sub_tokens` is the Agent tool's reported total and includes the ~32.5k
  fixed subagent floor. Differences between cells are the meaningful part.
- The generator wrote CSVs with `\r\n` line endings (Python `csv` default).
  Every reader handled it; one Haiku run normalised it on write and was
  scored as modifying the input. That is a fair reading of "read-only" but a
  harsher one than intended.

## Files

- `gen.py`, `score.py` — Task A fixtures, briefs, scorer
- `gen_b.py`, `score_b.py` — Task B
- `data/task_{a,b}_{briefs,scores,replies,usage}.json` — every brief with its
  gold, every check outcome, every reply, every usage line
- `recheck.py` — asserts the tables above against `data/`
- `PREDICTIONS.md`, `ERRORS.md`

Rerun: `python3 gen_b.py <root> --n 10`, dispatch each brief in
`<root>/briefs.json` to a subagent of the named model with the brief as the
whole prompt, record replies as `{run_id: reply}`, then
`python3 score_b.py <root> replies.json`.
