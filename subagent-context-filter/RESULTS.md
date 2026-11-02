# Subagent context filter: pass the task, let Jev pick the transcript

**Question.** When an orchestrator delegates, can it write only the task and
let a filter pass the parent transcript's relevant chunks, instead of writing
a background brief or forking the whole context? (Oskar, 2026-09-22: *"wrap
subagents in a jev-aided transcript-parser which per transcript chunk
determined whether that chunk should be included for a given subagent task"*,
then *"page in 30k token increments"*.)

**Answer.** Yes, on fact-carrying tasks. A fresh subagent given the task plus
~20k tokens of Jev-selected chunks got 99 of 105 required facts right. Given
the full transcript (a fork proxy, median ~81k tokens) it got 102. The paired
difference is −2.6 to −2.9 points, CI reaching zero, worse on 3 of 32 tasks and
never better. A brief the same model wrote from the full transcript got 82 of
105. That whole gap is five tasks (16%) where the brief pointed the subagent at
a lookup ("check the `pr-workflow` config entry", "find the PR with gh")
instead of stating the fact. On the other 27 tasks every arm lands within two
facts of every other (82 to 84 of 87). The filter took a median of 1.1 s and
$0.002 per delegation. The brief took a median 528 output tokens and 23 s of
generation.

Paging made no measurable difference on these sessions (19 to 97 chunks): the
condensed single window and the 4x paged view both reached 99/105. It did on
the one 181-chunk session tried outside the eval, where the condensed view
ranked the answer chunk 6th and the paged view 1st. The filter defaults to the
paged view because Jev costs almost nothing either way.

## Setup

- **Corpus.** 8 sessions from this account's transcript archive
  (`claude-workspace` `transcripts` branch), 19 to 97 chunks each, 54k to 375k
  characters. A chunk is one user message, one assistant text block, one
  harness-injected turn (`isMeta`: skill bodies, hook feedback) or one tool call
  paired with its result. Secret values from the environment and token-shaped
  strings are redacted before any chunk leaves the machine (8 redactions across the 8 sessions).
  The transcripts are private, so `data/` is gitignored; `results/summary.json`
  carries per-task counts with no transcript text.
- **Tasks.** 8 Sonnet labellers, one per session, each read its whole session
  and wrote 4 tasks an orchestrator might delegate at the end, phrased by name
  without the facts ("the xr search latency numbers"), with 2 to 4 required
  facts as regexes plus the chunk ids holding them. Labellers never saw any Jev
  output. 32 tasks, 105 facts, 95 must-have chunk ids. No task text matches its
  own fact regexes.
- **Arms.** Same executor for all: `claude -p --safe-mode --tools ""
  --strict-mcp-config --model sonnet`, one turn, from an empty directory, so no
  settings, CLAUDE.md, tools or MCP.
  - `none`: task only.
  - `jev1`: task + Jev-selected chunks, condensed view (each chunk clipped to
    ~1.5k chars for Jev), 20k budget.
  - `jev4`: same with a 4x view, paged into ~27k-token windows.
  - `full`: task + the whole transcript.
  - `brief`: the same model reads the whole transcript and writes the brief
    it would pass; the executor gets task + brief.
- **Filter.** `jevfilter.py`. Each window carries the task, the user messages
  (clipped) and a one-line index of the whole session, plus its own chunks;
  one Noul per chunk. User messages are force-kept, clipped, up to a quarter
  of the budget; everything else is ranked by score into the budget.
- **Scoring.** Fact regexes first. The regexes turned out too literal (`3868`
  missed "3.9 s"; "3 failed, 5 passed" missed "5 passed, 3 failed"), so every
  regex miss went to a blind adjudicator (Sonnet 5, batches of 20, arm never
  shown, critique before verdict). It was validated on items mixed into the
  same batches: 40 regex hits (TPR 100%) and 40 facts checked against a
  different task's deliverable from the same session (TNR 95%). It upgraded
  57 of 95 misses. Regex-only counts differ in the middle of the ranking (below): the filter
  arms and `full` swap places, and `brief` is last either way.

## Results

| arm | facts found | tasks fully correct | 95% CI (tasks) | context passed, est. tokens (median) |
|---|---|---|---|---|
| none | 0/105 | 0/32 | 0%-0% | 7 |
| jev1 | 99/105 (94%) | 27/32 | 72%-97% | 20,042 |
| jev4 | 99/105 (94%) | 26/32 | 69%-94% | 20,033 |
| full | 102/105 (97%) | 29/32 | 78%-100% | 80,743 |
| brief | 82/105 (78%) | 23/32 | 56%-88% | 675 |

Regex-only, before adjudication: none 0, jev1 83, jev4 86, full 83, brief 73.

Paired per-task fact recall against `full` (bootstrap 95% CI): jev1 −2.9%
(−6.2% to 0.0%), jev4 −2.6% (−5.7% to 0.0%), brief −17.7% (−32.3% to −5.2%).

| filter | windows (median) | Jev input tokens (median) | Jev $ (median) | seconds (median) | must-have chunks kept |
|---|---|---|---|---|---|
| jev1 | 1 | 23,216 | $0.0010 | 0.9 | 85/95 (89%) |
| jev4 | 2 | 44,466 | $0.0019 | 1.1 | 87/95 (92%) |

**The brief's failure.** In 5 tasks the brief-fed executor tried to call a
tool it did not have (`muninn_config key: pr-workflow`, `gh pr list`, `find
/home/user -iname '*claude-skills*'`, write to a scratchpad). No other arm
did this on any task. The writer had deferred the fact to a lookup. With tools
some of those lookups would succeed (a PR number is on GitHub) and some could
not (a latency measured in the session exists only in its transcript), and
every one costs the subagent tool calls it would not otherwise make. Excluding
those 5 tasks: jev1 83/87, jev4 83/87, full 84/87, brief 82/87.

**Cost of the handoff at Opus 5.5 prices** ($4 in, $20 out, 1-hour cache write
$8, read $0.20 per MTok), per delegation:

| path | orchestrator pays | subagent pays |
|---|---|---|
| filter | $0.002 Jev, ~1 s | a ~20k-token cache write on its first turn ($0.16 on Opus, $0.08 on Sonnet), then $0.004 per turn to re-read it |
| brief | 528 output tokens ($0.011) and the generation time: 23 s median here, reading the transcript cold | almost nothing: ~700 tokens |
| fork | nothing | ~81k inherited tokens read at $0.016 per turn, every turn |

So the brief is the cheapest path in tokens. The filter replaces the
orchestrator's writing time and the pointer-instead-of-fact failure with a
cache write in the subagent. Against a fork it passes a quarter of the tokens
at a 3-point cost in recall, and becomes cheaper than the fork after about
13 subagent turns ($0.16 / ($0.016 − $0.004)) on Opus.

## What this does not show

- **Tool-less, one-turn executors.** The deliverables are fact-carrying
  write-ups. Real subagents have tools and run many turns. That makes the
  brief's lookups partly recoverable, and it makes the per-turn cost difference
  against a fork matter more than it does here.
- **Tasks written to depend on the transcript.** A task that needs nothing
  from the session gains nothing from the filter; the skill routes those to
  `[no-context]`.
- **One budget.** 20k tokens throughout; recall against budget was not swept.
- **Labels by one model family.** Sonnet wrote the tasks, labelled the
  must-have chunks, executed and adjudicated. The adjudicator's 95% TNR means
  a couple of the 57 upgrades are likely false. Regex-only, jev4 leads (86)
  and jev1 ties full (83), so the filter-versus-full gap is inside the
  scoring noise; `brief` is last under both scorings.
- **n = 32.** The task-level CIs are wide (the jev and full intervals
  overlap almost entirely).

## Found along the way

- **TypeSafe's edge WAF blocks some transcript content.** Through the AI
  Gateway it arrives as `HTTP 402 Payment error from model using BYOK` wrapping
  a Cloudflare "Sorry, you have been blocked" page. It is deterministic for the
  same bytes: one session's condensed window was blocked on every retry while
  its 4x-view windows passed. The filter now splits a blocked window in half
  until the offending chunk is isolated (13 calls, 7 s, 1 chunk scored as
  neutral 0.5 instead of losing the whole window).
- **Gateway rate limit.** `HTTP 429 Rate limited, code 2003` once bisection
  fanned out. METHODS.md already said to start gateway concurrency at 2; the
  filter now does, and honours Retry-After.
- **Force-keeping user-role messages filled the budget.** See ERRORS.md #1.
- **`create_session` children need the task after the context.** The hook
  covers `mcp__*__create_session` (an MCP tool's `updatedInput` verified with a
  stub server and on the real remote server). Two Haiku cloud sessions got the
  appended context (first-turn token counts match) and stalled at need-input,
  one reading it as truncated and one as no request at all. With the block
  labelled as reference data and the task repeated after it, the third answered
  correctly on its first turn. n = 3; a direction, not a rate.

## Prior art

The 101-project roundup and the awesome-jev lists (reviewed in memories
`dd506da5`, `0070d581`) have compaction by Jev (`fast-jev-compaction`,
`yoshi`, `pi-fast-jev-compaction`): score each tool call for whether the main
session still needs it. I did not find task-conditioned selection for a
subagent handoff in those lists; no separate web search was run. The
components exist; this composition was not found there.

## Files

`chunking.py` (transcript to chunks, redaction), `jevfilter.py` (windows,
bisection on WAF blocks, rank-to-budget), `context_hook.py` (PreToolUse hook),
`preview.py` (what the hook would pass), `run_eval.py`, `adjudicate.py`,
`analyze.py`, `recheck.py`, `results/summary.json`, `ERRORS.md`. Shipped as the
`delegating-with-context` skill in `oaustegard/claude-skills`.

## Follow-up, 2026-11-02: a hierarchical filter

**Question.** DecisionRAG (arXiv 2610.35868) retrieves with Jev by descending a
tree: it scores coarse nodes, keeps the top k and scores only the children of
what it kept. Does that shape, applied to a transcript, match the flat filter?

**Answer.** On these 8 sessions it matches and costs 36-41% fewer Jev tokens.
Spending the saved room on a 2x larger chunk view (`hier8`) found 107 of 108
facts, against 105 for the flat `jev4` filter and 106 for the full transcript.
Every arm sits near the ceiling, so none of the differences is significant: the
paired `hier8` minus `jev4` CI is +0.0% to +7.1%, and the gain is one task.

**Filter (`hierfilter.py`).** Level 1 splits the session into segments (a new
segment at each user message, and every 8 chunks) and shows Jev a short digest
of each, so a whole session fits one window. It keeps the top 40% of segments,
never fewer than 3. Level 2 scores only the chunks inside kept segments, at
view scale 4 (`hier4`) or 8 (`hier8`). Chunks in dropped segments are never
scored. The budget packing is the flat filter's `select`. A score threshold at
level 1 was tried first and dropped: on a single-topic session every segment
scored 0.57 or more, so nothing was pruned.

**Labels are new.** The September tasks and regexes lived in the gitignored
`data/` and did not survive. The same 8 session snapshots (same commits, same
chunk counts) were relabelled by 8 tool-less Sonnet labellers using the
September prompt (`label.py`). 34 tasks were written; 6 were dropped for a
regex that missed its own evidence, leaked into the task text, or a fact
count outside 2-4. That left 28 tasks and 108 facts. `jev4` and `full` were
re-run on them, so the comparison is within this run only; the numbers do not
line up with the September table.

| arm | facts found | tasks fully correct | 95% CI (tasks) | context passed, est. tokens (median) |
|---|---|---|---|---|
| none | 0/108 (0%) | 0/28 | 0%-0% | 7 |
| jev4 | 105/108 (97%) | 26/28 | 82%-100% | 20,051 |
| hier4 | 106/108 (98%) | 26/28 | 82%-100% | 19,951 |
| hier8 | 107/108 (99%) | 27/28 | 89%-100% | 19,936 |
| full | 106/108 (98%) | 27/28 | 89%-100% | 80,744 |

Regex-only, before adjudication: none 0, jev4 96, hier4 98, hier8 103, full 99.
Adjudicator: TPR 95% on 40 regex hits, TNR 90% on 40 wrong-task deliverables;
28 of 36 misses upgraded. `hier8` leads under both scorings.

| filter | Jev calls (median) | Jev input tokens (median) | Jev $ (median) | seconds (median) | chunks scored at level 2 (median) | must-have chunks kept |
|---|---|---|---|---|---|---|
| jev4 | 2 | 50,954 | $0.0021 | 1.0 | all | 111/116 (96%) |
| hier4 | 2 | 30,046 | $0.0013 | 1.1 | 30 | 103/116 (89%) |
| hier8 | 2 | 32,620 | $0.0014 | 1.1 | 30 | 110/116 (95%) |

`hier4` lost 13 must-have chunks: 9 pruned with their segment at level 1 and 4
out-ranked at level 2. Its executor still found 106 facts, because most facts
appear in more than one chunk. `hier8` lost 6 (5 at level 1). Level 1 sees the
same digests in both arms, so the 9-versus-5 difference there is not the view
scale: the two arms kept different segments on 8 of 28 tasks, from identical
input. Jev's scores vary between calls enough to reorder segments near the 40%
cut. Only the level-2 difference (4 versus 1 out-ranked) is attributable to
the larger view.

**Limits.** Sessions are 19 to 97 chunks, where the flat filter needs only 1-2
windows; the hierarchy's case is longer sessions, which were not tested. The
eval sits at the ceiling (`full` misses 2 facts), so it cannot separate arms
that differ by a fact or two. The keep fraction (40%) and segment length (8)
were set once, not swept. One run per arm; Jev's level-1 ranking varies
between runs.

**Verdict.** Not worth replacing the shipped filter on recall. Worth it on
cost only if Jev spend matters, which at $0.002 per delegation it does not. It
becomes worth testing again on sessions past ~150 chunks, where the flat
filter pages into 4 or more windows.

Files: `hierfilter.py`, `label.py`, `analyze_hier.py`, `results/summary_hier.json`;
`run_eval.py` gained the `hier4`/`hier8` arms and, with `adjudicate.py`, an
`EVAL_DATA` override so transcript-derived data can live outside the repo.
