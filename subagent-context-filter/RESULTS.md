# Subagent context filter: pass the task, let Jev pick the transcript

When an orchestrating agent hands work to a subagent, the subagent starts
blank. The `delegating-with-context` skill lets the orchestrator write only the
task: a PreToolUse hook (a script Claude Code runs before the tool call) reads
the parent session's transcript, asks Jev which transcript chunks the task
needs, and appends those chunks to the subagent's prompt, up to a 20k-token
budget. Jev is a hosted decision model that answers many yes/no questions over
one shared text in about a second. A chunk is one message, or one tool call
with its result. The **flat** filter asks Jev about every chunk. The
**hierarchical** one (after DecisionRAG, arXiv 2610.35868, which retrieves
documents by descending a topic tree) first scores segments of the session and
then asks only about the chunks inside the best segments.

## Answer

Does the filtered transcript carry the facts a subagent needs, and does the
hierarchical filter beat the flat one?

The first holds. On 32 delegations from 8 sessions, the flat filter's
subagent found 99 of 105 facts. <!-- @evidence[[results/summary.json#arms.jev4.facts_found]] -->
The whole transcript found 102, <!-- @evidence[[results/summary.json#arms.full.facts_found]] -->
and a brief the orchestrator wrote found 82. <!-- @evidence[[results/summary.json#arms.brief.facts_found]] -->

The second is a cost win with recall at least equal. Over 52 later
delegations from 16 sessions of 19 to 581 chunks, the hierarchical filter at
8x view (`hier8`, the shipped default) cut
Jev input tokens by a median 31% per delegation. <!-- @evidence[[results/summary_pooled.json#hier8_jev_token_saving_pct]] -->
It was never worse than the flat filter on fact recall and better on 3 tasks. <!-- @evidence[[results/summary_pooled.json#pairs.hier8_minus_jev4.better]] -->
That is +2.7 points, <!-- @evidence[[results/summary_pooled.json#pairs.hier8_minus_jev4.mean_pct]] -->
with a 95% CI from +0.0 to +6.4. <!-- @evidence[[results/summary_pooled.json#pairs.hier8_minus_jev4.ci_hi_pct]] -->
The recall gain rests on 3 tasks and on an adjudicator that rescued most regex misses. Scored by regex alone,
the flat filter leads on long sessions. On cost alone the hierarchical filter
became the skill's default.

In both hierarchy rounds every filter arm found at least 96% of facts, so the
eval sits near its ceiling. Harder tasks would separate the arms more.

## Findings

1. On sessions over 150 chunks, `hier8` and `hier4` each found 93 of 93 facts and the flat filter (`jev4`) 90, after adjudication (`results/summary_hier_long.json`). <!-- @evidence[[results/summary_hier_long.json#arms.jev4.facts_found]] -->
2. The adjudicator decides that ordering: it upgraded 61 of 64 regex misses on that run (`results/summary_hier_long.json`). <!-- @evidence[[results/summary_hier_long.json#adjudicator.upgrades]] -->
   Regex-only scoring gives `jev4` 79, `hier4` 76 and `hier8` 77. The adjudicator was validated in the same batches at TPR 100% and TNR 98% (Round 2026-11-02, long sessions).
3. On those sessions `hier8` reads a median 142,841 Jev input tokens per delegation (`results/summary_hier_long.json`). <!-- @evidence[[results/summary_hier_long.json#filter.hier8.jev_tokens_median]] -->
   The flat filter reads 227,449. <!-- @evidence[[results/summary_hier_long.json#filter.jev4.jev_tokens_median]] -->
   It makes 6 Jev calls against 8, and selection takes 1.8 s against 2.5 s. On the 19-to-97-chunk sessions the token figures are 32,620 against 50,954 (`results/summary_hier.json`). <!-- @evidence[[results/summary_hier.json#filter.hier8.jev_tokens_median]] -->
4. A written brief loses facts by pointing at lookups. On 5 of 32 tasks the brief told the subagent to look a fact up instead of stating it. Excluding those 5, the brief, the flat filter and the full transcript land within 2 facts of each other (Round 2026-09-23).
5. Jev's scores vary between identical calls. Given the same segment-level input, two arms kept different segments on 8 of 28 tasks, so a gain of a few tasks between Jev-ranked arms can be noise (Round 2026-11-02, short sessions).
6. The segment level needs a top-k budget, not a score threshold. Jev scores are probabilities from 0 to 1. On a single-topic session every segment scored at least 0.57, so a 0.3 threshold kept them all (Round 2026-11-02, short sessions).
7. The shipped filter never returned on sessions of about 500 chunks or more, so delegations there went out with no context. It is fixed in skill 0.2.3, and the hierarchical default shipped in 0.3.0 (Round 2026-11-02, long sessions).

## Method

- **Rounds.** 2026-09-23: 8 sessions of 19 to 97 chunks, 32 tasks, 105 facts.
  2026-11-02 short: the same 8 sessions relabelled, 28 tasks, 108 facts.
  2026-11-02 long: 8 sessions of 159 to 581 chunks, 24 tasks, 93 facts. Each
  round's tasks were written fresh, so its arms compare only with each other.
- **Corpus.** Archived sessions from the `claude-workspace` `transcripts`
  branch, chunked by `chunking.py`, with secrets redacted before any chunk
  leaves the machine. Transcript-derived data stays out of this public repo:
  `data/` is gitignored, and `EVAL_DATA` points the scripts at a directory elsewhere.
- **Tasks.** Per session, a tool-less Sonnet labeller reads the whole session
  and writes 4 delegation tasks. Each task is phrased by name, without the facts, and
  carries 2 to 4 required facts as regexes, with their transcript evidence and the chunk
  ids that hold them (`label.py`). A task whose regex misses its own evidence or
  leaks into the task text is dropped.
- **Arms.**
  - `none`: the task alone. It found 0 or 1 facts in every round.
  - `jev1`/`jev4`: the flat filter, which shows Jev each chunk clipped to 1x or 4x a base
    length (the **view**; at 1x, 1,500 characters of text, or 300 of tool input
    and 900 of result) and pages the chunks into Jev requests of about 27k tokens
    (**windows**). `jev1` ran only in the first round, where it tied `jev4` at 99 of 105 facts. <!-- @evidence[[results/summary.json#arms.jev1.facts_found]] -->
  - `hier4`/`hier8` (`hierfilter.py`): **level 1** cuts the session into
    segments at each user message and every 8 chunks, shows Jev a short
    **digest** of each (clipped lines of its messages and tool calls), and
    keeps the top 40%, at least 3. **Level 2** scores only those segments'
    chunks at 4x or 8x view.
  - `full`: the whole transcript.
  - `brief`: the executor model writes a brief from the full transcript.
  
  Every filter arm packs its top-scored chunks into the same 20k-token budget,
  with the same `select` function.
- **Executor.** `claude -p` with no tools, settings or MCP, one turn: Sonnet,
  and `sonnet[1m]` in the long round, where the full transcript reaches 391k tokens.
- **Scoring.** Fact regexes first. Every miss then goes to a blind Sonnet
  adjudicator that never sees the arm (`adjudicate.py`). It is validated in
  the same batches on 40 regex hits (TPR) and on 40 facts checked against
  another task's deliverable (TNR).
- **Files.** `chunking.py`, `jevfilter.py`, `hierfilter.py`, `context_hook.py`,
  `preview.py`, `label.py`, `run_eval.py`, `adjudicate.py`, `analyze.py`,
  `analyze_hier.py`, `pool.py`, `recheck.py`; `results/*.json` hold counts only.
  To rerun a round, set `EVAL_DATA` and run `python3 label.py`, then `python3 run_eval.py
  --arms none,jev4,hier4,hier8,full`, `python3 adjudicate.py` and `python3
  analyze_hier.py`; pool the rounds with `python3 pool.py <dir> ...`. The
  filter ships as the `delegating-with-context` skill in `oaustegard/claude-skills`.

## Log

### 2026-09-23: filter against brief and full transcript, 8 sessions, 32 tasks

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

#### Setup

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

#### Results

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

#### What this does not show

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

#### Found along the way

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

#### Prior art

The 101-project roundup and the awesome-jev lists (reviewed in memories
`dd506da5`, `0070d581`) have compaction by Jev (`fast-jev-compaction`,
`yoshi`, `pi-fast-jev-compaction`): score each tool call for whether the main
session still needs it. I did not find task-conditioned selection for a
subagent handoff in those lists; no separate web search was run. The
components exist; this composition was not found there.

#### Files

`chunking.py` (transcript to chunks, redaction), `jevfilter.py` (windows,
bisection on WAF blocks, rank-to-budget), `context_hook.py` (PreToolUse hook),
`preview.py` (what the hook would pass), `run_eval.py`, `adjudicate.py`,
`analyze.py`, `recheck.py`, `results/summary.json`, `ERRORS.md`. Shipped as the
`delegating-with-context` skill in `oaustegard/claude-skills`.

### 2026-11-02: hierarchical filter, the same 8 sessions relabelled

**Question.** DecisionRAG (arXiv 2610.35868) retrieves with Jev by descending a
tree: it scores coarse nodes, keeps the top k and scores only the children of
what it kept. Applied to a transcript, does that shape find as many facts as
the flat filter described above (`jevfilter.py`, shipped in the
`delegating-with-context` skill as the `jev4` setting), and at what Jev cost?
Arms, scoring and the 20k-token context budget are as in Setup above.

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

**Labels are new.** The tasks and fact regexes of the 2026-09-23 eval above
lived in the gitignored `data/` and did not survive. The same 8 session snapshots (same commits, same
chunk counts) were relabelled by 8 tool-less Sonnet labellers using the
same labelling prompt (`label.py`). 34 tasks were written; 6 were dropped for a
regex that missed its own evidence, leaked into the task text, or a fact
count outside 2-4. That left 28 tasks and 108 facts. `jev4` and `full` were
re-run on them, so the comparison is within this run only; the numbers do not
line up with the first Results table.

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

**Verdict.** Not worth replacing the shipped flat filter (`jev4`) on recall. Worth it on
cost only if Jev spend matters, which at $0.002 per delegation it does not. It
becomes worth testing again on sessions past ~150 chunks, where the flat
filter pages into 4 or more windows.

Files: `hierfilter.py`, `label.py`, `analyze_hier.py`, `results/summary_hier.json`;
`run_eval.py` gained the `hier4`/`hier8` arms and, with `adjudicate.py`, an
`EVAL_DATA` override so transcript-derived data can live outside the repo.
With `EVAL_DATA` pointing at a directory holding `chunks/`, `rendered/` and
`tasks/`: `python3 label.py`, then `python3 run_eval.py --arms
none,jev4,hier4,hier8,full`, `python3 adjudicate.py`, `python3 analyze_hier.py`.

### 2026-11-02: hierarchical filter on 8 sessions over 150 chunks

**Asked.** Run the hierarchical filter on sessions over 150 chunks, where the
flat filter pages into many windows.

**Ran.** 30 of the largest archived sessions were chunked, and 12 had more than 150
chunks. 8 were taken, spanning 159 to 581 chunks, on which the flat filter
needs 3 to 11 Jev windows. Labelling them produced 32 tasks. 8 were dropped,
each for a regex that missed its own evidence (one also leaked into its task),
leaving 24 tasks and 93 facts. The executor was `sonnet[1m]` in every arm.

**Broke first.** The chunk survey hung. `jevfilter._global_state` never
terminated on the five sessions of 498 or more chunks: `clip(x, 20)` returns 25
characters, a fixed point, and the loop's exit test required at most 24. The
shipped hook would hit its 120 s timeout there and pass no context. The loop now stops on a pass that changes
nothing, then thins the index. The same fix shipped as skill 0.2.3. The adjudicator's
negative-control sampler also failed on a session left with one task; it now
samples only sessions with two or more.

| arm | facts found | tasks fully correct | 95% CI (tasks) | context passed, est. tokens (median) |
|---|---|---|---|---|
| none | 1/93 (1%) | 0/24 | 0%-0% | 7 |
| jev4 | 90/93 (97%) | 22/24 | 79%-100% | 20,062 |
| hier4 | 93/93 (100%) | 24/24 | 100%-100% | 20,064 |
| hier8 | 93/93 (100%) | 24/24 | 100%-100% | 20,065 |
| full | 93/93 (100%) | 24/24 | 100%-100% | 231,577 |

Regex-only, before adjudication: none 1, jev4 79, hier4 76, hier8 77, full 76.
Adjudicator: TPR 100% on 40 regex hits, TNR 98% on 40 wrong-task
deliverables; 61 of 64 misses upgraded. The regex ranks the flat filter
first, and the adjudicated count ranks it last. The two flat-filter task
misses (`597340de-2`, `-3`) were read by hand, and in both its executor wrote that
the facts were missing. In `-3` the flat filter dropped chunk 65, which `hier8`
kept. In `-2` both dropped chunk 111, and `hier8`'s executor found the same
facts in other kept chunks.

| filter | Jev calls (median) | Jev input tokens (median) | seconds (median) | chunks scored at level 2 (median) | must-have chunks kept |
|---|---|---|---|---|---|
| jev4 | 8 | 227,449 | 2.5 | all | 97/117 (83%) |
| hier4 | 5.5 | 138,294 | 1.8 | 182.5 | 100/117 (85%) |
| hier8 | 6 | 142,841 | 1.8 | 186 | 100/117 (85%) |

`hier8` dropped 17 must-have chunks: 9 with a segment pruned at level 1 and 8 out-ranked
at level 2.

**Pooled with the short-session round** (52 tasks, `pool.py`): `hier8` minus
`jev4` is +2.7% fact recall (CI +0.0% to +6.4%), worse on 0 tasks and better on 3.
`hier4` minus `jev4` is +2.1% (CI −0.5% to +5.3%), worse on 1 and better on 4.
`hier8` minus `full` is +0.5%. The median per-task Jev saving is 31% for `hier8`
and 37% for `hier4`. The two rounds used different executors (Sonnet and
`sonnet[1m]`).

**Tokenizer.** Claude counted 391k tokens for the 890k-character rendering of
the 581-chunk session, which is 2.3 characters per token. The filter's budget
estimate assumes 3.3, so a "20k" context is nearer 29k Claude tokens on this
content. Jev's window limits use Jev's own counts and were not exceeded.

**Reading.** The hierarchy holds or improves recall and costs less. Both of
the flat filter's lost tasks come from one 159-chunk session, so the gain rests
on very few tasks. On these numbers `hier8` became the skill default (0.3.0), with
`CONTEXT_FILTER_MODE=flat` to switch back.
