# CLM context management as a relay of stateless calls

Run 2026-10-01, one seed, Haiku 4.5 subagents, Jev through the Cloudflare AI Gateway. Two tasks:
Needle Retention (keep lines verbatim) and Custody Register (keep a running result), below as
Phase 1 and Phase 2.

**Phase 2 in brief.** When the thing to keep is a running result, the subagent wins. Updating a
40-row register in place, it got every current holder right at 32 and 96 chunks and every
handover count at 32; at 96 it was off by one on 7 of 40 counts (1,529 of 1,536 handovers
counted), for $9.77. Jev-kept lines cannot carry the counts once the handover stream outgrows
the cap: 0 of 40 exact at 96 chunks under either eviction rule. Keeping the most recent lines
preserved every current holder, and a one-pass reader still found only 9 of them; the same
lines read with tools gave 40.

**Phase 3 in brief.** Showing the subagent Jev's flagged lines cut its cost per chunk by 12–25%,
and the register still ended with every holder right. But at 96 chunks one rewrite near chunk
71 reset 14 counts to zero, and nothing caught it. The harness now has an edit gate for the
register (`--guard`).

**Phase 1 in brief.** Jev alone kept all 497 required lines exactly and no noise at 96 chunks (about 11× a 32k
budget) for $0.04. A Haiku 4.5 subagent keeping its own notes file kept 496 for $7.74; it lost
the last one by retyping it with an invented hash. Running both together kept 75%: the
subagent's notes, which were mostly commentary and bulleted re-copies of lines the filter had
already saved, took over the shared cap and pushed 122 of the filter's lines out. That last
result depends on how this harness splits the cap (ERRORS.md #2).

# Phase 1: Needle Retention

## Question

The CLM paper ([arXiv 2609.37725](https://arxiv.org/abs/2609.37725)) has one model edit its
own live context across hundreds of turns. Claude Code cannot edit a live transcript, so the
nearest equivalent is a relay: each chunk of work goes to a fresh call, and only a capped file
carries over. Who should fill that file — the model writing its own notes, an outside scorer
copying lines verbatim (the Jev filter behind `delegating-with-context`), or both?

## Setup

- **Task.** ContextBench Needle Retention, rebuilt from the paper's Appendix D (`gen.py`):
  chunks of ~3.9k o200k tokens, each with 140 filler lines and 2–8 needle lines, every line
  prefixed `[id#hash]` so a retyped line that changes a character stops matching.
  - `labeled`: the paper's format — needles under a NEEDLES header, filler in a block marked
    for deletion.
  - `unlabeled` (added here): needles shuffled into the filler. A needle is any line recording
    a change in who is responsible for an asset, in six phrasings; filler includes near-misses
    naming the same people and assets ("shadowed for training", "discussed whether ... should
    change hands; no decision", "remains with").
- **Sizes.** 8, 32 and 96 chunks (31k, 124k and 373k tokens of input; 1×, 4× and 11× a 32k
  context).
- **Carry cap.** 24,000 tokens: a 32k limit minus one chunk and a 2k response reserve.
- **Conditions** (`relay.py`), one call per chunk:
  - `state`: `claude -p` with Haiku 4.5, Read/Edit/Write tools, no project settings or MCP.
    The chunk arrives in the prompt; `state.md` in the working directory is the only memory.
    Over the cap, `state.md` is cut.
  - `jev`: no model. One Jev call scores every line of the chunk with the delegation filter's
    question ("would an assistant given `task` need to keep this?"); lines at p ≥ 0.5 are
    appended verbatim. Over the cap, the lowest-scored lines go.
  - `both`: Jev runs first, then the subagent is shown which lines Jev kept and told to use
    `state.md` only for lines Jev missed. Graded on the union. The two stores share the cap.
- **Grading** (`grade.py`): a needle counts only as an exact line. `corrupted` = a needle whose
  `[id#` prefix survives with different text.

## Results

| variant | chunks | cond | recall | corrupted | noise kept | carried tok | subagent $ | Jev $ | cut at cap |
|---|---|---|---|---|---|---|---|---|---|
| labeled | 32 | jev | 1.000 | 0 | 0 | 4669 | 0 | 0.0147 | 0 |
| labeled | 32 | state | 1.000 | 0 | 0 | 4669 | 1.218 | 0 | 0 |
| unlabeled | 8 | jev | 1.000 | 0 | 0 | 1242 | 0 | 0.0037 | 0 |
| unlabeled | 8 | state | 1.000 | 0 | 0 | 1259 | 0.528 | 0 | 0 |
| unlabeled | 8 | both | 1.000 | 0 | 0 | 4897 | 0.583 | 0.0037 | 0 |
| unlabeled | 32 | jev | 1.000 | 0 | 0 | 4193 | 0 | 0.0147 | 0 |
| unlabeled | 32 | state | 1.000 | 0 | 0 | 4193 | 1.897 | 0 | 0 |
| unlabeled | 32 | both | 1.000 | 0 | 0 | 11512 | 2.409 | 0.0147 | 0 |
| unlabeled | 96 | jev | 1.000 | 0 | 0 | 14816 | 0 | 0.0444 | 0 |
| unlabeled | 96 | state | 0.998 | 1 | 0 | 15501 | 7.741 | 0 | 0 |
| unlabeled | 96 | both | 0.755 | 0 | 0 | 24000 | 8.137 | 0.0444 | 21 |

Total spend $22.6 ($22.51 subagent, $0.09 Jev). Full per-episode rows in `results/summary.json`.

## Findings

1. **Jev made no errors on this task.** 835 needles across four episodes, every one kept
   byte-exact, zero filler kept, including the near-miss filler built to catch keyword
   matching. One gateway call per chunk, paced at 45/min; the 96-chunk episode took 3 minutes,
   against about 85 for the subagent.
2. **The subagent was nearly as accurate at about 175× the cost.** Its one miss at 96 chunks
   is a retyping error: `[c053L025#09c855ef] Castillo now maintains chiller-29C, taking over from Xu.`
   came back as `[c053L025#d1b9e2e4] ...` — right text, invented hash. A model that copies
   text through its own output can corrupt it; an extractive store cannot.
3. **Most of the subagent's cost is reading, not writing.** Output tokens (thinking included)
   were 140k on the unlabeled 32-chunk run against 39k on the labeled one with the same input
   volume: finding unmarked handovers among 148 lines costs it ~3.6× the tokens of copying a
   marked list. Jev's cost is the same for both variants.
4. **Combining the two did worse than either, for a reason specific to the setup.** In `both`,
   the subagent never added a missed line (Jev missed none). It rewrote `state.md` each step
   into a running commentary — "No additional handover lines found that the filter missed" —
   followed by bulleted copies of the lines Jev had already saved, prefixed `- ` and so not
   exact matches. The two stores reached the cap together at chunk 74. From then on the notes
   sat at ~12.9k tokens, cut back by the harness most steps, which left Jev's store ~11.1k; each
   new line Jev kept evicted an older one. Jev's store ended with 375 of the 497 needles, and
   the notes held none of the other 122 as exact lines.
   The subagent ignored a direct instruction not to re-copy the filter's lines, at every step.

## Against PREDICTIONS.md

| # | prediction | outcome |
|---|---|---|
| 1 | `jev` ≥ 0.95 at every size | held: 1.000 everywhere |
| 2 | `state` falls with length, below `jev` at 96 | held only technically: 1.000 → 0.998, one corrupted line |
| 3 | `both` ≈ `jev` recall at `state`'s cost | wrong: 0.755, through cap displacement (finding 4) |
| 4 | ~$0.07 per chunk for subagents; Jev under a cent per episode | subagents $0.06–0.08 per chunk held; Jev was $0.04 at 96 chunks, not under a cent |
| 5 | labeled ≈ 1.0 for both | held |

## Caveats

- One seed per cell. The `state` result rests on one corrupted line; seed 1 would show whether
  0.998 is typical or lucky.
- Haiku 4.5 only. A stronger subagent may copy more faithfully and follow the "don't re-copy"
  instruction; the paper's ContextBench runs used GPT-5.4.
- The task is synthetic and favours a per-line classifier: every needle is a self-contained
  line whose relevance can be judged without context. Tasks where relevance depends on what
  came earlier, or where the useful thing to keep is a synthesis rather than a line (the
  paper's Sudoku Sketchpad, or a running tally), are where model-written state should earn
  its cost. This run does not test them.
- `both` lets the notes file squeeze the filter's store. Reserving the store's share of the
  cap, or making the notes append-only, would likely have kept `both` at `jev`'s recall; that
  version was not run.
- Relay ≠ CLM. The paper's model edits its live context in place; here each step is a fresh
  process, so this tests the relay design proposed in the session, not the paper's harness.

# Phase 2: Custody Register

## Setup

`gen_tally.py`: 40 assets with known starting holders. Each ~3.6k-token chunk carries 16
handover lines (the same six phrasings as Phase 1) in chronological order, interleaved with 120
filler lines; a "remains with" filler line always names the true holder. The task is the final
holder and handover count of every asset. 32 chunks = 512 handovers, whose lines (~15k tokens)
fit the 24k cap; 96 chunks = 1,536 handovers (~46k tokens), which do not.

`relay_tally.py`, one call per chunk, same 24k cap:
- `state`: the Phase 1 subagent, with `state.md` seeded with the initial register as a table.
- `jev`: Jev keeps handover lines verbatim; over the cap the lowest-scored go.
- `jev_fifo`: the same, but over the cap the oldest go.

Every condition ends with one Haiku 4.5 answer call, no tools, given the task and the carried
text. `answer_tools.py` re-answers the two Jev conditions from the same carried text with Read
and Bash (`+tools` rows). `grade_tally.py` also replays the carried lines with a parser that is
exact on the full stream (40/40 holders and counts at both sizes), giving the best any reader
could do with what was stored (`replay` columns).

## Results

| chunks | handovers | cond | holder | count exact | count MAE | replay holder | replay count | carried tok | relay $ | answer $ | Jev $ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 32 | 512 | state | 1.000 | 1.000 | 0.0 | — | — | 521 | 3.44 | 0.013 | 0 |
| 32 | 512 | jev | 0.925 | 0.575 | 0.6 | 1.000 | 0.900 | 15576 | 0 | 0.165 | 0.0146 |
| 32 | 512 | jev+tools | 1.000 | 0.900 | 0.1 | 1.000 | 0.900 | 15576 | 0 | 0.181 | 0.0146 |
| 32 | 512 | jev_fifo | 0.950 | 0.575 | 0.78 | 1.000 | 0.900 | 15590 | 0 | 0.413 | 0.0146 |
| 32 | 512 | jev_fifo+tools | 1.000 | 0.900 | 0.1 | 1.000 | 0.900 | 15590 | 0 | 0.237 | 0.0146 |
| 96 | 1536 | state | 1.000 | 0.825 | 0.17 | — | — | 511 | 9.769 | 0.015 | 0 |
| 96 | 1536 | jev | 0.200 | 0.000 | 20.7 | 0.650 | 0.000 | 23993 | 0 | 0.189 | 0.0438 |
| 96 | 1536 | jev+tools | 0.600 | 0.000 | 19.18 | 0.650 | 0.000 | 23993 | 0 | 0.183 | 0.0438 |
| 96 | 1536 | jev_fifo | 0.225 | 0.000 | 19.98 | 1.000 | 0.000 | 23983 | 0 | 0.19 | 0.0438 |
| 96 | 1536 | jev_fifo+tools | 1.000 | 0.000 | 18.75 | 1.000 | 0.000 | 23983 | 0 | 0.185 | 0.0438 |

Holder and count are fractions of 40 assets. Phase 2 spend $15.21, plus $0.49 for the pilot.
Per-episode rows in `results_tally/summary.json`.

## Findings

5. **The register stayed at ~515 tokens for 96 chunks.** The subagent rewrote counts and
   holders in place every step; the table never grew. All seven of its count errors at 96
   chunks are undercounts by exactly one, on assets whose holder it still got right: a handover
   it missed mid-stream, after which later handovers corrected the holder but nothing could
   restore the lost increment. That is 7 misses in 1,536 events, about 0.5%, and in a running
   count every miss is permanent.
6. **Kept lines lose counts by construction once the stream outgrows the cap.** At 96 chunks Jev
   kept 786 (`jev_fifo`) and 809 (`jev`) of the 1,536 handover lines. Whatever eviction rule picks the survivors, the
   dropped handovers are gone and so is every count (replay count 0.000 for both). At 32 chunks,
   where everything fits, Jev's own misses set the ceiling: it kept 508 of 512 handovers,
   capping counts at 36/40.
7. **The eviction rule decides whether holders survive.** Keeping the newest lines kept each
   asset's latest handover (replay holder 1.000); evicting by score dropped lines regardless of
   age and left only 26 of 40 holders recoverable (0.650). Jev's scores rate relevance, not
   recency, so score-based eviction is the wrong rule for anything that is overwritten.
8. **A tool-less reader fails on what was stored.** At 96 chunks `jev_fifo` carried every
   current holder, and a one-pass Haiku answer found 9 of 40. Given Read and Bash, the same model wrote
   a parser and hit the replay ceiling exactly (40/40). At 32 chunks the one-pass reader got
   23 of 40 counts where 36 were recoverable. Verbatim storage moves the synthesis to the end,
   and the end is one long read.
9. **Cost.** The subagent cost $0.10 per chunk at both sizes and the register's size did not
   change it: about 10.7k output tokens per step, mostly thinking through 136 lines. Jev cost
   $0.04 for 96 chunks; its answer calls cost $0.17–0.41.

## Against PREDICTIONS.md (6–10)

| # | prediction | outcome |
|---|---|---|
| 6 | Jev at 32: holders ≥ 0.9, counts < 0.7 | held (holders 0.925 / 0.950, counts 0.575) |
| 7 | `state` at 32: holders ≥ 0.9, beats Jev on counts, below 1.0 | held except the last clause: it scored 1.000 |
| 8 | `jev` at 96: ≤ 0.5 on both | held (0.200 / 0.000); the reader accounts for part of it, since replay reaches 0.650 holders |
| 9 | `jev_fifo` at 96: holders ≥ 0.8, counts → 0 | storage held (replay 1.000; 1.000 with tools); the one-pass answer scored 0.225, so as stated, wrong |
| 10 | `state` best at 96, counts below its 32 level, ~$0.12/chunk | held: best on both, counts 1.000 → 0.825, $0.10/chunk |

## Caveats

- One seed. The subagent's seven misses are a rate estimate from one stream.
- Haiku 4.5 does all the reading. A stronger one-pass reader would narrow finding 8; it would
  not touch finding 6.
- The task is regular enough for a 7-pattern parser to replay exactly, which is why `+tools`
  works. Messier phrasing would make the tool-using reader's job harder too, and is the case
  where a model-maintained register should pull further ahead.
- No combined condition here. A store that keeps the register in model notes and recent lines
  verbatim is the obvious next arm.

# Phase 3: register plus Jev, combined

## Setup

`both` in `relay_tally.py`. Each chunk, Jev scores every line first. The subagent sees the lines
Jev flagged as likely handovers ("usually right, check the chunk too") and updates the register,
which has 8k of the cap reserved. The flagged lines also go to a verbatim store of the most
recent handovers, 16k reserved, so neither store can squeeze the other (the Phase 1 failure).
The answer call reads the register, labelled authoritative, then the recent lines.

## Results

| chunks | cond | holder | count exact | count MAE | relay $ per chunk | subagent output tok per step |
|---|---|---|---|---|---|---|
| 32 | state | 1.000 | 1.000 | 0.0 | 0.107 | 10,800 |
| 32 | both | 1.000 | 0.975 | 0.03 | 0.080 | 7,273 |
| 96 | state | 1.000 | 0.825 | 0.17 | 0.102 | 10,689 |
| 96 | both | 1.000 | 0.550 | 10.05 | 0.090 | 8,807 |

Phase 3 spend $11.49 (relay $11.25, answers $0.18, Jev $0.06), plus $0.25 for the pilot.

## Findings

10. **One rewrite erased 14 counts.** At 96 chunks the register held every holder, but 18 of
    40 counts were wrong. Fourteen of them match the number of handovers since chunk 71 almost
    exactly (13 of 14 within one): in one step near there the subagent rewrote the table and
    reset those rows' counts, then counted correctly from that point on. Nothing in the
    harness checked the register between steps, so a single bad write was permanent. Without
    those 14, `both` had 4 wrong counts against `state`'s 7: three short by one and one high by
    nine.
11. **Jev's flags cut the subagent's work.** Output tokens per step fell 33% at 32 chunks and 18%
    at 96, and cost per chunk 25% and 12%. The flags did not make its counting more reliable at
    32 chunks (one undercount where `state` had none).
12. **The recent-lines store did not confuse the answer.** In both runs the final answer copied
    the register exactly.

## Applied

- `relay_tally.py --guard` adds an edit gate run by the harness, after the paper's own: a
  register edit that drops a row, lowers any count, or raises the counts by more than the chunk
  holds is undone and the subagent retries once with the reason. `test_gate.py` checks it on a
  14-row reset. A guarded rerun of `both` at 96 chunks is in progress (`run_guard.log`).
- METHODS.md gains the portable lessons: reserve cap shares, evict by recency for overwritten
  state, give the final reader tools, gate model-maintained state.

## Against PREDICTIONS.md (11–14)

| # | prediction | outcome |
|---|---|---|
| 11 | holders 40/40 at both sizes | held |
| 12 | ≤ 3 counts wrong at 96 | wrong: 18, of which 14 from one reset (4 otherwise) |
| 13 | cheaper per chunk than `state` | held: 25% at 32, 12% at 96 |
| 14 | recent lines don't hurt the answer | held: answer = register in both runs |

## Caveats

One seed, so the reset is one event: it says a single bad write can happen and is never
repaired, not how often. Whether the flags made it more or less likely is unknown; the run
cannot separate that from chance.

## Files

`gen.py` episodes · `relay.py` runner (resumable, per-chunk checkpoints in `work/`, gitignored)
· `grade.py` scoring → `results/summary.json` · `results/*.json` per-episode finals and step
logs · `results/pilot/` 3-chunk pilot · Phase 2: `gen_tally.py`, `relay_tally.py`, `answer_tools.py`,
`grade_tally.py` → `results_tally/` (pilot in `results_tally/pilot/`), `run_tally.log` · Phase 3: `run_both.log`,
`test_gate.py`, `run_guard.log` · `PREDICTIONS.md` ·
`ERRORS.md` · `recheck.py`.
