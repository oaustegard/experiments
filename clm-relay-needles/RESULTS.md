# CLM context management as a relay of stateless calls — Needle Retention

Run 2026-10-01, one seed, Haiku 4.5 subagents, Jev through the Cloudflare AI Gateway.

Jev alone kept all 497 required lines exactly and no noise at 96 chunks (about 11× a 32k
budget) for $0.04. A Haiku 4.5 subagent keeping its own notes file kept 496 for $7.74; it lost
the last one by retyping it with an invented hash. Running both together kept 75%: the
subagent's notes, which were mostly commentary and bulleted re-copies of lines the filter had
already saved, took over the shared cap and pushed 122 of the filter's lines out. That last
result depends on how this harness splits the cap (ERRORS.md #2).

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

## Files

`gen.py` episodes · `relay.py` runner (resumable, per-chunk checkpoints in `work/`, gitignored)
· `grade.py` scoring → `results/summary.json` · `results/*.json` per-episode finals and step
logs · `results/pilot/` 3-chunk pilot · `PREDICTIONS.md` · `ERRORS.md` · `recheck.py`.
