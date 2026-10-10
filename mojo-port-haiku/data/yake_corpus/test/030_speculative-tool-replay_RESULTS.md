# Speculative tool execution — replay over archived Claude Code sessions

Perfect speculation over this account's archived sessions would save 1.5% of active time. A selector that picks the next call from a menu built out of earlier session context can reach 0.56% at most. Nearly all of that is WebFetch. The idea is dead as a general agent speedup. The only niche left is a URL prefetcher.

## Question

Idea 1 from memory `aae7ec10`: while the model is generating, Jev picks the likely next read-only tool call from a candidate list. The call runs in the background, and a PreToolUse hook serves the cached result when the model actually makes it. Before any Jev call, this experiment measures how much time that could save.

## Data

- The `transcripts` branch of `oaustegard/claude-workspace`, latest commit per session.
  - 86 sessions, 28.75 active hours. A gap between records counts only up to 10 minutes; longer gaps are treated as idle.
  - Subagent records (`isSidechain`) are excluded.
- A **prediction point** is the moment after a tool result or user message when the next assistant message starts. There are 3,330.
  - Its label is the set of read-only calls that message makes. 578 points have at least one, 833 calls in all.
- **Read-only** means Read, Grep, Glob, WebFetch, WebSearch, the Muninn read ops and the GitHub read tools, plus Bash commands made only of allowlisted read verbs with no write hint (`parse.py`).
- **Savable time** for a call is `min(tool duration, generation window)`.
  - Tool duration: tool_use timestamp to tool_result timestamp.
  - Generation window: prediction point to the tool_use timestamp.
  - A speculative run started at the point cannot finish before its own duration, and cannot save more time than the model spent generating.

## Results

Numbers are from `results/offline.json`.

| | minutes | share of active time |
|---|---|---|
| read-only tool time | 37.5 | 2.2% |
| oracle speculation (every call predicted) | 26.5 | 1.5% |
| reachable by a selector (call in the candidate pool) | 9.6 | 0.56% |

| tool | calls | median duration | in candidate pool | reachable min / oracle min |
|---|---|---|---|---|
| WebFetch | 168 | 5.2 s | 119 | 8.7 / 10.3 |
| WebSearch | 120 | 6.9 s | 0 | 0 / 9.7 |
| Bash (read-only) | 202 | 0.22 s | 0 | 0 / 2.9 |
| Muninn recall | 32 | 0.95 s | 0 | 0 / 1.1 |
| Muninn config get | 41 | 1.2 s | 7 | 0.1 / 1.0 |
| Muninn memory_get | 29 | 0.74 s | 29 | 0.4 / 0.4 |
| Read | 157 | 0.02 s | 67 | 0.2 / 0.4 |

The candidate pool (`candidates.py`) holds everything earlier in the session that code can turn into a call:

- file paths, URLs, 8-hex memory ids and `config_get('…')` keys
- exact repeats of earlier read-only calls
- a four-command git/ls menu

The median pool has 47 entries, and 242 of the 247 reachable calls sit in the 60 most recent.

WebSearch queries and read-only Bash commands are written fresh by the model and never repeat verbatim, so a selector cannot put them on the menu. Jev picks from options and cannot compose a call. Read and Grep finish in about 20 ms, so predicting them saves nothing even when a prediction is right. The median generation window is 3.0 s (p90 10.7 s). The model's own generation, and long commands that change files, account for most of the wall time, and speculating on those is unsafe.

## Not run

`replay.py` asks Jev one Choice per point: which of up to 60 candidates, or none, the next message calls. It is written but was not run. The session's auto-mode classifier denied sending transcript excerpts to TypeSafe. With 0.56% as the ceiling, the Jev hit rate only matters for the WebFetch prefetcher: 119 of 168 fetched URLs appear earlier in the session.

## Files

- `parse.py` — prediction points, read-only classification, call keys, timing
- `candidates.py` — candidate pool from earlier context
- `offline.py` → `results/offline.json`
- `replay.py` — the unrun Jev pass. Its state is scrubbed with `subagent-context-filter/chunking.scrub`, and it sends through `jev-tag-encoder/jev.py`.

Transcripts live outside the repo (`/home/user/spec-data/`). Only aggregates are committed. The hub clone is shallow: run `git fetch --unshallow origin transcripts` first, and take paths with `--diff-filter=AM`. Plain `--name-only` also lists paths a commit deleted, and `git show` on those comes back empty.
