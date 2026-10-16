# Long-horizon execution on our delegation workers

2026-10-16. Oskar shared Sinha, Arun, Goel, Staab, Geiping, *The Illusion of Diminishing Returns: Measuring Long Horizon Execution in LLMs* (arXiv 2509.09677, ICLR 2026; code `long-horizon-execution/measuring-execution`) as relevant to agent delegation. This directory re-runs two of its experiments on the models we actually delegate to.

The task, from the paper: a dictionary maps five-letter words to integers in [-99, 99] (the paper uses 100 words; this re-run uses 50). The model keeps a running sum. Plan and knowledge are both supplied, so only execution is measured. Ground truth is computed in Python.

## 1. Single turn: sum K keys in one reply

Design: one prompt holds the dictionary and K keys; the reply must be only `<answer>N</answer>`. Items: `single-turn/gen*.py` → `items.json` (answers included). n = 3 per cell unless noted.

Claude workers ran as native Agent-tool subagents (general-purpose, default thinking), told to make one `Read` of the item file and compute without tools. Every transcript was checked: each agent made exactly that `Read` plus its hand-back. Raw: `single-turn/results_agents.json`.

| K | Haiku 5.5 | Sonnet 5.5 | Muse Spark 1.3 contributor, effort low | gemini-3.8-flash, thinking budget 0 |
|---|---|---|---|---|
| 10 | | | | 3/3 |
| 50 | 3/3 | 3/3 | 3/3 | 3/3 |
| 200 | 3/3 | 3/3 | 3/3 | 2/5 |
| 500 | 2/3 | 3/3 | 2/3 | |
| 1000 | 3/3 | 3/3 | | |
| 2000 | 1/3 | 3/3 | | |

- gemini-3.8-flash at budget 8192: K=200 2/2. Budget 0 is not "no thinking" on Gemini 3.x: the model chose ~180 thinking tokens at K=10, ~930 at K=50, and 0 to 2.9k at K=200; the run with 0 thinking tokens was wrong.
- Haiku's two K=2000 misses are budget exhaustion, reported as incomplete with a partial sum rather than a guessed total. One partial was already wrong at key 400 (−1123 vs −1115). Its K=500 miss was off by 3.
- Muse's K=500 miss (−951 vs −1071) cannot be fully separated from a transcription error: the Muse and Gemini arms are reachable only through an MCP tool that takes the prompt inline, so key lists were copied into the call by hand. Muse reasoning tokens: ~1k at K=50, 1.7–4.4k at K=200, 5.8–27.9k at K=500.
- The paper's 2025 single-turn ceilings (Claude Sonnet 4: 432 steps; GPT-5: 2,176) are superseded for these workers.
- Cost note: each Agent-tool subagent carried ~69–71k tokens of fixed context even at K=50, and every launch and completion re-entered the parent session. METHODS.md already says to launch volume as `claude -p` children; section 2 does that.

## 2. Self-conditioning: does a history of the model's own errors cause new ones?

Design (paper §3.2): a session with 100 past turns, one key per turn, where each past assistant answer is wrong with probability e ∈ {0, 0.5, 1}; an error adds ±1–30 and later turns build on it. Then one new key. Scored as **turn accuracy**: correct iff the reply equals the last answer in the history plus the new key's value. n = 20 per cell, the same 20 seeds in every cell.

How the history is planted without an API key: run one real `claude -p --session-id <uuid>` to get a session file under `~/.claude/projects/<cwd>/`, copy its first user and assistant records (`self-conditioning/template.jsonl`), write a new session file with 100 user/assistant pairs and a rebuilt `uuid`/`parentUuid` chain, then `claude -p "Key: <k>" --resume <sid>`. Sanity check: asked to quote its previous three answers, Haiku quoted the planted ones verbatim. Flags: `--system-prompt <own> --tools "" --strict-mcp-config --setting-sources "" --output-format json --effort <low|high>`, stdin from /dev/null. `MAX_THINKING_TOKENS=0` did not stop thinking; thinking tokens are logged per call. Runner `run.py`, analysis `analyze.py`, raw `results_full.jsonl`. 240 calls, $1.24.

| | e = 0 | e = 0.5 | e = 1 |
|---|---|---|---|
| Haiku 5.5, effort high | 20/20 | 19/20 | 14/20 |
| Haiku 5.5, effort low | 20/20 | 19/20 | 18/20 |
| Sonnet 5.5, effort high | 20/20 | 19/20 | 19/20 |
| Sonnet 5.5, effort low | 20/20 | 19/20 | 19/20 |

Each miss was then compared with the **true** cumulative sum (what the history would hold if it had no planted errors):

- 6 of Haiku-high's 7 misses returned the true sum exactly. Haiku recomputed all 101 values from the dictionary (2.6–3.9k thinking tokens) and overrode its corrupted history. That is repair, the opposite of self-conditioning. The 7th was a recompute with an arithmetic error.
- Genuine slips (neither the expected value nor the true sum), pooled over models and effort: e = 0 → 0/80, e = 0.5 → 3/80, e = 1 → 5/80. The direction matches the paper; at n = 80 per level the intervals overlap.
- Haiku thinks at both effort levels (median 87–275 thinking tokens). Sonnet mostly uses 0 thinking tokens at both.

## What this does and does not show

The paper's large self-conditioning drops (open non-thinking 2025 models at turn 100) do not reproduce at this size on Haiku 5.5 or Sonnet 5.5. A fully corrupted 100-turn history costs a few percent of new slips. Given the chance, Haiku at high effort distrusts its own prior answers and rebuilds state from the source.

Not tested: more than one probe turn after the planted history, K > 1 per turn, histories longer than 100 turns, and multi-step tool-using agent runs, which is where delegated work actually accumulates its own outputs. The transcript-planting runner extends to those.
