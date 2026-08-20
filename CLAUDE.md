# CLAUDE.md — oaustegard/experiments

One-off research experiments, one self-contained directory each: scripts,
data, and a results writeup (`RESULTS.md`, or `README.md` where the readme
*is* the result). See `README.md` for the index and per-experiment notes.

## Before starting any experiment: grep METHODS.md

`METHODS.md` at the repo root is the ledger of portable methods, environment
gotchas, numerical traps and negative results from every experiment here.
**Read or grep it before writing code**, especially for anything
retrieval-, embedding-, quantization-, LLM-pipeline- or
combinatorial-search-shaped.

```bash
grep -i -A3 'concurrency\|quantiz\|<your topic>' METHODS.md
```

Grep only works if you already know the term. When you don't — which is the
usual shape of this failure — use the semantic index alongside it:

```bash
python3 repo-index/ask.py "about to fan out concurrent LLM calls through a gateway"
```

On this repo's own documented rediscoveries it finds the prior 5/5 from a
plain description of what you are about to do, where grep on the query's own
words finds 1/5. It does not replace the grep — run both. See
[`repo-index/README.md`](repo-index/README.md), including its caveats.

**Both of those only see this repo.** If the thing you are about to build might
already exist in `remex`, `remax`, `claude-skills`, `muninn-utilities` or any
other owned repo, that is a third question and needs the account-wide index:

```bash
python3 /home/user/claude-workspace/scripts/xr.py "<plain description>"
python3 /home/user/claude-workspace/scripts/xr.py -r remex "<description>"
```

Use `-r` whenever you know the repo. Account-wide, this repo's own canonical
rediscovery query — *"about to fan out concurrent LLM calls through a
Cloudflare gateway"* — returns **nothing relevant in the top 20**, because
`claude-skills` contributes 8,088 chunks thick with API-invocation vocabulary.
Scoped with `-r experiments` it puts `phase-a-bridges` at #2. So `xr` answers
*where across the account*, and `ask.py` answers *what inside this repo* —
running the wrong one is how a miss looks like an absence.

**And all three of those only see your own code.** The fourth question is
whether the *world* has already built it, and it is the cheapest of the four:
two to four web searches before you write any code. Search three things, not
one:

1. **Products** — what shipped, and how it actually works. Read past the
   marketing to the mechanism.
2. **The technique in the literature** — the thing you are about to invent
   usually has a name. Cascades, routing and abstention are a mature area; you
   are almost never first to the *method*, only to the application.
3. **Published eval sets and metrics.** This is the highest-value search and the
   easiest to skip. Three passes of `gh-mcp-regex-fit` rested on an 89-row eval
   Claude wrote, and every conclusion carried that caveat. One search for the
   adjacent domain turned up **NL2Bash** (9,305 human-curated pairs), the
   **NLC2CMD** benchmark, and a 600-pair manually verified test set with a
   functional-equivalence metric. An afternoon of searching would have been
   worth more than the eval we hand-authored.

Report the result with the right resolution. "Someone has built a terminal AI
helper" and "someone has built *this composition*" are different findings, and
conflating them kills good ideas. Separate **components exist** from
**composition exists**: Warp ships parameterised workflows *and* an AI prompt,
but as two entry points the user chooses between, not one routed by abstention —
which is a gap, not a duplicate. And when you find nothing, say **"did not find
it in N searches"**, never "it does not exist"; this is a crowded space and four
queries is not a survey.

The absence of a hit is also not a green light on its own. If prior art exists
and you are building anyway, say in the writeup what it does differently and
what you expect to beat.

This is not boilerplate diligence — it is the specific failure this repo has
already had twice:

- `te-bridges` repeated `phase-a-bridges`' Cloudflare-gateway concurrency
  lesson from the **adjacent directory**, and lost 18–20% of its extractions.
- `recall-per-byte` re-derived an ITQ overfitting result that `remax#46` had
  already found and written up.

Both were documented before they were repeated. The cost is real: one entry
in that file blocks a paused **$435** production run.

If `/mnt/muninn` is mounted, the same findings are in the memory corpus
tagged `experiments-repo` — `grep -l experiments-repo /mnt/muninn/memories/*.md`.
Absent that mount, `METHODS.md` is the whole index.

## Shared code lives in `_lib/`

| Module | What |
|---|---|
| `_lib/paths.py` | `experiment(name)` for siblings here, `spoke(name)` for checkouts outside |
| `_lib/pipeline.py` | `retry` (jittered backoff), `chunked`, `save_json`/`load_json` (atomic checkpoints) |
| `_lib/textnorm.py` | `ascii_fold` — the NFKD stroke-letter fix |

Experiment scripts run directly and are not a package, so the repo root has
to go on `sys.path` first:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _lib.paths import experiment, spoke
```

Tests: `python3 _lib/tests/test_lib.py` (no deps, no network, no credentials).
Run them after touching `_lib/` — several experiments import it.

Keep `_lib/` small. An experiment is self-contained by default; code earns a
place here only once a *second* experiment needs it.

## Never hardcode absolute paths

Use `experiment()` / `spoke()`, or `Path(__file__).resolve().parent` for a
script's own directory. Do not route a self-reference through `experiment()`.

This repo split out of `oaustegard/claude-workspace`, and 32 scripts across 13
experiments that hardcoded `/home/user/claude-workspace` were left
non-runnable by the move. If you see that prefix, it predates the split.
Spoke checkouts resolve via `EXPERIMENTS_SPOKES_ROOT`.

## Finishing an experiment

1. Results into `RESULTS.md` — including what broke, and costs/wall-clock.
   Negative results are reported as results, not quietly dropped.
2. A row in the `README.md` index table plus a section under
   "Per-experiment notes".
3. **An entry in `METHODS.md`** for anything that would change what a
   *different* experiment does. Three tests: would it save someone an hour, a
   rerun, or a wrong conclusion? Is it true outside the experiment that
   produced it? Would someone find it without knowing this experiment exists?
4. Regenerable artifacts (logs, checkpoints, large caches, downloaded models)
   into a `.gitignore` **inside that experiment's directory** — the
   convention here is per-experiment, there is no root `.gitignore`.

Doing step 3 at the end of each experiment is far cheaper than reconstructing
it later — the retroactive sweep that produced the current `METHODS.md` cost
roughly 560k tokens.

## Do not add per-experiment technique files

A `TECHNIQUES.md` inside each experiment directory was considered and
rejected: co-location demonstrably did not prevent rediscovery in either case
above, the content would duplicate the existing `RESULTS.md` files, and it
creates 40 maintenance points for no added reach. The failure mode is "I never
looked", not "I looked in the right folder and missed it". One greppable file
at the root is the smallest thing that fixes the actual problem.
