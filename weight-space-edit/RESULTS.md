# weight-space-edit

Two tests of "weight-space programming" ([Takehi, Teoh, Zhang and Khattab, MIT CSAIL](https://rikiyat.github.io/blog/weight-space-programming/)):
a frontier LLM reading and writing a network's weights directly, with no training. Here the LLM is Muninn (Claude), and the
two targets are our own models: the TRM route planner of the DOOM bot in [`trm-doom/`](../trm-doom/RESULTS.md), and the
LFM2.5-350M chat model behind [`voice-loop/`](../voice-loop/README.md). Write-up for readers:
[muninn.austegard.com/blog/hand-written-weights-for-a-doom-route-planner.html](https://muninn.austegard.com/blog/hand-written-weights-for-a-doom-route-planner.html).

## Answer

Can hand-written weights beat the trained ones? For the TRM planner, yes. Every weight of the shipped 820k-parameter conv TRM
was written by hand (`trm/handtrm.py`) so that each network pass is one step of Lee's 1961 maze router (a mod-3 wavefront
from the goal, then marks running downhill from the start). On held-out E1M9 grids it reaches 100% exact at 32×32 by round 9 <!-- @evidence[[results/summary.json#trm.e1m9_32_hand.first_round_100pct]] -->
,
at 64×64 by round 17 <!-- @evidence[[results/summary.json#trm.e1m9_64_hand.first_round_100pct]] -->
and at 96×96 by round 23 <!-- @evidence[[results/summary.json#trm.e1m9_96_hand.first_round_100pct]] -->,
where the trained run7 weights peak at 88.6% <!-- @evidence[[results/summary.json#trm.e1m9_32_run7.best]] -->,
81% <!-- @evidence[[results/summary.json#trm.e1m9_64_run7.best]] --> and 63% <!-- @evidence[[results/summary.json#trm.e1m9_96_run7.best]] -->.
The hand-written network is slower per round (one cell per pass), so at 2 rounds the trained one leads. In the DOOM arena
the bot still cleared no level: the planner was never its bottleneck.

For the LFM2.5 loop edit, there was nothing to edit: 0 repetition loops in 200 greedy GSM8K answers <!-- @evidence[[results/summary.json#lfm.gsm8k_greedy.loops]] -->
, the post's Gemma null case.

What would change it: a planner task without a known local algorithm, where writing weights by hand stops being possible.

## Findings

1. **Hand-written TRM weights are exact on every held-out grid size tested, given enough rounds**: 100% at 32×32, 64×64 and
   96×96 (500, 200 and 100 grids), where run7 plateaus at 88.6%, 81% and 63% (`trm/results.json`, `results/summary.json`).
2. **At the bot's round budget the hand-written weights still win**: 92.2% vs 87.2% at 32×32 (4 rounds) <!-- @evidence[[results/summary.json#trm.e1m9_32_hand.at_bot_budget]] -->
   , 79.5% vs 75.5% at 64×64 (6 rounds) <!-- @evidence[[results/summary.json#trm.e1m9_64_hand.at_bot_budget]] -->
   , 71% vs 56% at 96×96 (8 rounds) <!-- @evidence[[results/summary.json#trm.e1m9_96_hand.at_bot_budget]] -->
   . At 2 rounds run7 leads (85.8% vs 61.2% at 32×32) <!-- @evidence[[results/summary.json#trm.e1m9_32_run7.at_2]] -->
   .
3. **The halting output is exact**: every grid that halted had an exact answer, 408 of 408 reachable at 32×32 <!-- @evidence[[results/summary.json#trm.e1m9_32_hand.halted]] -->
   and all 100 at 96×96 <!-- @evidence[[results/summary.json#trm.e1m9_96_hand.halted]] --> with the v2 weights, which also
   halt when no route exists.
4. **In the arena the hand-written planner never wrongly said "no route"**: 0 of 249 <!-- @evidence[[results/summary.json#arena.hand.no_route]] -->
   against 14 of 147 <!-- @evidence[[results/summary.json#arena.run7_ghoz.no_route_wrong]] --> and 15 of 350 for run7 in the
   two previous sessions. Whole-level plans averaged 68–600 ms
   per attempt against 269–1,451 ms, because a finished plan stops (older bot build in the run7 sessions, so loose).
5. **The bot's play did not change**: 0 exits in 14 attempts <!-- @evidence[[results/summary.json#arena.hand.exits]] -->,
   9 dead and 5 stuck <!-- @evidence[[results/summary.json#arena.hand.dead]] -->, against 6/8 and 7/7 for run7.
6. **LFM2.5-350M does not loop**: 0 loops in 200 greedy and 100 sampled GSM8K answers and 64 chat replies; 4 answers hit the
   512-token limit, all long derivations (`runs/base_*.jsonl`, `results/summary.json`).
7. **The construction is small**: 69 of 1,024 hidden units and 19 of 128 channels; everything else is zero (`trm/handtrm.py`).

## Method

- **TRM target**: run7 step 13,500 as shipped at `austegard.com/games/doom/bot/trm.{bin,json}` (d=128, two conv blocks,
  n=4, T=3), read back into PyTorch by `handtrm.load_bin`. The hand weights use the same shapes and export format
  (`trm/weights/hand.{bin,json}`; also deployed as `trm-doom/web/weights/hand`).
- **Construction** (`trm/handtrm.py` docstring): channel 0 holds a large constant so RMSNorm only rescales; silu with a large
  constant gate is a linear pass-through; pairs of steep silu units make clamped steps; block 0 re-quantises the summed
  x+y+z bits (threshold 1/6 unit), block 1's conv counts neighbours per label and mark, and its MLP applies one automaton
  step. Halting head: v1 fires on "goal marked"; v2 also on "no cell labelled this pass and start unlabelled".
- **Test grids**: generated with `trm-doom/gen/gen.mjs` (E1M9, 1,000 at 32×32, seed 777) and `gen_big.mjs` (E1M9, 300 at
  64×64 and 96×96, 48-unit cells, seed 777); labels are BFS shortest-route unions. Evaluated on the first 500/200/100.
- **Metric**: exact match of every cell per supervision round (`trm/eval_doom.py`, `trm/eval_run7_96.py`); hand weights run to
  halt (cap 40), run7 to 16 rounds.
- **WGSL check**: `trm/wgsl_check.html` on SwiftShader through the shipped `trm.js`: 4 of 4 grids exact before the page timed out.
- **Arena**: Oskar's Mac, `?weights=weights/hand`, build 20261102-230527, session `2026-11-03T00-46-41-pxcj` in
  `oaustegard/gpu-bridge-comms`; compared with run7 sessions `ghoz` and `mtud` (older builds, random seeds).
- **LFM**: `lfm_eval.py`, LiquidAI/LFM2.5-350M fp32 on CPU, GSM8K test (first 200 greedy, first 100 sampled at T 0.3,
  min_p 0.15, repetition penalty 1.05) and 32 chat prompts, 512 new tokens. A loop is the tail repeating a period of 1–64
  tokens three times.
- **Hero**: `hero/state.py` dumps the hand network's own state on the longest-route 96×96 grid (checked against BFS);
  `hero/hero.html` is the WGSL shader, `animate.cjs` renders the frames.

## Log

### 2026-11-02 — LFM loop baseline, hand-written TRM

Oskar shared the MIT post; I picked the LFM2.5 loop edit as the Exp-8 analogue and measured first. Greedy GSM8K: 0 loops in
200, 33% correct. Oskar suggested the trm-doom TRM mid-turn. I wrote the Lee-router construction; the first build left
off bits at 0.17 because block 0 used the integer-sum gain on 1/3-unit inputs (fixed with gain 12). Then 100% on random
16×16 and 32×32 grids, and on E1M9 32×32 by round 9.

### 2026-11-02 — arena

Oskar asked to try the weights in the arena. v1 halted only when a route was complete, so unreachable targets ran to the
step cap and the arena crawled. v2 adds frontier and start-labelled bits to the halting head; walled-off goals then halt at
round 1. Session `pxcj`: 14 attempts, no exits, 0 wrong "no route" answers.

### 2026-11-03 — 64×64 and 96×96, post

Background jobs hit the 2-hour limit twice and a container restart once; `eval_run7_96.py` checkpoints per batch.
Published the post with a WGSL-rendered hero; the trained 96×96 row was added when it finished. The first published table
said "100% from round 24" for 64×64 and 96×96; `summarize.py` showed the first exact rounds are 17 and 23, and the post was
corrected.
