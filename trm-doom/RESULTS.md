# TRM plays DOOM

Samsung SAIL Montreal's Tiny Recursive Model (TRM, Jolicoeur-Martineau, [arXiv 2510.04871](https://arxiv.org/abs/2510.04871))
is a network of about 7M parameters (the version here is 820k) that solves grid puzzles (Sudoku, mazes, ARC) by applying one tiny network over and
over. This experiment asks whether TRM can be the route planner of a bot that plays DOOM: the bot runs inside
[gPu(doom)](https://austegard.com/fun-and-games/doom/), the WebGPU port of DOOM, and TRM runs in WGSL on the same GPU
device. TRM can't render frames or read the screen, so the bot turns the level around the player into a grid, asks TRM
which cells lie on a shortest route to its current goal, and walks that route while it fights. A scripted layer picks
goals (keys, the exit, switches, items, unexplored places) and does the shooting.

## Answer

Yes, for navigation. TRM is the bot's only route planner, and the bot clears Freedoom's first level (E1M1, of nine in
episode 1) from a pistol start: 3 of 3 runs on the second difficulty and 2 of 3 on the third. TRM is exactly right on
94.0% of 32×32 test grids, every one of the 1,024 cells matching breadth-first search (BFS), and on 93.3% from a level it
never trained on. In play, when TRM finds no route to a target, BFS on the same grid agrees 93% of the time (204 of 220):
the target really is closed off until a door opens or a lift comes.

The bot does not finish levels 2–8. Fixes to the grid and the hand-written decision layer around TRM stopped it getting
stuck at lifts, behind thin walls and in nukage (toxic floor). In the last cross-level check (8 runs on levels 2, 3, 4 and 6),
each ended in a fight (5) or at the 240 s game-time cap (3), none stuck. What is left, fighting and multi-step puzzles (which switch
opens which door), belongs to that decision layer, not to the planner. Exact BFS would do the planning more cheaply; the
question was whether a small learned recursive model can carry it.

A bot that survives the fights on levels 4–6 is what it would take to test TRM's planning through the whole episode.

## Findings

1. **TRM cleared E1M1 by itself**, planning every route: 3 of 3 runs at skill 2 (27.5–146.7 s of game time) and 2 of 3
   at skill 3 (94.1 s with 43 health, 101.1 s), across code versions and seeds; runs are deterministic for a given seed and
   code (`results/trm-r7-sk2-*.json`, `results/doom-sweep-*.json`, `results/trm-r7-sk3-1.json`). The fastest clear, 24.3 s,
   was in the on-screen arena (gpu-bridge-comms `doom-logs/2026-10-30T11-43-25-4ddu/001-E1M1-exit.json`).
2. **Planning accuracy on 32×32 grids: 94.0% exact on validation, 93.3% on the held-out level E1M9** (run4, step 60k;
   95.5% on 24-unit fine grids) (`results/training/run4.jsonl`). A prediction is exact only if every one of the 1024 cells is
   right.
3. **In play, TRM's empty whole-level plans are almost always right.** On E1M2 (3 seeds), BFS on the same grid found a
   route for 16 of the 220 targets TRM returned no route to; for the other 204 the level was shut at that moment
   (`results/doom-miss-e1m2-trm.json`). The stalls are a target-choice problem: the bot asked for places a door, lift or
   switch still closed off.
4. **Pooling a fine grid 2×2 into a coarse one joins rooms across thin walls.** A coarse cell was open when any of its four
   fine cells was, so a wall straddling a coarse boundary vanished: routes ran through E1M2's locked blue door and into
   E1M3's switch room through the corridor wall. Cutting coarse neighbours that no open fine pair or wall-free segment joins
   dropped broken local plans from 63% to 33% on E1M2 and from 81% to 17% on E1M3, and the bot pressed the E1M3 switch it
   had walked into a wall beside (`results/doom-cut-trm-sk2-1.json` vs `results/doom-lift-trm-sk2-1.json`).
5. **Setting unreachable targets aside until the level changes ("parking") beat retrying them on a timer.** A target with
   no route on the grid now waits for a switch, trigger, lift ride or boss death. On E1M2, same seeds: kills 11/11/3 became 28/21/8,
   items 8/13/6 became 23/16/8, and the bot never wandered idle (`results/doom-park-e1m2-trm.json` vs
   `results/doom-miss-e1m2-trm.json`).
6. **Nukage deaths were bot behaviour, fixed in two steps.** The bot dropped into E1M3's pit for the switches inside it,
   then stood in the nukage to fight. Skipping targets down in a damaging pit, then keeping the route at full speed on a
   damaging floor, took E1M3 (3 seeds) from deaths at 121/186/164 s to 230 s/alive/alive
   (`results/doom-escape-trm-3s.json`, `results/doom-pit-trm-3s.json`, `results/doom-acid-e1m3-trm.json`).
7. **Lift riding works once the bot faces the call switch.** On E1M2 the bot went from stuck at the first lift (7 kills) to
   600 s alive with 31 kills and a key (`results/doom-lift-trm-sk2-1.json`). On E1M7 an unreachable monster kept it turned
   away from a call switch 16 units off; facing the switch for the press got it past that lift with 7 switches pressed
   (`results/doom-liftface-e1m7-trm.json`).
8. **Seeking health and armour from farther away helped; shooting hitscanners (zombiemen, shotgun guys, whose shots land
   instantly) first hurt, twice.** On E1M4–E1M6 (6 runs,
   same seeds), health below 70 and armour below 50 as long-range targets raised game time survived from 819 s to
   1,017 s (5 deaths to 4). Hitscan-first targeting on top fell back to 827 s; with retreat on fast health loss it had also
   lost before (`results/doom-cmb-*-m456.json`, `results/doom-combat2-m456.json`).
9. **Attention as TRM's token mixer did not learn the task in the time available; a 3×3 convolution did.** With
   attention, validation exact was 1.7% after 4000 steps on an A100; with a 3×3 conv, 4× faster and 94% by step 60k
   (`results/training/run2.jsonl`, `run4.jsonl`).
10. **Three bugs voided results silently until found**: WebGPU's 65,535-workgroups-per-dimension limit (every whole-level
    plan came back empty), a halting head that stopped after one step on large grids, and a runner that read game state
    every second tic so every turn was applied twice.

## Method

Terms: a **supervision step** is one round of TRM's recursion after which its answer is read out (training supervises
every round); the **halting head** is a small output trained to say when the answer is final. **run2…run7** are training
runs: run2 attention on 32×32, run4 3×3 conv on 32×32, run6 and run7 fine-tunes of run4 on 32/48/64 and 32/64/96 grids.
**Skill 2/3** are DOOM's "Hey, not too rough" and "Hurt me plenty"; a **pistol start** begins a level with 100 health, a
pistol and 50 bullets. The **GPU bridge** runs jobs from this session in a browser tab on Oskar's Mac. **SwiftShader** is
the software WebGPU in the container, used for correctness checks only.

- **Grids** (`web/grid.js`, shared by the bot and the data generator). Cells are open floor, wall, door the bot can open,
  nukage, start and goal. A cell is open when its centre lies in a sector with room to stand, more than 14 units from any
  blocking line; blocking lines (one-sided, impassable, a step over 24 or a gap under 56) are drawn as every cell they
  touch, so no 4-connected path crosses a wall. Closed manual doors are door cells when the player has the key.
  The local window is 32×32 cells of 24 units around the player. The whole-level grid pools 2×2 of those into 48-unit
  cells and covers the level's blockmap (96 to 152 cells a side).
- **Labels**: every cell on any shortest 4-connected route from start to goal (all zeros when the goal can't be reached),
  computed by BFS. 300k 32×32 samples from E1M1–E1M8 with random positions, goals, door and lift states, keys and
  decorations; E1M9 is held out. 48/64/96 grids for the size fine-tune come from the same generator.
- **Model** (`train/trm.py`): TRM's recursion (z ← net(x+y+z) n=4 times, y ← net(y+z), T=3 cycles, the first two
  without gradients), deep supervision with a carry and a halting head, post-norm RMSNorm and SwiGLU, d=128, two layers.
  Mixer: 3×3 convolution (820k parameters). Trained on a Colab A100: run4 60k steps on 32×32 (about 2.5 h), then run6/run7
  on mixed sizes from run4's weights with positions zeroed.
- **Inference** (`web/trm.js`): tiled matmul, the 3×3 conv as a gathered matmul, row kernels, all in WGSL on the game's
  `GPUDevice`. The bot asks for 2–4 supervision steps per local plan and ⌈n/16⌉+2 per whole-level plan.
- **Bot** (`web/bot.js`): picks a target, plans the whole level with TRM, takes the farthest route cell inside the local
  window as the local goal, plans that with TRM, and walks the predicted cells (ordered by a BFS restricted to the cells TRM
  marked). A waypoint the local plan can't reach is marked as wall in later whole-level grids for 40 s. Targets with no route
  on the grid are parked until the level changes. Lifts are called (switch or walk-over line), waited for and ridden. On a
  damaging floor the bot keeps its route at full speed; targets down in a damaging pit are skipped. Combat: face the nearest
  visible monster, fire when aimed, back off when close, retreat along the breadcrumbs below 40 health; health below 70
  and armour below 50 are long-range targets.
- **Evaluation** (`bridge/campaign.js`): the GPU-bridge page on Oskar's Mac runs the shipped engine headless with the bot,
  one level from a pistol start per run, 10 minutes of game time per level, about 12× real time. A/B comparisons run both
  arms on the same seeds; a run is deterministic for a seed and code, so differences come from the change.
- **Arena** (`web/arena.html`, live at austegard.com/experiments/trm-doom/web/arena.html): the same bot plays on screen;
  after each attempt the log and thumbnails go to the GPU-bridge repo and a comment wakes the session. A level with no
  progress for 45 s is restarted, two tries per level, each with a fresh seed.

## Log

### Round 1: attention TRM, grids, WGSL

Built the grid rasterizer, the BFS data (300k samples), the PyTorch TRM, its WGSL port (parity 2e-6 on random weights) and
the bot. The first attention run did not learn: token embeddings scaled by √d drowned a 0.02-scale position table. With
both at one scale and a 2D relative bias added, validation exact reached 1.7% after 4000 steps. With a BFS stand-in, the
bot cleared E1M1 headless on SwiftShader after fixes for wall clearance and re-pressing a moving door.

### Round 2: conv TRM

Swapped the mixer to a 3×3 conv (run4): 27% exact by step 5000, 84% by 21k, 94% at 60k. The first TRM-driven runs (run4
at step 5000) played like the BFS stand-in. The windowed coarse level (1536 units) proved unreliable in both directions,
dropping corridors and joining rooms across walls.

### Round 3: whole-level planning

The conv TRM runs on any grid size, so the bot plans the whole level in one grid. Trained only on 32×32, it extrapolated
poorly (30–35% joined on 64×64); run6 and run7 fine-tuned on 32/48/64 and 32/64/96. The first whole-level runs returned
empty plans: the 65,535 dispatch limit, then the early-halting head. After both fixes, plus reading state every tic, TRM
cleared E1M1 at skill 2 (43.8 s) and skill 3 (94.1 s).

| level | skill 2, TRM | skill 3, TRM |
|---|---|---|
| E1M1 | exit, 43.8 s | exit, 94.1 s |
| E1M2 | timeout, 4/59 kills | timeout, 41/93 kills, a key |
| E1M3 | died at 73 s | died at 38 s |
| E1M4 | timeout | time limit at 314 s |
| E1M5 | died at 60 s | — |
| E1M6 | died at 73 s | — |
| E1M7 | time limit at 196 s | — |

### Round 4: overnight sweep

Fixes to the scripted layer: targets that keep failing back off exponentially, walk-over lines that open doors or lower
floors become targets, and fighting moves never step into a wall, a drop-off or nukage. Then every level from a pistol
start with three seeds, run7's step-13,500 weights for both planning levels, five or ten minutes of game time each.

| level | skill 2 runs | exits | how the rest ended | most kills | skill 3 runs | exits |
|---|---|---|---|---|---|---|
| E1M1 | 3 | 3 (27.5–146.7 s) | — | 7/17 | 3 | 2 (94.1, 101.1 s) |
| E1M2 | 8 | 0 | out of time; one run of 19 game minutes | 8/59 | 1 | 0 (41/93 kills, a key) |
| E1M3 | 7 | 0 | died at 70–80 s crossing nukage | 10/48 | 4 | 0 |
| E1M4 | 4 | 0 | out of time, one death | 11/81 | 1 | 0 |
| E1M5 | 9 | 0 | died, 22–230 s | 19/84 | 3 | 0 |
| E1M6 | 3 | 0 | died at about 73 s | 7/124 | — | — |
| E1M7 | 3 | 0 | out of time | 9/129 | — | — |

E1M3's deaths are all in the same place: the route crosses a nukage floor with no radiation suit. E1M2 and E1M4 end in
search: the bot keeps finding new places but not the key and door sequence that opens the exit within the time.

### Round 5: the arena, then fixes from its logs

Oskar asked for the bot to play on screen with its logs shipped back, and for lift riding. The arena (`web/arena.html`)
ran on the Mac and an iPhone through the day; each stall or death in its logs became a fix, each fix a same-seed bridge A/B
before it shipped. Changes in order, with their measurement:

| change | test (seeds, game time) | before | after |
|---|---|---|---|
| ride lifts | E1M2, 1 seed | stuck at the lift, 7 kills | 600 s alive, 31 kills, a key |
| cut pooled thin walls | E1M2/E1M3, share of local plans that failed | 63% / 81% | 33% / 17% |
| skip pit targets, then keep moving on nukage | E1M3, 3 seeds | died 121/186/164 s | died 230 s, alive, alive |
| exits and hunts back off as a group; boss deaths refresh exits | E1M8, 2 seeds | stuck at 123.9 s, 0 areas explored | 27 areas explored, floor trigger crossed |
| set aside targets with no route | E1M2, 3 seeds, kills | 11/11/3 | 28/21/8 |
| face the lift call switch | E1M7, 2 seeds | "lift never came" | past the lift, 7 switches |
| seek health and armour | E1M4–E1M6, 6 runs, total game seconds survived | 819 | 1,017 |

Rejected: hitscan-first targeting with retreat on fast health loss (826 → 782 s survived on E1M4–E1M6) and hitscan-first
alone (827 s against the same 819 s baseline); escaping a damaging floor once in it (all three E1M3 seeds still died in the
pit). On the final build (20261030-151341) no E1M2–E1M7 run stalls at a lift, a thin wall or nukage; the bot dies in a
fight or reaches the time cap. E1M8's two Barons engaged in one arena run; the others stopped at 209 s with 0/2 kills. That
looked like an engine wake-up bug and is not one ([experiments#107](https://github.com/oaustegard/experiments/issues/107)):
the engine's sight and sound wake-up match Doom's, and the stalled runs never entered a sector the Barons can see or that
gunfire reaches from, because the bot spent 40 s oscillating between two sectors toward an unreachable exit and then
stopped exploring. Those stalled runs share one byte-identical trace across three seeds: with nothing awake the seed changes
nothing, so they are one sample, not several.

