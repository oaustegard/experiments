# TRM plays DOOM

Samsung SAIL Montreal's Tiny Recursive Model (TRM, Jolicoeur-Martineau, [arXiv 2510.04871](https://arxiv.org/abs/2510.04871))
is a network of about 7M parameters (the version here is 820k) that solves grid puzzles (Sudoku, mazes, ARC) by applying
one tiny network over and over. This experiment asks whether TRM can be the route planner of a bot that plays DOOM. The
bot runs inside [gPu(doom)](https://austegard.com/fun-and-games/doom/), the WebGPU port of DOOM, and TRM runs in WGSL on
the same GPU device. TRM can't render frames or read the screen, so the bot turns the level into a grid, asks TRM which
cells lie on a shortest route to its current goal, and walks that route. Hand-written code around TRM (the "decision
layer") picks goals (keys, the exit, switches, items, unexplored places), rides lifts and does the shooting.

## Answer

Yes, for navigation. TRM is the bot's only route planner, and the bot clears Freedoom's first level (E1M1, of nine in
episode 1) from a pistol start: 3 of 3 runs on the second difficulty and 2 of 3 on the third. TRM is exactly right on
94.0% of 32×32 test grids (every cell matching breadth-first search, BFS) and on 93.3% from a level it never trained on.
In play, when TRM finds no route to a target, BFS on the same grid agrees 93% of the time (204 of 220): the target really
is closed off until a door opens or a lift comes.

The bot does not finish levels 2–8. The last full arena session (build 20261030-151341, 14 attempts) ended in a fight 6
times, all on levels 4–6, and stalled 8 times, on levels 2, 3, 7 and 8, where the bot made no progress for 45 s. Each
stall diagnosed during the day sat in the decision layer: lifts it didn't ride, a grid that joined rooms through walls,
nukage pits, unreachable targets retried on a timer. Fixing them raised kills and survival in same-seed comparisons, but
the bot still doesn't find the order of switches, doors and lifts that opens each level. Exact BFS would do the planning
more cheaply; the question was whether a small learned model can carry it, and it can.

## Findings

1. **TRM cleared E1M1 by itself**, planning every route: 3 of 3 runs at skill 2 (27.5–146.7 s of game time) and 2 of 3
   at skill 3 (94.1 s with 43 health, 101.1 s), across code versions and seeds (`results/trm-r7-sk2-*.json`,
   `results/doom-sweep-*.json`, `results/trm-r7-sk3-1.json`). The fastest clear, 24.3 s, was in the on-screen arena
   (gpu-bridge-comms `doom-logs/2026-10-30T11-43-25-4ddu/001-E1M1-exit.json`).
2. **Planning accuracy on 32×32 grids: 94.0% exact on validation, 93.3% on the held-out level E1M9** (run4, step 60k;
   95.5% on 24-unit fine grids) (`results/training/run4.jsonl`). A prediction is exact only if all 1,024 cells are right.
3. **In play, TRM's "no route" answers are almost always right.** On E1M2 (3 seeds), BFS on the same grid found a route
   for 16 of the 220 targets TRM returned none for; for the other 204 the level was shut at that moment
   (`results/doom-miss-e1m2-trm.json`). The stalls are a target-choice problem, not a planning one.
4. **Levels 2–8 are not cleared, and the remaining failures are in the decision layer.** Arena session `ghoz` (build
   20261030-151341): E1M4–E1M6 six deaths in fights (49–174 s), E1M2, E1M3, E1M7 and E1M8 eight stalls with health
   83–101 (gpu-bridge-comms `doom-logs/2026-10-30T15-42-57-ghoz/`).
5. **Pooling a fine grid 2×2 into a coarse one joins rooms across thin walls.** A coarse cell was open when any of its
   four fine cells was, so a wall straddling a coarse boundary vanished: routes ran through E1M2's locked blue door and into
   E1M3's switch room through a corridor wall. Cutting coarse neighbours that no open fine pair or wall-free segment joins
   dropped failed local plans from 63% to 33% on E1M2 and from 81% to 17% on E1M3 (`results/doom-cut-trm-sk2-1.json` vs
   `results/doom-lift-trm-sk2-1.json`).
6. **Setting unreachable targets aside until the level changes beat retrying them on a timer.** On E1M2, same seeds: kills
   11/11/3 became 28/21/8, items 8/13/6 became 23/16/8 (`results/doom-park-e1m2-trm.json` vs
   `results/doom-miss-e1m2-trm.json`). On E1M8, adding "explore after an unreachable target" took the bot from 0 areas
   explored to 25–58 and across the map to the Barons on 3 of 3 seeds (`results/doom-m8fix-*-e1m8.json`); it still kills
   neither.
7. **Nukage deaths were bot behaviour.** The bot dropped into E1M3's pit for the switches inside it, then stood in the
   nukage to fight. Skipping targets down in a damaging pit, then keeping the route at full speed on a damaging floor, took
   E1M3 (3 seeds) from deaths at 121/186/164 s to death at 230 s and two runs alive at the 300 s cap
   (`results/doom-escape-trm-3s.json`, `results/doom-pit-trm-3s.json`, `results/doom-acid-e1m3-trm.json`).
8. **Seeking health and armour from farther away helped; shooting hitscanners first hurt, twice.** On E1M4–E1M6 (6 runs,
   same seeds), health below 70 and armour below 50 as long-range targets raised game time survived from 819 s to 1,017 s
   (5 deaths to 4). Targeting zombiemen and shotgun guys first, whose shots land instantly, fell to 827 s on top of that,
   and to 782 s when combined with retreating on fast health loss (`results/doom-cmb-*-m456.json`,
   `results/doom-combat2-m456.json`).
9. **Attention as TRM's token mixer did not learn the task in the time available; a 3×3 convolution did.** With
   attention, validation exact was 1.7% after 4,000 steps on an A100; with a 3×3 conv, training ran 4× faster and reached
   94% by step 60k (`results/training/run2.jsonl`, `run4.jsonl`).
10. **Three bugs voided results silently until found**: WebGPU's 65,535-workgroups-per-dimension limit (every whole-level
    plan came back empty), a halting head that stopped after one step on large grids, and a runner that read game state
    every second tic so every turn was applied twice.

## Method

Terms: a **supervision step** is one round of TRM's recursion after which its answer is read out (training supervises
every round); the **halting head** is a small output trained to say when the answer is final. **run2…run7** are training
runs: run2 attention on 32×32, run4 3×3 conv on 32×32, run6 and run7 fine-tunes of run4 on 32/48/64 and 32/64/96 grids.
**Skill 2/3** are DOOM's "Hey, not too rough" and "Hurt me plenty"; a **pistol start** begins a level with 100 health, a
pistol and 50 bullets. **E1M1…E1M9** are the nine levels of Freedoom's episode 1. The **GPU bridge** runs jobs from this
session in a browser tab on Oskar's Mac. **SwiftShader** is the software WebGPU in the container, used for correctness
checks only.

- **Grids** (`web/grid.js`, shared by the bot and the data generator). Cells are open floor, wall, door the bot can open,
  nukage, start and goal. A cell is open when its centre lies in a sector with room to stand, more than 14 units from any
  blocking line (24 from a drop of over 64 units or into nukage); blocking lines (one-sided, impassable, a step over 24 or
  a gap under 56) are drawn as every cell they touch. Closed manual doors are door cells when the player has the key; a
  lift's edge is a door cell when the lift can stand level with the floor beside it. The local window is 32×32 cells of 24
  units around the player. The whole-level grid pools 2×2 of those into 48-unit cells over the level's blockmap (96 to
  152 cells a side), then cuts coarse neighbours that only a wall separates (finding 5). `test/reach.mjs` checks both ways:
  places reached in play stay reachable, and two known leaks stay cut.
- **Labels**: every cell on any shortest 4-connected route from start to goal (all zeros when the goal can't be reached),
  computed by BFS. 300k 32×32 samples from E1M1–E1M8 with random positions, goals, door and lift states, keys and
  decorations; E1M9 is held out. 48/64/96 grids for the size fine-tune come from the same generator.
- **Model** (`train/trm.py`): TRM's recursion (z ← net(x+y+z) n=4 times, y ← net(y+z), T=3 cycles, the first two
  without gradients), deep supervision with a carry and a halting head, post-norm RMSNorm and SwiGLU, d=128, two layers.
  Mixer: 3×3 convolution (820k parameters). Trained on a Colab A100: run4 60k steps on 32×32 (about 2.5 h), then run6/run7
  on mixed sizes from run4's weights with positions zeroed. The bot uses run7's step-13,500 export.
- **Inference** (`web/trm.js`): tiled matmul, the 3×3 conv as a gathered matmul, row kernels, all in WGSL on the game's
  `GPUDevice`. The bot asks for 2–4 supervision steps per local plan and ⌈n/16⌉+2 per whole-level plan.
- **Bot** (`web/bot.js`): picks a target, plans the whole level with TRM, takes the farthest route cell inside the local
  window as the local goal, plans that with TRM, and walks the predicted cells (ordered by a BFS restricted to the cells TRM
  marked). A waypoint the local plan can't reach is walled in later whole-level grids for 40 s. A target with no route on
  the grid (checked with BFS) is set aside until a switch, trigger, lift ride or monster death changes the level; after an
  unreachable target, unexplored places outrank everything but keys, health and boss hunts for 30 s. Lifts are called (switch or
  walk-over line), waited for and ridden. On a damaging floor the bot keeps its route at full speed; targets down in a
  damaging pit are skipped. On ExM8 it hunts the remaining monsters before the exit. Combat: face the nearest visible
  monster, fire when aimed, back off when close, retreat along its own path below 40 health; health below 70 and armour
  below 50 are long-range targets. Tie-breaks between unexplored places draw on the game seed.
- **Evaluation** (`bridge/campaign.js`): the GPU-bridge page on Oskar's Mac runs the shipped engine headless with the bot,
  one level from a pistol start per run, capped at 200–600 s of game time, about 5–12× real time. Bridge runs have no
  stall detection, so "timeout" means alive at the cap, stalled or not. A/B comparisons run both arms on the same seeds; a
  run is deterministic for a seed and code, so differences come from the change.
- **Arena** (`web/arena.html`, live at austegard.com/experiments/trm-doom/web/arena.html): the same bot plays on screen.
  After each attempt the log and thumbnails go to the GPU-bridge repo and a comment wakes the session. An attempt with no
  progress (kills, items, new places, targets reached) for 45 s ends as "stuck"; two tries per level, each with a fresh
  seed, 8 minutes per level at most.

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
ran on the Mac and, for a while, an iPhone; each stall or death in its logs became a fix, and each fix a same-seed bridge
A/B before it shipped. Changes in order, with their measurement:

| change | test | before | after |
|---|---|---|---|
| ride lifts | E1M2, 1 seed | arena: stuck at the lift, 7 kills | 600 s alive, 31 kills, a key |
| cut pooled thin walls | E1M2 / E1M3, share of local plans that failed | 63% / 81% | 33% / 17% |
| skip pit targets, then keep moving on nukage | E1M3, 3 seeds | died at 121/186/164 s | died at 230 s, alive, alive |
| exits and hunts back off as a group; a boss death refreshes the exits | E1M8, 2 seeds | stuck at 123.9 s, 0 areas explored | 27 areas explored, a floor trigger crossed |
| set aside targets with no route | E1M2, 3 seeds, kills | 11 / 11 / 3 | 28 / 21 / 8 |
| face the lift call switch | E1M7, 2 seeds | "lift never came" | past the lift, 7 switches |
| seek health and armour | E1M4–E1M6, 6 runs, total game seconds survived | 819 | 1,017 |
| explore after an unreachable target; drop a target after back-and-forth; seeded tie-breaks | E1M8, 3 seeds | stuck, 0 areas, never west of x = 1776 | 25–58 areas, reaches the Barons; 0/2 kills |

Rejected: hitscan-first targeting with retreat on fast health loss (826 → 782 s survived on E1M4–E1M6), hitscan-first
alone (827 s against an 819 s baseline), and escaping a damaging floor once in it (all three E1M3 seeds still died in the
pit).

E1M8's Barons engaged in only one arena run, and the others stopped at 209 s with 0/2 kills. That looked like an engine
wake-up bug and is not one ([experiments#107](https://github.com/oaustegard/experiments/issues/107)): gPu(doom)'s sight and
sound wake-up match Doom's, and the stalled runs never entered a sector the Barons can see or that gunfire reaches. The
three stalled runs share one byte-identical trace across three seeds, because with nothing awake the seed changed nothing;
the bot's tie-breaks now draw on the seed.

The last complete arena session on build 20261030-151341 (`ghoz`, before the #107 fixes) ended 6 times in fights on
E1M4–E1M6 and 8 times stalled on E1M2, E1M3, E1M7 and E1M8. A session on 20261030-163344 is running.
