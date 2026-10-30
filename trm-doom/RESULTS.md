# TRM plays DOOM

Samsung SAIL Montreal's Tiny Recursive Model (TRM, Jolicoeur-Martineau, [arXiv 2510.04871](https://arxiv.org/abs/2510.04871))
is a network of about 7M parameters (the version here is 820k) that solves grid puzzles (Sudoku, mazes, ARC) by applying one tiny network over and
over. This experiment asks whether TRM can be the route planner of a bot that plays DOOM: the bot runs inside
[gPu(doom)](https://austegard.com/fun-and-games/doom/), the WebGPU port of DOOM, and TRM runs in WGSL on the same GPU
device. TRM can't render frames or read the screen, so the bot turns the level around the player into a grid, asks TRM
which cells lie on a shortest route to its current goal, and walks that route while it fights. A scripted layer picks
goals (keys, the exit, switches, items, unexplored places) and does the shooting.

## Answer

Yes, for navigation. TRM is the only route planner in the bot, and the bot cleared Freedoom E1M1 in all 3 runs at skill 2,
(fastest 27.5 s of game time), and 2 of 3 at skill 3, Hurt Me Plenty (fastest 94.1 s), each from
a pistol start, running on a Mac's GPU through the GPU bridge.
On 32×32 grids labelled by breadth-first search (BFS), the best 32×32 model (run4) is exactly right on 94.0% of
validation grids and 93.3% of grids from E1M9, a level it never trained on. The model the bot uses (run7, the same
820k-parameter network fine-tuned on several grid sizes) scores 90% on 32×32 and plans a whole level in one 96×96 grid
(59% exact on E1M9), a route sketch the bot repairs when a waypoint proves unreachable.

The bot does not finish the episode. On E1M2–E1M7 every run ended in death or ran out of time: combat, nukage, and the
key-and-door sequences are the limits now, not planning. With exact BFS in TRM's place (a debug stand-in) the bot also
failed every one of those levels (`results/oracle-global-*.json`), which places the failures in the scripted layer. BFS is
exact and cheaper; the question was whether a learned recursive model can carry the planning.

Better fighting and key-door handling would show whether TRM's planning holds up on the later levels.

## Findings

1. **TRM cleared E1M1 by itself**, planning every route: 3 of 3 runs at skill 2 (27.5, 43.8 and 146.7 s; the 43.8 s run
   is `results/e1m1-trm-skill2.mp4`) and 2 of 3 at skill 3 (94.1 s with 43 health, 101.1 s), across two code versions and
   three random seeds; runs are deterministic for a given seed and code
   (`results/trm-r7-sk2-*.json`, `results/doom-sweep-*.json`, `results/trm-r7-sk3-1.json`).
2. **Planning accuracy on 32×32 grids: 94.0% exact on validation, 93.3% on the held-out level E1M9** (run4, step 60k;
   95.5% on 24-unit fine grids) (`results/training/run4.jsonl`). A prediction is exact only if every one of the 1024 cells is
   right.
3. **Attention as TRM's token mixer did not learn the task in the time available; a 3×3 convolution did.** With
   attention (learned positions plus a 2D relative bias), validation exact was 1.7% after 4000 steps at 1.8 steps/s on an
   A100. With a 3×3 conv in its place, the run was 4× faster (7.2 steps/s) and went from 4% to 27% between steps 3000 and
   5000, then to 94% (`results/training/run2.jsonl`, `run4.jsonl`). The TRM paper itself swaps attention for an MLP on small
   fixed grids; the recursion and deep supervision are unchanged here.
4. **The conv TRM plans on grids larger than it was trained on once fine-tuned on mixed sizes.** Trained only on 32×32, it
   joined start to goal on 30–35% of 64×64 E1M9 grids. After fine-tuning on 32/48/64 and then 32/64/96 grids with no
   position table, exact match on held-out grids was 75% (64×64) and 59% (96×96), and 32×32 stayed at 90% (run7,
   `results/training/run7.jsonl`). An 8-step plan over a 96×96 level grid takes 255 ms on the Mac. Run7 then fell apart
   after step 22,500 (32×32 down to 76%, 96×96 to 27%) while its training loss kept falling. Only the latest checkpoint
   was kept, so the bot uses the step-13,500 export.
5. **WGSL inference matches PyTorch**: max logit difference 9e-6 on a 32×32 grid with trained weights, and all 9,216 cells
   agree on a 96×96 grid (max difference 0.011 after 45 recursive network calls). One supervision step on 32×32 takes
   12 ms on the Mac and about 1 s on an iPhone.
6. **Three bugs voided results silently until found.** Whole-level grids passed WebGPU's 65,535-workgroups-per-dimension
   limit in one elementwise kernel, so every whole-level plan came back empty without an error. The halting head, trained
   on 32×32, stopped after one step on large grids, which again returned empty plans. The lockstep runner read the game
   state every second tic while the bot steered every tic, so each turn was applied twice and aim never settled.
7. **Combat decides the other levels.** Every E1M2–E1M7 run ended in death or a timeout, with the TRM planner and with the
   exact BFS stand-in alike (`results/oracle-global-sk2.json`, `results/trm-r7-sk2-*.json`).

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
  marked). A waypoint the local plan can't reach is marked as wall in later whole-level grids. Combat: face the nearest
  visible monster, fire when aimed, back off when close, retreat along the breadcrumbs below 50 health.
- **Evaluation** (`bridge/campaign.js`): the GPU-bridge page on Oskar's Mac runs the shipped engine headless with the bot,
  one level from a pistol start per run, 10 minutes of game time per level, about 12× real time.

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
