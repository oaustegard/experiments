# RESULTS: does trying fights out in copies of the game make the gPu(doom) bot better?

The [trm-doom](../trm-doom/RESULTS.md) bot plays [gPu(doom)](https://austegard.com/games/doom/), a version of
DOOM (with the free Freedoom levels) whose game logic runs on the GPU. DOOM advances in steps of 1/35 s
called tics. The bot's routes through each level, the cells it walks, come from TRM, a small neural network
(Samsung's Tiny Recursive Model). Hand-written code chooses where to go and how to fight. That bot clears the
first level, but on levels 4 to 6 (E1M4 to E1M6) it usually dies in a fight.

This experiment replaces only the hand-written fighting. In a fight, every 12 tics (a third of a second), the
new controller copies the game and tries 6 moves for 2 s each: the bot's own move, stand still, and toward,
away, left and right relative to the enemy. Each move is played in 2 copies with different random numbers,
so 12 short futures per decision. It makes the move whose futures scored best on damage dealt and taken,
kills and deaths. Below, **baseline** is the trm-doom bot unchanged and **rollout fights** is the same bot
with this controller. A sibling experiment, [doom-planner](../doom-planner/RESULTS.md), chooses every move
this way in a different Doom engine; this one keeps the bot's targets and routes and changes only fights.

## Answer

Does it get better? At surviving and killing, yes, clearly. A pair runs the same level, random seed and code
twice, once per arm, for up to 200 s of game time. Across 18 pairs, the baseline died 16 times <!-- @evidence[[results/summary.json#sets.all.base.deaths]] -->
and rollout fights died 1 time. <!-- @evidence[[results/summary.json#sets.all.roll.deaths]] -->
Kills went from 288 <!-- @evidence[[results/summary.json#sets.all.base.kills]] -->
to 663. <!-- @evidence[[results/summary.json#sets.all.roll.kills]] -->
In 15 pairs only the baseline died, in 1 both died, and in none did only rollout fights die; a two-sided
sign test on those 15 against 0 gives p = 6.1e-05.
<!-- @evidence[[results/summary.json#sets.all.survivalSignP]] -->

It does not finish these levels: 1 of 18 rollout runs reached the exit. <!-- @evidence[[results/summary.json#sets.all.roll.exits]] -->
The other 17 were alive at the 200 s cap, so whether more time would get them out is untested. 6 pairs ran
on a Mac's GPU with TRM routes. 12 ran on SwiftShader, the software GPU in a cloud container, with exact
breadth-first-search (BFS) routes, which TRM was trained to imitate, because TRM there ran far slower than
the game.

What would change this: runs long enough to reach the exit, or the 12 SwiftShader pairs rerun on a real GPU
with TRM.

## Findings

1. **Rollout fights almost stop the deaths.** 16 of 18 baseline runs died within 200 s; 1 of 18 with
   rollout fights. On the Mac with TRM routes: 4 of 6 against 0 of 6. On SwiftShader with BFS routes: 12 of
   12 against 1 of 12. <!-- @evidence[[results/summary.json#sets.swiftshader-bfs.base.deaths]] -->
   (`results/summary.json`)
2. **They kill more than twice as much:** 663 against 288. More kills in 15 of 18 pairs <!-- @evidence[[results/summary.json#sets.all.pairsRollMoreKills]] -->
   and fewer in 3, <!-- @evidence[[results/summary.json#sets.all.pairsRollFewerKills]] -->
   sign test p = 0.0075. <!-- @evidence[[results/summary.json#sets.all.killsSignP]] -->
   On E1M5 rollout fights made 50 to 63 kills in every run, against 6 to 26 for the baseline.
3. **They do not finish levels.** One exit in 18 runs (E1M5, seed 0, 185 s, Mac); the other 17 were alive
   at the 200 s cap. The one rollout death (E1M6, seed 111, 93 s) is also the only pair where the baseline
   lived longer (123 s).
4. **About half the time the futures favour the bot's own move.** Of 7,688 decisions, <!-- @evidence[[results/summary.json#decisions.total]] -->
   48.2% kept the bot's own move. <!-- @evidence[[results/summary.json#decisions.keptBotShare]] -->
   Of the rest, "toward" (1,317) and "stand" (1,209) were chosen most, then "left" (552) and "right"
   (508), and "away" (393) least: the controller holds or closes the distance more often than it backs off.
5. **Two seconds is about the shortest useful look-ahead.** With 1 s futures, most of 154 decisions in a 40 s test
   gave every move the same score of 0 (a tie), since a shot rarely lands within a second. With 2 s, 35 of 37 decisions in a 20 s
   test told the moves apart. (Log, smoke tests)
6. **Copying gPu(doom)'s game is a GPU buffer copy.** The whole game lives in four GPU buffers (sectors,
   things, line state and a player/world block). Copying them aside and back restores the game exactly: a
   restored copy played 175 tics matched the original exactly, except one flag ("this wall has been seen") that only drawing
   sets. Running a tic without drawing it takes 6.8 ms on SwiftShader against 38 ms drawn, so the rollouts
   skip drawing. (`test/fork-check.html`)
7. **On the Mac a decision took 434 ms** <!-- @evidence[[results/summary.json#sets.mac-trm.meanDecisionMs]] -->
   for 840 tics of futures (6 moves x 2 copies x 70 tics). Decisions come every 12 tics, 343 ms of game
   time, so in fights the game ran a little slower than real time: playable
   on a screen, with brief stalls. On SwiftShader a decision took 6 to 9 s.

## Method

- **Bot**: `trm-doom/web/bot.js` unchanged (build of 2026-11-02), difficulty 2 of 5 ("Hey, not too rough"), each level started with 100 health and only a pistol. On
  the Mac, routes come from TRM (run7 weights shipped at `austegard.com/games/doom/bot/`); on SwiftShader from
  `OraclePlanner`, exact BFS on the same grids.
- **Rollout fights** (`web/combat.js`): `RolloutBot` wraps the Bot. When the Bot's enemy check finds a
  monster it can see (and it isn't opening a door or freeing itself from a wall), every 12 tics it saves the four state
  buffers (`web/fork.js`). For each of 6 moves, 2 copies are played with the game's random-number index
  set to two other values, never the real game's, so the futures are ones that could happen rather than
  the one that will. Each copy runs for 70 tics (2 s). In a copy the move aims at the enemy, fires by
  the Bot's own rule (when aimed within the monster's width) and moves in the move's direction, re-reading positions every 3 tics. A move whose first 40
  map units (about a player's width) would leave the floor the Bot's route grid marks walkable is skipped. Score: damage dealt + 50 per kill − 4 per point of health
  lost (each point of armour lost costs a third as much) − 2000 for dying; the move's score is its mean pulled halfway toward its worst
  copy, plus 5 for the move already being played.
- **Arms**: base (the Bot alone) and roll (`RolloutBot`), same level, seed and code. 200 s cap
  (7,000 tics). Levels E1M4, E1M5, E1M6 of Freedoom's episode 1.
- **Runs**: on a Mac (Apple GPU, Chrome) through the GPU bridge, a web page that runs jobs pushed to a
  repository, for seeds 0 and 37 with TRM: 2 seeds x 3 levels = 6 pairs. On SwiftShader in headless Chromium
  for seeds 74, 111, 148, 185 with BFS: 12 pairs, spread over four cloud containers.
  Pairs are only formed within one GPU. `summarize.py` builds `results/summary.json`.
- **Not controlled**: the two GPUs used different route planners and seeds (the Mac ran first, on seeds 0
  and 37; the SwiftShader runs took the four seeds the Mac could not finish), so the Mac and SwiftShader sets
  are reported separately as well as together. The Mac also ran the baseline on the four SwiftShader seeds
  with TRM (3 levels each, 12 runs, 8 deaths) before the Mac overheated (Log), so those runs are not in
  any pair.

## Log

### 2026-11-14: the fork and the controller

Question: Oskar asked whether the gPu(doom) bot could get better with the doom-planner's approach. The
proposal was trm-doom's targets and routes, with forked rollouts only in fights.

`web/fork.js`: copy `secBuf`, `mobjBuf`, `lsBuf`, `worldBuf` to saved buffers and back; run a tic through
`encodeFrame` into a command encoder that is never submitted, so the tic kernels (which `encodeFrame`
submits itself) run and the drawing passes do not. `test/fork-check.html` on SwiftShader, E1M1: 120 drawn
tics, save, 175 more; restore and replay the same inputs undrawn, then drawn. Sectors, things and World
(minus view interpolation and status-bar words) hashed identical in all three. Line state differed by bit
31, the automap's "seen" bit that only drawing sets; the Bot masks it off. Reseeding the random index
changed World as expected. Undrawn tic 6.77 ms, drawn 38.28 ms, small readback 1.59 ms.

Smoke tests on SwiftShader, E1M4 seed 0, BFS routes:

| test | settings | decisions | outcome |
|---|---|---|---|
| base, 40 s | | | alive, 9 kills |
| roll, 40 s | 10 moves, 35-tic horizon, commit 8 | 154 at 5.5 s | alive, 10 kills; most decisions scored every move 0 |
| roll, 20 s | 6 moves, 70-tic horizon, commit 12 | 37 at 6.6 s | 35 of 37 told moves apart |

The second setting is the one used for every run below.

### 2026-11-14: the Mac pairs

GPU bridge, Mac tab, TRM routes, seeds 0 and 37. The iPhone tab took the first two jobs and lost its GPU
on both ("map async was not successful", "No WebGPU adapter"); both were rerun on the Mac.

| level | seed | base | roll |
|---|---|---|---|
| E1M4 | 0 | alive 200 s, 7 kills | alive 200 s, 24 kills |
| E1M4 | 37 | died 65.9 s, 9 kills | alive 200 s, 8 kills |
| E1M5 | 0 | died 76.5 s, 18 kills | exit 185 s, 63 kills |
| E1M5 | 37 | died 44.1 s, 13 kills | alive 200 s, 63 kills |
| E1M6 | 0 | alive 200 s, 24 kills | alive 200 s, 11 kills |
| E1M6 | 37 | died 111.9 s, 13 kills | alive 200 s, 34 kills |

A rollout run took 3 to 5.5 minutes of wall clock for 200 s of game. The baseline on four more seeds
(`rollfight-base-3`) died 8 times in 12. The rollout jobs for those seeds were withdrawn after the Mac
overheated: three 7 to 9 minute rollout jobs back to back drove it to a forced shutdown. One had already
started.

### 2026-11-14: the SwiftShader pairs

TRM on SwiftShader did not finish 77 s of a baseline run in 50 minutes, so both arms used BFS routes.
Four containers ran three level-seed pairs each (`test/batch2.sh`). Two sibling sessions lost the planner
change in a worker restart and stalled for hours before rerunning. A baseline run took about a minute,
a rollout run 40 to 96 minutes.

| level | seed | base | roll |
|---|---|---|---|
| E1M4 | 74 | died 54.7 s, 11 kills | alive 200 s, 27 kills |
| E1M4 | 111 | died 44.3 s, 10 kills | alive 200 s, 25 kills |
| E1M4 | 148 | died 130.1 s, 16 kills | alive 200 s, 23 kills |
| E1M4 | 185 | died 60.8 s, 16 kills | alive 200 s, 30 kills |
| E1M5 | 74 | died 40.3 s, 13 kills | alive 200 s, 59 kills |
| E1M5 | 111 | died 62.3 s, 6 kills | alive 200 s, 63 kills |
| E1M5 | 148 | died 49.2 s, 17 kills | alive 200 s, 50 kills |
| E1M5 | 185 | died 94.8 s, 26 kills | alive 200 s, 58 kills |
| E1M6 | 74 | died 68.8 s, 10 kills | alive 200 s, 36 kills |
| E1M6 | 111 | died 123.4 s, 34 kills | died 92.9 s, 25 kills |
| E1M6 | 148 | died 143.9 s, 31 kills | alive 200 s, 32 kills |
| E1M6 | 185 | died 106.7 s, 14 kills | alive 200 s, 32 kills |

Reading at the time: the same result as the Mac's on a different route planner and twice the seeds.
BFS routes made the baseline die more often (12 of 12) than TRM routes did on the Mac (12 of 18 over
all six seeds), on different seeds and hardware, so that difference is not a finding.

Prior art: two searches found forward-model search for game combat in StarCraft (combat models standing in
for a forward model, arXiv 1605.05305) and in general video-game AI (GVGAI). They found no FPS bot that
forks the real engine state to choose combat moves while scripted code navigates. gPu(doom) makes that
cheap, since the whole game is four GPU buffers.
