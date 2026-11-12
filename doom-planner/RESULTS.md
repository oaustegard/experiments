# RESULTS: a Doom player that plans by playing out futures

Oskar Austegard (who owns this repository), 2026-11-12, after asking whether a Haiku agent running at low
reasoning effort could play Doom: *"What if it was treated as a decision model with a war game-like
multiple scenarios, pick the best one, approach?"*, then *"Build it in
experiments."* Live page with videos of the runs:
[Doom by Rollouts](https://austegard.com/experiments/doom-planner/web/).

## Answer

Can Haiku play Doom? From the screen alone, not in this run: choosing every
move for five minutes of E1M1, it completed 0 levels <!-- @evidence[[results/summary.json#e1m1-haiku-screen.levels_completed]] -->
and had 1 death. <!-- @evidence[[results/summary.json#e1m1-haiku-screen.deaths]] -->
Shown each option's outcomes in three copies of the game played 1.5 s ahead,
it cleared E1M1 at 266.9 s of game time, <!-- @evidence[[results/summary.json#e1m1-haiku-plan.first_exit_seconds]] -->
after 2 deaths, <!-- @evidence[[results/summary.json#e1m1-haiku-plan.deaths]] -->
picking the option the planner ranked first in 16.8% of decisions. <!-- @evidence[[results/summary.json#e1m1-haiku-plan.haiku.matched_planner_top_share]] -->

The planner is code Claude Opus wrote: every 0.23 s it copies the running
game 90 times (30 options, 3 reseeded scenarios each), plays each copy 1.5 s,
scores it from game state, and plays the best option. With no knowledge of
the exit it cleared E1M1 at 162.5 s <!-- @evidence[[results/summary.json#e1m1-explore.first_exit_seconds]] -->
with 0 deaths. <!-- @evidence[[results/summary.json#e1m1-explore.deaths]] -->
Haiku breaking only the planner's close ties took 216.8 s. <!-- @evidence[[results/summary.json#e1m1-judged.first_exit_seconds]] -->
In 15 minutes the planner never found E1M2's exit.

Each setting ran once, with no random or human baseline.

## Findings

1. **Haiku from the screen alone did not clear E1M1 in five minutes.** It
   saw 318 wall lines, <!-- @evidence[[results/summary.json#e1m1-haiku-screen.e1m1_lines_seen_max]] -->
   about as many as the planner saw before its exit, mostly by walking
   forward and turning; it ran out of ammo, died once and ended with no kills
   in its last life. Its notes described the scene correctly. 1,311 calls <!-- @evidence[[results/summary.json#e1m1-haiku-screen.haiku.calls]] -->
   at 4.8 s each, none unreadable. (`e1m1-haiku-screen`)
2. **Haiku choosing from played-out outcomes cleared E1M1, slower and less
   safely than the planner.** Exit at 266.9 s <!-- @evidence[[results/summary.json#e1m1-haiku-plan.first_exit_seconds]] -->
   with 6 of 6 kills in the life that left the level <!-- @evidence[[results/summary.json#e1m1-haiku-plan.e1m1_kills]] -->
   and 2 deaths before it. It agreed with the planner's first choice in only
   16.8% of decisions, <!-- @evidence[[results/summary.json#e1m1-haiku-plan.haiku.matched_planner_top_share]] -->
   so it weighed the outcomes its own way. (`e1m1-haiku-plan`)
3. **Rollouts alone clear E1M1.** Exit at 162.5 s, <!-- @evidence[[results/summary.json#e1m1-explore.first_exit_seconds]] -->
   all 6 monsters killed, no deaths. The planner knows the map's geometry
   (it measures walking distance through it) but not which line is the exit:
   once no unseen wall is reachable it heads for wall switches it has not
   pressed, and the exit switch is one of them. (`e1m1-explore`)
4. **Knowing where the exit is makes it 2.8 times faster and less thorough.**
   58.2 s to the exit, <!-- @evidence[[results/summary.json#e1m1-assisted.first_exit_seconds]] -->
   with 3 of 6 kills. <!-- @evidence[[results/summary.json#e1m1-assisted.e1m1_kills]] -->
   Each 5-minute run carries on into E1M2 after the exit, where the assisted
   run had 1 death. <!-- @evidence[[results/summary.json#e1m1-assisted.deaths]] -->
   (`e1m1-assisted`)
5. **The Haiku judge mostly agreed with the numbers and did not speed the
   level up.** 197 calls <!-- @evidence[[results/summary.json#e1m1-judged.judge.calls]] -->
   at 5.56 s each; <!-- @evidence[[results/summary.json#e1m1-judged.judge.mean_seconds]] -->
   it kept the planner's top-scoring option 66.5% of the time <!-- @evidence[[results/summary.json#e1m1-judged.judge.picked_planner_top_share]] -->
   and the exit came at 216.8 s instead of 162.5 s, in a single run. Haiku sees the
   planner's scores, with the top option listed first, which may anchor it;
   that is untested. (`e1m1-judged`)
6. **E1M2 is where the planner stops.** The 15-minute run repeats the 5-minute run
   exactly through E1M1 (the game is deterministic), then spends 12 minutes on
   E1M2 without reaching the exit. It abandoned 124 targets <!-- @evidence[[results/summary.json#e1-long.giveups]] -->
   (unseen walls or switches it was heading for and could not reach); the
   identical 5-minute run abandoned 31 in all, <!-- @evidence[[results/summary.json#e1m1-explore.giveups]] -->
   so at least 93 were on E1M2. (`e1-long`)
7. **Forked copies shared one open game-data file and corrupted each other.**
   Doom reads its graphics and maps from a WAD file. Until the WAD was loaded
   into memory before the first fork, about a third of the copies crashed
   while loading graphics and scored as 0, and two identical runs diverged at
   the first decision. After the fix two runs produce decision logs
   with matching MD5 checksums. (Log, "Determinism")
8. **A planner that re-plans every few steps needs a "keep going" option and
   must finish plans that reach the goal.** Without the first, two
   turn-then-run plans (turn 90°, then run forward) restarted each other 122 times each and the player
   stood still; without the second, the player stood at the exit switch for
   6 s while every copy predicted pressing it. (Log, "Getting unstuck")
9. **A 1.5 s lookahead needs a long-range pull.** With only local rewards
   every option scored about 0 once a room was seen. Walking distance through the
   map to the nearest unseen wall fixed that; straight-line distance pointed
   through walls. (Log, "Getting unstuck")
10. **Rollouts are cheap.** On 4 CPUs the planner ran 27,141 tics of rollout play <!-- @evidence[[results/summary.json#e1m1-explore.rollout_tics_per_wall_second]] -->
   per second of wall time, so 5 minutes of game time with 117,174 rollouts <!-- @evidence[[results/summary.json#e1m1-explore.rollouts]] -->
   took 224.5 s, <!-- @evidence[[results/summary.json#e1m1-explore.wall_seconds]] -->
   which overstates the wall time because the judged run shared the CPUs. (The first decision
   on each map has no "keep going" option, hence 6 fewer than 1,302 × 90.) The judged run took
   1432.8 s, <!-- @evidence[[results/summary.json#e1m1-judged.wall_seconds]] -->
   most of it waiting on Haiku.

## Method

- **Terms.** A *tic* is Doom's fixed game step, 1/35 s; 8 tics are 0.23 s
  and 52 tics 1.5 s. A *rollout* is one forked copy of the game playing one
  option under one scenario. A *line* is one wall or edge of the 2D map Doom's
  levels are drawn from, in map units (a player is 32 units wide); a
  *two-sided* line separates two floor regions (*sectors*) and can be walked
  through if the gap is tall enough, a one-sided line is solid wall. An *exit
  line* carries the special that ends the level. *Manual doors* open when the
  player presses use facing them. To *pulse use* is to press and release use
  every 6 tics. Skill 3 is "Hurt Me Plenty", the middle difficulty.
- **Game.** PrBoom 2.5 (from retro-go via `oaustegard/m5`) built for Linux
  in single-tic mode: one game tic and one rendered frame per loop, no wall
  clock. Shareware DOOM1.WAD, E1M1, skill 3. The binary runs in a sandbox
  with no network (`scripts/jail.sh` from `oaustegard/claude-workspace`) and
  talks to `play.py` over stdin/stdout. Every frame of the real game (not the
  rollouts) goes to ffmpeg.
- **Roles.** Claude Opus, in a Claude Code session, wrote `src/host_main.c`
  (1,164 lines) and `play.py` (250 lines), chose the options, score weights
  and anti-stall rules, and tuned them over about eight rounds by reading the
  planner's videos and decision logs (Log, "Getting unstuck"). The
  planner-only runs (explore, assisted, 15-minute) consult no model during
  play. In the judged run, Haiku answers 197 of 1,301 decisions, each a choice
  among the 2 to 4 options the planner scored highest. In the two Haiku-plays
  runs, Haiku answers every decision.
- **Options per decision.** 18 combinations of forward/back/none × turn
  left/right/none × fire, strafe left/right with and without fire, walk
  forward pressing use, use in place, five turn-then-run plans (45° left or
  right, 90° left or right, or 180°, then run and pulse use), and "continue the current plan
  from where it is" with a bonus of 10.
- **Scenarios.** Each option runs under 3 perturbations of Doom's random
  number generator state (its seeds and its table positions). The real game's seed is never used, so the
  planner never sees the actual future.
- **Score of a rollout.** 12 × damage dealt + 150 × kills − 10 × damage taken
  − 3,000 × death + 4 × new wall lines drawn + 25 × new 128-unit cells entered
  + 40 × items − 2 × ammo spent + (10,000 + 50 × tics left) × level exit
  + 0.3 × walking distance gained toward the nearest unseen wall (or untried
  switch) − 20 if the player moved less than 64 units. Exit-assisted runs
  add 1.0 × walking distance gained toward an exit line. An option's value is
  the mean over scenarios minus half the gap to the worst.
- **Walking distance** is Dijkstra over line midpoints through two-sided
  lines whose opening is at least 56 units tall with a step of at most 24 units,
  plus manual doors.
  It does not model lifts, keys or switches that open remote doors.
- **Targets and give-ups.** The walking-distance target is the nearest wall
  line not yet drawn. Lines Doom never draws (open portals between sectors of
  equal height) and lines within 128 units of anywhere the player has stood do
  not count. A target that stays nearest for 25 decisions with no new line
  drawn is dropped, with every line within 192 units of it. When no target is
  reachable, the target becomes the nearest wall switch not yet used.
- **Commit.** The chosen option plays for 8 tics, or to the end of its
  horizon when it reaches the exit in every scenario.
- **Judge (judged run only).** When the two best options score within 10% of
  each other, at most every 4 decisions, `play.py` builds one image (the view
  now and the end frame of the 2 to 4 best options, with their scores) and runs
  `claude -p --model haiku --effort low` with only the Read tool. It answers
  `PICK <letter>` and a note of at most 25 words, which the next call sees.
- **Haiku plays (two runs).** `--haiku-screen`: no rollouts; every
  decision, Haiku gets the screen (640×400), health, armor and ammo, the 30
  numbered options with a key to their names, and the note it left last
  turn, and answers `PICK <n>` and a new note. `--haiku-plan`: the rollouts
  run as usual and each option also lists its outcomes averaged over the 3
  scenarios (damage dealt and taken, kills, deaths and exits out of 3, new
  walls, new areas, items, ammo, walking distance gained toward unexplored
  space, distance moved), in a fixed order and without the planner's score.
  The system prompt says it is playing Doom's shareware E1M1 on the middle
  difficulty, that the goal is the exit, and what "use" does. An answer that
  cannot be parsed falls back to "continue" (screen) or the planner's choice
  (plan); neither run needed it. Code presses use at intermission and after a
  death in every mode.
- **Runs.** One per setting, 5 minutes of game time each, plus one 15-minute
  unassisted run. Runs without the judge are deterministic: the same settings
  give the same decisions. Haiku's answers can vary, so the judged run and the
  two runs where Haiku chooses every move would not repeat exactly.
- **Outputs.** `results/<run>.json` and `results/summary.json` (`report.py`);
  videos on R2 under `video/doom-planner/`, linked from each results file;
  `runs/` (gitignored) holds frames, decision logs and judge logs.

## Log

### Round 1: build and first runs (2026-11-12)

Built `doom_host` and `play.py`. A 10-second smoke run played sensibly. That
first run ran outside the network-less sandbox; every run after it, and every
run reported here, ran inside it.

### Determinism

The smoke video had 212 frames for 350 tics. `D_Display` calls
`D_BuildNewTiccmds`, which runs `I_StartTic` once per tic the clock has
advanced, and a clock that moved on every call ran it about 1.65 times per
tic. That shortened every rollout. Fixed: the clock moves only in the screen-wipe
loop, and `I_StartTic` acts once per `gametic`.

Two identical 60-second runs still diverged. Decision 0's per-scenario scores
were exactly 0.0 in a random third of the slots; the children had died with
SIGSEGV in `R_CachePatchNum`. All forked children share the parent's open
WAD file and its read position, and children loading lumps at the same time
read each other's bytes. Loading the WAD into memory in `I_InitGraphics`
fixed it: two runs then wrote decision logs with matching MD5 checksums. Every run before
this fix was discarded, including an exit-assisted run that appeared to
finish E1M1 in 24 s. Unwritten slots now start as NaN and abort the run.

A second, unrelated failure: `play.py` read the game's output as strict
UTF-8, PrBoom printed a non-UTF-8 byte at the level change, and the Python
side's death killed the game through SIGPIPE.

### Getting unstuck

With correct rollouts, the planner explored for 11 s and then backed into a
corner with every option scoring about 0: nothing new was visible within
1.5 s. In order:

| problem seen | change | effect |
|---|---|---|
| flat scores once a room is seen | walking distance to the nearest unseen wall, through the sector graph | moved on, 317 lines seen |
| standing by a door it never faced | turn-then-run plans pulse use; −20 for not moving | opened doors |
| two plans alternating 122 times each | "continue" option with a +10 bonus | stopped oscillating |
| pinned beside portal lines Doom never draws | ignore lines with no visible texture; lines near the player count as seen | moved on |
| pinned beside unreachable lines (a room seen through a window) | drop a target that stays nearest for 25 decisions, with its neighbours within 192 units | 447 lines seen, 6 of 6 kills |
| nothing unseen left, exit never pressed | target untried switches once no unseen wall is reachable | reached the exit switch |
| standing at the exit switch, every rollout predicting the exit | earlier exits score higher; a plan that exits in every scenario is played to the end | E1M1 cleared at 162.5 s |

### Haiku chooses every move

Oskar asked what Haiku's role really was, then to run both tests. Both ran in
parallel through `claude -p --model haiku --effort low`. The screen-only run
took 6,373 s of wall time. The outcomes run was stopped by the session's
two-hour limit for background commands at tic 10,403 of 10,500; ffmpeg had
already closed the video, and `summary.json` was rebuilt from the decision
log and console output (`damage_dealt` and item counts are missing for it).

| run | E1M1 exit | deaths | E1M1 lines seen | Haiku calls | s per call | agreed with planner's top |
|---|---|---|---|---|---|---|
| e1m1-haiku-screen | none | 1 | 318 | 1,311 | 4.8 | – |
| e1m1-haiku-plan | 266.9 s | 2 | 314 | 1,292 | 5.4 | 16.8% |

In the screen run Haiku's choices were mostly `fwd` (413), `fwd+turnL` (259)
and `fwd+turnR` (210). Later notes read like "Standing against a blue-lit
wall with a corpse just below view, no ammo. Heading forward to find the
exit."

### Final runs

| run | E1M1 exit | E1M1 kills | deaths | decisions | wall time |
|---|---|---|---|---|---|
| e1m1-explore (scores only) | 162.5 s | 6 / 6 | 0 | 1,302 | 224.5 s |
| e1m1-assisted (exit distance) | 58.2 s | 3 / 6 | 1 | 1,302 | 224.6 s |
| e1m1-judged (Haiku on close calls) | 216.8 s | 6 / 6 | 0 | 1,301 | 1,432.8 s |
| e1-long (scores only, 15 min) | 162.5 s | 6 / 6 | 0 | 3,927 | 622.8 s |

Decisions come every 8 tics of play and stop during the intermission screen
between levels; where the judged run's exit fell on that grid left it one
decision short of the others.

The explore and assisted runs shared the CPUs with the judged run, so their
wall times are upper bounds. In the judged run, Haiku overrode the planner on
66 of 197 calls. Its notes describe the view and its pick ("D backs up and
turns right, revealing a new open area with water and sky").
