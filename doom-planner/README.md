# doom-planner

**In progress: built and smoke-tested once; the full runs have not happened yet.**

A Doom player that decides by war-gaming. Every 8 game tics (0.23 s) it forks
the running game once per candidate action and scenario. Each copy reseeds
Doom's random number generator, plays its candidate for 35 tics (1 s), and is
scored from engine state: damage dealt and taken, kills, deaths, items, map
lines seen for the first time, new 128-unit areas entered, ammo spent, and
reaching the exit. The planner plays the candidate with the best
risk-adjusted score over its scenarios (mean minus half the gap to the
worst). No scenario uses the game's real seed, so it plans against plausible
futures, not the actual one. When the two best candidates score within 10%
of each other, Haiku (`claude -p`, low effort) can be asked to choose from the
candidates' end frames, with a short note carried from one call to the next.

Prior art: VideoGameBench Lite pauses the emulator while a vision-language
model chooses each action from the screen; its best models finished well under
1% of Doom. Two searches did not find a Doom agent that forks the engine to
play out candidate futures and judges them, which is what this tries.

## Layout

| path | contents |
|---|---|
| `engine/` | PrBoom 2.5 from retro-go via `oaustegard/m5`, with two hooks: analog movement and `I_HapticHit` (damage dealt) |
| `src/host_main.c` | Linux platform layer, the forking planner and ffmpeg recording |
| `play.py` | runs `doom_host` in the hub's `scripts/jail.sh`, answers JUDGE requests with Haiku, captions and uploads the video |
| `wads/DOOM1.WAD` | id Software's shareware IWAD |

## Status

- A first unjailed smoke run (10 s of E1M1, 24 candidates × 2 scenarios)
  played: it explored, picked up armour bonuses, shot, and took damage from
  an imp, at 2.6 s of wall time for 44 decisions.
- That run exposed a bug, since fixed in the source: `D_Display` runs
  `I_StartTic` once per tic of `I_GetTime`, and a clock that moved on every
  call ran it about 1.65 times per real tic. That double-counted tics and
  cut every rollout's horizon short. The fix has not been run yet.
- The next run was stopped by the session's auto-mode safety classifier.
