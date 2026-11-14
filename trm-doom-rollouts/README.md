# trm-doom-rollouts

The [trm-doom](../trm-doom/) bot plays [gPu(doom)](https://austegard.com/games/doom/), a DOOM whose game logic
runs on the GPU. A small neural network (TRM, Samsung's Tiny Recursive Model) plans its routes, and
hand-written code chooses targets and fights. This experiment replaces only the hand-written fighting. In a
fight, every 12 tics (DOOM's 1/35 s steps, so a third of a second), it copies the game on the GPU and plays
6 moves for 2 s each, each in 2 copies with different random numbers. It then makes the move whose copies
came out best on damage dealt and taken, kills and deaths. The 6 moves are the bot's own move, stand still, and toward, away, left and right relative
to the enemy.

Over 18 pairs on levels 4 to 6 (same level, seed and code in both arms, 200 s each), the unchanged bot died
16 times and with rollout fights once; kills went from 288 to 663. Rollout fights reached the exit once in 18
runs; the rest were alive at the 200 s cap.
Results, method and the log: [`RESULTS.md`](RESULTS.md).

## Layout

| path | contents |
|---|---|
| `web/fork.js` | `Forker`: saves and restores the game's four state buffers on the GPU, and runs a tic without drawing it |
| `web/combat.js` | `RolloutBot`: wraps the trm-doom `Bot`, decides fights by rollouts, passes everything else through |
| `web/runner.js` | plays one level without a screen, reading the game state back every tic; `wrap` puts a controller around the bot |
| `bridge/job.js` | a job for the GPU bridge (a web page on a laptop that runs jobs pushed to a repository): both arms on a real GPU, TRM routes |
| `test/fork-check.html` | checks that a restored game continues exactly as the original, and times tics |
| `test/play.html`, `test/play.py`, `test/run_page.cjs` | one level in headless Chromium on SwiftShader, Chromium's software GPU; writes `results/runs/NAME.json` |
| `test/batch.sh`, `test/batch2.sh` | the SwiftShader batches (`batch.sh` is the first, TRM-routed version, abandoned as too slow); `batch2.sh` takes `LEVEL:SEED` pairs, runs both arms and commits each result as it lands |
| `summarize.py` | builds `results/summary.json` from the bridge results and the SwiftShader runs |
| `results/bridge/<job>/result.json` | GPU-bridge results from the Mac (and two jobs an iPhone tab lost when its WebGPU device died) |
| `results/runs/sw-*.json` | SwiftShader runs, with every rollout decision's scores |

## Running it

Needs Node with Playwright and the pre-installed Chromium, and a checkout of
[oaustegard.github.io](https://github.com/oaustegard/oaustegard.github.io) next to this repo
(`test/run_page.cjs` serves the parent directory of both). From `trm-doom-rollouts/`:

```sh
python3 test/play.py sw-oracle-roll-E1M5-s74 'map=E1M5&skill=2&tics=7000&seed=74&arm=roll&planner=oracle'   # one level
test/batch2.sh E1M4:74 E1M5:111                                                         # both arms, BFS routes
python3 summarize.py
```

Each comparison has two arms: `arm=base` is the unchanged bot, `arm=roll` the one with rollout fights,
played on the same level and seed as a pair. `planner=oracle` routes with exact BFS (`OraclePlanner`).
`summarize.py` reads results named `sw-oracle-<arm>-<level>-s<seed>`, the names `batch2.sh` gives them; the controller's settings (6 moves, 2 copies, 70 tics, every 12 tics) are the defaults in `web/combat.js`. A rollout run takes 40 to 96 minutes on SwiftShader. `planner=trm` uses TRM for routes. On SwiftShader that is too slow to be
practical, so the TRM runs went through `bridge/job.js` on the GPU bridge (`docs/gpu-bridge.md` in
oaustegard/claude-workspace). A rollout job keeps a laptop GPU at full load for minutes; leave it time to
cool between jobs.
