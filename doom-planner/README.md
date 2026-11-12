# doom-planner

A Doom player that decides by playing out futures instead of reading the
screen. Doom advances in steps of 1/35 s called tics. Every 8 tics (0.23 s)
the planner copies the running game 90 times: 30 options (29 moves plus
"keep going") under 3 scenarios, each a copy with its random number generator
reseeded, played for 52 tics (1.5 s). Each copy is scored from engine state: damage dealt and taken,
kills, deaths, walls seen for the first time, walking distance to unexplored
space, and the exit. The planner plays the move with the best average score,
pulled halfway toward its worst scenario. No copy uses the game's real seed,
so it plans against futures that could happen, not the one that will. When
the two best moves are close, Haiku can be asked to choose from where each
one leads.

Results, method and the log of what broke: [`RESULTS.md`](RESULTS.md). Videos
of the three 5-minute runs are on the [Doom by Rollouts](https://claude.ai/artifact/LTAJbziwPNEFX8E49modKu)
page (private until shared) and full-size on Cloudflare R2, at the public
URLs in `results/*.json`.

Prior art: VideoGameBench Lite pauses the emulator while a vision-language
model picks each action from the screen (arXiv 2505.18134); the best model's
average progress through the games was 0.48%, Doom among them. Two searches did not find a Doom agent that
forks the engine and judges real outcomes.

## Layout

| path | contents |
|---|---|
| `engine/` | PrBoom 2.5 from retro-go via `oaustegard/m5`, with two hooks: analog movement and `I_HapticHit` (damage dealt by the player) |
| `src/host_main.c` | Linux platform layer, the forking planner, and ffmpeg recording |
| `play.py` | runs `doom_host` inside a sandbox with no network (`scripts/jail.sh` from `oaustegard/claude-workspace`), asks Haiku when the game binary requests a judgement, captions the video, uploads it to R2 |
| `report.py` | collects each run into `results/summary.json` |
| `make_page.py`, `page_template.html` | build the published page |
| `wads/DOOM1.WAD` | id Software's shareware IWAD |

## Running it

```sh
make                                   # builds doom_host
python3 play.py --name e1m1-explore --max-seconds 300
python3 play.py --name e1m1-assisted --max-seconds 300 --assist-exit
python3 play.py --name e1m1-judged --max-seconds 300 --judge
python3 report.py e1m1-explore e1m1-assisted e1m1-judged
```

`play.py` expects `oaustegard/claude-workspace` checked out at
`/home/user/claude-workspace` for the sandbox script and the R2 upload helper; pass `--no-upload` to skip R2. A 5-minute run
takes about 4 minutes of wall time on 4 CPUs without the judge and about 24
with it.

## License

GPL-2.0 (`LICENSE`), as PrBoom is; `play.py`, `report.py` and
`make_page.py` are MIT, as marked. `DOOM1.WAD` is id Software's shareware
release, redistributable unmodified.
