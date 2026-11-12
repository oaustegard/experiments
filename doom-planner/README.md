# doom-planner

A Doom player that decides by playing out futures instead of reading the
screen. Doom advances in steps of 1/35 s called tics. Every 8 tics (0.23 s)
the planner copies the running game 90 times: each of 30 options is tried in
3 copies, and each copy has its random number generator reseeded differently.
29 options are moves; the 30th, "keep going", resumes a move that takes
longer than 8 tics, such as "turn 90°, then run". Each copy plays its option
for 52 tics (1.5 s). Each copy is scored from engine state: damage dealt and taken,
kills, deaths, items, ammo used, standing still, walls seen for the first
time, walking distance to unexplored space, and a bonus for reaching the
exit. The planner plays the move with the best average score,
pulled halfway toward its worst scenario. No copy uses the game's real seed,
so it plans against futures that could happen, not the one that will. When
the two best moves score within 10% of each other, Haiku (Anthropic's smallest Claude model) can
be asked to choose from pictures of where each one leads.

With no knowledge of where the exit is, it cleared the first level, E1M1,
in 2:43 of game time with all six monsters killed and no deaths; it did not
find the second level's exit in 15 minutes. Two other runs have Haiku choose
every move. From the screen alone it did not clear E1M1 in five minutes;
choosing from each option's played-out outcomes, it cleared it in 4:27 after
two deaths.

Results, method and the log of what broke: [`RESULTS.md`](RESULTS.md). Videos
of the runs are on the [Doom by Rollouts](https://austegard.com/experiments/doom-planner/web/)
page and full-size on Cloudflare R2, at the public
URLs in `results/*.json`.

Prior art: VideoGameBench Lite pauses the emulator while a vision-language
model picks each action from the screen (arXiv 2505.18134); the best model's
average progress through the games was 0.48%, Doom among them. Two searches did not find a Doom agent that
forks the engine and scores the outcomes it actually plays out.

## Layout

| path | contents |
|---|---|
| `engine/` | PrBoom 2.5 (copied from the Doom port in `oaustegard/m5`) with two hooks: analog turning and movement, so an option can turn at a set rate, and `I_HapticHit`, which reports damage the player deals |
| `src/host_main.c` | Linux platform layer, the forking planner, and ffmpeg recording |
| `play.py` | runs `doom_host` inside a sandbox with no network (`scripts/jail.sh` from `oaustegard/claude-workspace`), asks Haiku when the game binary requests a judgement, captions the video, uploads it to R2 |
| `report.py` | collects each run into `results/summary.json` |
| `make_page.py`, `page_template.html` | build the published page |
| `wads/DOOM1.WAD` | id Software's shareware IWAD |

## Running it

Needs Linux, `gcc` and `make`, `ffmpeg` with libx264 and libass, and Python 3
with Pillow. The Haiku modes (`--judge`, `--haiku-screen`, `--haiku-plan`) also need the
`claude` command-line tool, signed in. `play.py` runs the game inside `scripts/jail.sh` from
`oaustegard/claude-workspace`, a sandbox with no network, and expects that
repository at `/home/user/claude-workspace`; edit the `JAIL` path at the top
of `play.py` to point elsewhere. Uploading writes to the `austegard-media` R2
bucket through that repository's `scripts/embstore.py` (a general R2
uploader), with the
`CF_ACCOUNT_ID` and `CF_API_TOKEN` environment variables, so use
`--no-upload` without access to that bucket.

From this directory:

```sh
make                                   # builds doom_host
python3 play.py --name e1m1-explore --max-seconds 300 --no-upload
python3 play.py --name e1m1-assisted --max-seconds 300 --assist-exit --no-upload
python3 play.py --name e1m1-judged --max-seconds 300 --judge --no-upload
python3 play.py --name e1m1-haiku-screen --max-seconds 300 --haiku-screen --no-upload
python3 play.py --name e1m1-haiku-plan --max-seconds 300 --haiku-plan --no-upload
python3 report.py e1m1-explore e1m1-assisted e1m1-judged e1m1-haiku-screen e1m1-haiku-plan
python3 make_page.py                   # only after runs made without --no-upload
```

Runs start at E1M1 (`--warp E M` for another map) and carry on into the
next level after the exit. `--max-seconds` is game time; the 15-minute run
used `--max-seconds 900`. `--assist-exit` adds points for walking toward the
level's exit switch; `--judge` hands close calls to Haiku;
`--haiku-screen` and `--haiku-plan` have Haiku choose every move, from the
screen alone or from each option's played-out outcomes. Each Haiku call
takes about 5 s, so a 5-minute run of either takes close to two hours. Each run
writes `runs/<name>/` (the raw and captioned videos, `decisions.jsonl` with
every option's score at every decision, the game's console output, and
Haiku's answers) and `results/<name>.json`; `report.py` collects those into
`results/summary.json`. `make_page.py` takes the video links from those
files, so it needs runs made without `--no-upload`. A 5-minute run takes about 4 minutes of wall time on
4 CPUs without the judge and about 24 with it.

## License

GPL-2.0 (`LICENSE`), as PrBoom is; `play.py`, `report.py` and
`make_page.py` are MIT, as marked. `DOOM1.WAD` is id Software's shareware
release, redistributable unmodified.
