# Swarm run statistics (as of 2026-10-13T22:43:48Z)

workers 2489, spend $281.13, merges on main 862, $0.3261 per merge, reverts 0

| worker kind | merged | rejected | failed | reverted | launched | finished |
|---|---|---|---|---|---|---|
| feature | 154 | 92 | 102 | 0 | 0 | 0 |
| high effort | 216 | 149 | 241 | 0 | 0 | 0 |
| normal | 438 | 314 | 682 | 0 | 0 | 0 |
| rebase | 54 | 44 | 3 | 0 | 0 | 0 |

## Harness events

- 09:45 [refresh] full refresh at 395118e01e
- 09:46 [note] launch obscure target (neat*) at -O2, concurrency 16
- 09:46 [loop] loop start: run --config swarm/swarm.toml --only neat* --concurrency 16 --max-usd 33 --max-minutes 110 --batch 6 --sample 0
- 10:04 [loop] loop stopped
- 10:05 [note] runner rejects gcc-steering attributes (used/noinline/noipa); stripped from 31 files, 14 matches lost
- 10:05 [refresh] full refresh at 57181ebc3c
- 10:05 [loop] loop start: run --config swarm/swarm.toml --only neat* --concurrency 16 --max-usd 33 --max-minutes 110 --batch 6 --sample 0
- 10:59 [loop] loop stopped
- 11:05 [note] partial-credit gate on (frac, min gain 0.05): 125 partials (39% of spend) had been discarded
- 11:05 [refresh] full refresh at e66e329087
- 11:06 [loop] loop start: run --config swarm/swarm.toml --only neat* --concurrency 16 --max-usd 36 --max-minutes 110 --batch 6 --sample 0
- 11:58 [note] neat phase ended ($41.12); lua phase starts, cap $54.5 cumulative
- 11:58 [loop] loop start: run --config swarm/swarm.toml --only lua* --concurrency 16 --max-usd 54.5 --max-minutes 110 --batch 6 --sample 0
- 12:28 [loop] loop stopped
- 12:28 [note] Oskar lifts the $60 cap: both targets run together, soft cap $150 cumulative
- 12:28 [loop] loop start: run --config swarm/swarm.toml --only neat*,lua* --concurrency 20 --max-usd 150 --max-minutes 110 --batch 6 --sample 0
- 12:37 [refresh] full refresh at 0e58625018
- 14:19 [loop] loop exit at --max-minutes (relaunch to continue)
- 14:19 [loop] loop start: run --config swarm/swarm.toml --only neat*,lua* --concurrency 20 --max-usd 145 --max-minutes 110 --batch 6 --sample 0
- 14:21 [refresh] full refresh at ec02193b5e
- 15:31 [loop] loop stopped
- 15:31 [note] Oskar: run until neat stalls (spend cap lifted)
- 15:31 [loop] loop start: run --config swarm/swarm.toml --only neat*,lua* --concurrency 20 --max-usd 1000 --max-minutes 110 --batch 6 --sample 0
- 16:23 [unpark] attempt epoch: parked tests unparked
- 16:23 [note] unpark: queue at 24 items; attempts before partial credit (11:05) discarded progress
- 17:22 [loop] loop exit at --max-minutes (relaunch to continue)
- 17:22 [loop] loop start: run --config swarm/swarm.toml --only neat*,lua* --concurrency 20 --max-usd 1000 --max-minutes 110 --batch 6 --sample 0
- 17:50 [loop] loop stopped
- 17:50 [unpark] attempt epoch: parked tests unparked
- 17:50 [note] tail phase A: Haiku at xhigh for every attempt (harness 0.4.0 effort ladder); unpark
- 17:50 [loop] loop start: run --config swarm/swarm.toml --only neat*,lua* --concurrency 20 --max-usd 1000 --max-minutes 110 --batch 6 --sample 0
- 17:59 [refresh] full refresh at c3c046c914
- 19:40 [loop] loop exit at --max-minutes (relaunch to continue)
- 19:40 [loop] loop start: run --config swarm/swarm.toml --only neat*,lua* --concurrency 20 --max-usd 1000 --max-minutes 110 --batch 6 --sample 0
- 19:47 [loop] loop stopped
- 19:47 [unpark] attempt epoch: parked tests unparked
- 19:47 [note] tail phase B: Haiku max for every attempt; unpark (xhigh phase: 295 workers, 28 flips, $41.38)
- 19:47 [loop] loop start: run --config swarm/swarm.toml --only neat*,lua* --concurrency 20 --max-usd 1000 --max-minutes 110 --batch 6 --sample 0
- 21:17 [loop] loop stopped
- 21:17 [note] paused 21:17 for 45 min (quota); 21 in-flight max workers terminated
- 22:05 [note] resumed after quota pause (22:05)
- 22:05 [loop] loop start: run --config swarm/swarm.toml --only neat*,lua* --concurrency 20 --max-usd 1000 --max-minutes 110 --batch 6 --sample 0
- 22:34 [loop] loop stopped
- 22:34 [note] Oskar 22:20: stop at Haiku max, no Sonnet; launches end, in-flight max workers drain (--no-launch)
- 22:34 [loop] loop start: run --config swarm/swarm.toml --only neat*,lua* --no-launch --concurrency 20 --max-minutes 60 --batch 6 --sample 0
