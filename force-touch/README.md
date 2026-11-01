# Force touch

Can a phone browser tell how hard a thumb presses? iPhones since the XS have no pressure-sensing screen, so `force` and `pressure` are expected to be fixed values. The open question is whether the browser still reports the size of the contact, which a firmer thumb flattens and enlarges. If it does, a game pedal (Tilt Rally's brake) could be progressive. As a second channel, the probe records the phone's motion sensors during each press: a firm press might tip the phone back measurably.

Status: **built; waiting for the first run on a real phone.** The page, observer and analysis have only run end to end with a stand-in thumb in headless Chromium (`test/`).

Origin: Oskar, 2026-11-01, after the Tilt Rally brake felt too sensitive: "I assume you can't register Force Touch? Could we register that by size of thumb press area?", then: "Let's do a Force Touch experiment in experiments with feedback to a running sibling instance, along with the instructions in the readme required for both the human and the bot as to how to re-play the experiment".

## Prior art

Using contact size as a stand-in for force is established: "Fat Thumb" (Boring et al., 2012) simulated pressure from the thumb's contact size. A 2024 *Scientific Reports* study on a Galaxy S21 divided sensor force by screen-reported thumb area and found area the parameter most tied to pressure, while the reported radii were noisy. Neither says what an iPhone browser exposes. The Pointer Events spec says `width` and `height` default to 1 when the hardware cannot report contact geometry. A 2016 Apple forum thread reported `Touch.force` as 0 on iPhones without 3D Touch. We did not find a current measurement of iPhone WebKit's `radiusX`/`width` in two searches. Getting that measurement is the purpose of this run.

## How it works

The phone page (`web/index.html`, live at <https://austegard.com/experiments/force-touch/web/>) joins a room on the `rally-lobby` Worker, the same matchmaking server Tilt Rally uses (source: `workers/rally-lobby` in oaustegard.github.io). It sends each finished press there as JSON. A Claude session runs `observe.mjs`, which joins the same room as `observer:claude` and writes every record to `results/raw-<ROOM>.jsonl`. `analyze.py` then reduces each press to one number per channel and reports which channels vary and how well each separates light from firm presses.

The phone is prompted for 13 held presses of 2 s each. Light, medium and firm are interleaved three times each, so drift can't masquerade as a trend. Then come two flat-thumb-light and two tip-of-thumb-firm presses. Those last four separate area caused by force from area caused by how the thumb lies: if "area" is really posture, flat-light will read larger than tip-firm.

For every pointer and touch event during a press, the page records:
- PointerEvent `pressure`, `width`, `height`, `tiltX` and `tiltY`;
- Touch `force`, `radiusX`, `radiusY` and `rotationAngle`;
- the motion sensors from 0.5 s before the press until 0.3 s after: acceleration with and without gravity, and rotation rate.

Browsers only fire move events when the touch changes, so a perfectly still thumb gives only its landing values. A real thumb jitters enough to produce more.

## Re-running it

### The bot (a Claude Code session in the cloud, or any machine with Node and Playwright)

Run everything from the root of this repo. You need Node, the `playwright` package and its Chromium; a Claude Code cloud container has these preinstalled. In a cloud session the observer goes out through the session's HTTPS proxy (`HTTPS_PROXY`), which it finds by itself. The phone must open the page from austegard.com; the room server refuses other sites.

1. Pick a **new room code for every attempt, including retries**: 4–8 characters, capital letters and digits only, for example `TOUCH7`. Avoid I, L, O, 0 and 1, which are easy to misread aloud. The observer rejects any other code.
2. Start the observer as a background job, so it outlives the turn:
   `node force-touch/observe.mjs TOUCH7 30`
   It waits up to 30 minutes and prints each press as it arrives (`press 3 firm#1: …`). If the proxy drops the socket, it prints `closed (…); reconnecting` and rejoins, and the phone then resends every press so far. It exits 5 s after the phone reports it is done. An earlier file for the same code is moved aside, never mixed in.
3. Give the human the link with the code in it: `https://austegard.com/experiments/force-touch/web/?room=TOUCH7`
4. When the job ends, check that the log shows 13 different press numbers. Resent presses appear twice in the log but only once in the analysis. Then run `python3 force-touch/analyze.py force-touch/results/raw-TOUCH7.jsonl`.
5. If the human sends pasted text instead (the Copy fallback), save it as `force-touch/results/raw-TOUCH7.jsonl`, one JSON record per line exactly as pasted, and run step 4.
6. Write up following this repo's `CLAUDE.md`: `RESULTS.md` with Answer, Findings, Method and Log, `_lib/results_lint.py`, evidence tags pointing into `results/summary-TOUCH7.json`, the index row, and a `METHODS.md` entry if the result would change another experiment. Commit the raw and summary files; they hold no personal data beyond the browser's user-agent string.

**Reading the analysis.** "varies NO" with a constant value means the phone doesn't report that channel: the defaults are pressure 0 or 0.5 (or 1), width and height 1, force 0. AUC firm>light is the chance that a random firm press reads higher than a random light one: 0.5 means no information, 1.0 perfect separation, 0.0 perfectly reversed. AUC tip>flat repeats the comparison for tip-firm against flat-light. A value near 0 there means the channel tracks how the thumb lies more than how hard it presses. With 3 light against 3 firm presses (9 pairs) and 2 against 2 (4 pairs), these are coarse: one run can show that a channel works or is absent, not how accurate it is. A press where the thumb never moved reports only its landing values.

### The human (on the phone being measured)

1. Open the link the bot gave you. The room code at the top must match. The status line should read "Claude is watching room …". If it says Claude isn't watching yet, wait for the bot to start its observer; the line changes by itself when it does.
2. Hold the phone sideways in both hands, as in Tilt Rally, and use your right thumb on the pad.
3. Tap **Start** and allow motion access if asked. Refusing only loses the motion data; the live view then shows "motion off".
4. For each of the 13 prompts, read it, then wait for the pad to turn yellow and say Press. Touches before that are ignored. Press as the prompt says and hold until the green bar fills (2 s). A press under 0.4 s doesn't count: the prompt says so and repeats.
5. Watch the numbers on the left while you press. `pressure` and `force` are expected to stay fixed on an iPhone. If `width`, `height` or `radius` move as you press harder, the phone reports contact size.
6. When it says Done, tell the bot. If the status said "Disconnected; retrying…" at any point, finish anyway: the presses are resent when it reconnects. If the status never said Claude was watching, tap **Copy results** and paste the text to the bot.

To compare conditions, repeat with a different code per run: another phone, Safari against Chrome, or a different thumb.

### Without a phone (plumbing check)

`LOBBY_DIR=../oaustegard.github.io/workers/rally-lobby node force-touch/test/e2e.mjs` runs the page, a local room under `wrangler dev`, and the observer, with Chromium's touch emulation as the thumb. `test/synthetic-run.jsonl` is one such run. In it, the stand-in's contact sizes separate light from firm perfectly, and the posture check scores 0, flagging the flat-thumb confound that was built in on purpose. No motion data comes through, since headless Chromium has no motion sensors.

## Results

None yet. `RESULTS.md` follows the first phone run.
