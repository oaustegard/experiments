# Jev plays from the hymnal

Jev performs a 32-bar piece one bar at a time. It picks each layer's next bar from a fixed catalog of vetted patterns, and code plays whatever it picks.

- It follows the piece's arc. The correlation between the energy of the chosen patterns and the arc's target is r = 0.89–0.96 over four runs, against 0.06 for random picks from the same catalog.
- Every decision came back well inside the bar: median 0.39 s, max 0.64 s, against a 2 s bar.
- Choosing a chord every bar collapsed to an Am/E vamp. Choosing a 4-bar progression from the catalog fixed it.

The movie (`results/jev-hymnal.mp4`, 75 s) is rendered offline from run `v2-sampled-3`. Jev's probabilities and latencies in it are the recorded ones.

## Question

This is idea 2 from memory `aae7ec10`. A congregation doesn't compose; it sings the hymn whose number is on the board. Can Jev act as a live performer that only ever selects, so that it can't produce a broken bar, and still shape a piece?

## Setup

`hymnal.py` holds the catalog of one-bar patterns (16 steps each):

| layer | options |
|---|---|
| drums | 10 |
| bass | 8 |
| keys | 6 |
| lead | 7 |
| chords (v1) | 7 |
| 4-bar progressions (v2) | 7 |

Each option carries a one-line description of its effect only ("a snare roll that speeds up through the bar and ends on a crash"), never when to use it. Every option also has a code-side energy of 0 to 3. Jev never sees the energies; they are used only to score how well a run follows the arc.

The arc has seven sections: intro, verse, build, drop, breakdown, finale, outro. Each has a one-sentence intent and a target energy.

`perform.py` makes one Jev call per bar (`jev-1.13.0`, direct TypeSafe):

- **State:** the bar's position in the arc, its section intent, and the last four bars' choices as text.
- **Questions:** one Choice per layer. In v2 there is also a progression Choice at bars 1, 5, 9 and so on.
- **Modes:**
  - *sampled:* draw from Jev's probabilities after dropping options under 0.05.
  - *argmax:* take the top option.
  - *random:* no Jev.

In the "Chords used" column below, v1 counts chords used over the piece. In v2 every run uses all 7, so the column shows the progression Jev picked for each phrase instead.

`synth.py` turns the choices into note events and numpy-synthesized stereo audio. `render.py` draws the frames with PIL and muxes them with ffmpeg: hymn boards with Jev's top-4 probabilities per layer, a step grid, a piano roll, the arc timeline with per-bar energy, and decision times.

## Results

| run | arc r | arc MAE | chords used | E → Am | hook bars in drop or finale | bars identical to previous |
|---|---|---|---|---|---|---|
| v1 sampled 1 | 0.81 | 0.47 | 4 | 3/15 | 10/11 | 2 |
| v1 sampled 2 | 0.80 | 0.44 | 3 | 6/18 | 11/11 | 0 |
| v1 sampled 3 | 0.89 | 0.36 | 3 | 6/16 | 11/11 | 7 |
| v1 argmax | 0.92 | 0.34 | 3 | 2/13 | 11/11 | 22 |
| v1 random (20) | 0.00 | 0.85 | | | | |
| v2 sampled 1 | 0.89 | 0.35 | home ×3, tension, home, anthem, brood, anthem | | 11/11 | 4 |
| v2 sampled 2 | 0.96 | 0.18 | home ×3, anthem ×3, brood, cadence | | 11/11 | 4 |
| v2 sampled 3 | 0.92 | 0.29 | home, warm, cadence, tension, cadence, anthem, brood, cadence | | 11/11 | 0 |
| v2 argmax | 0.94 | 0.30 | home ×3, tension, anthem ×2, brood, cadence | | 11/11 | 9 |
| v2 random (20) | 0.06 | 0.83 | | | | |

In v1, Jev stays on Am and E. It picks E for 13 to 18 of 32 bars, resolves E to Am only a third of the time, and holds E through most of the drop. The descriptions read "A minor, the home chord" and "E major: a strong pull straight back to A minor". Chosen bar by bar with four bars of history, those two options win and F, C, G and Dm almost never do. The first smoke call had already shown the same thing: after an E in bar 16 it chose E again for bar 17 (p = 0.67).

In v2, code plays a vetted 4-bar progression and Jev chooses only which one. That puts "tension" (Dm Em F E) in the build before the drop in three of four runs, "brood" (Am Dm Em Am) in the breakdown in all four, and "cadence" (F G E Am) at the end of three of the four. TypeSafe's own guidance is to keep structure in code and give the model narrow selections. Here the structure is the phrase, and it has to be one of the options.

The hook result shows Jev following instructions rather than working anything out. The drop and finale intents name the hook, and Jev never played it anywhere else. Argmax repeats itself (9–22 bars identical to the one before), so the movie uses a sampled run. The ViZDoom runs found the same thing: sampling from Jev's probabilities beats taking the top option.

## Cost and timing

- 256 Jev calls in all (v1 and v2, four runs each).
- About 1,270 input tokens per call, roughly $0.014 in total.
- Median latency 0.39 s, max 0.64 s over the 128 v2 calls. A live version would ask for bar n+1 while bar n plays, with at least 1.3 s to spare every time.

## Not done

- Live playback. The movie is rendered offline from a recorded run.
- A listening test. The arc score uses code-side energies that I assigned, which is a proxy for musical quality, not a measure of it.
- n = 4 Jev runs. The v1-to-v2 harmony change is large enough to see at that n. The differences between v2 runs are not.
- Any real-time input for Jev to react to, such as crowd noise or a second player. The arc is a fixed script, so this tests reading a plan, not listening.

## Files

- `hymnal.py` — catalog, arc, energies, v2 progressions
- `perform.py` — `python3 perform.py v1|v2` → `runs/<variant>-<mode>-<seed>.json`, `results/metrics-<variant>.json`
- `synth.py` — events and audio
- `render.py` — `python3 render.py runs/v2-sampled-3.json out.mp4` (needs `pip install imageio-ffmpeg`)
- `results/jev-hymnal.mp4` — the movie
