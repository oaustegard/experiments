# jev-live-set

Artifact: https://claude.ai/artifact/FqKu246ZMecrNvm2UP8KCT (private to Oskar until shared)

Oskar asked whether Jev can rate its own knowledge of music theory and Strudel,
and whether it can choose among several Strudel options for the next bar. He
then asked for a less constrained successor to `jev-hymnal/`: a live stream
where Jev makes the choices, with a visual.

## What Jev said, and what it did

Numbers are in `results/jev-probes.json`.

Asked to rate itself on a 0–3 scale, Jev put music theory at 1.34, between
basic and competent. It put Strudel at 1.69 (competent) and choosing the next
bar at 1.51. Asked whether it could predict what a snippet sounds like, it said
yes with p = 0.53. The quiz scores were higher than the self-rating:

| probe | score | note |
|---|---|---|
| music theory | 11/11 | p 0.81–1.00; lowest on Dorian |
| Strudel syntax | 10/11 | missed `n("0 2 4").scale("C:minor")`: answered C D E♭ at p 0.89 |
| next bar, one clear best option | 5/5 | p 0.91–1.00 |
| next bar, all plausible, steer "more movement, not louder" | 3/3 | p 0.99–1.00 on the same-volume variation |
| F chord as code (`n().scale()`) vs as words | 0.46 vs 1.00 | right both times, near chance on code |

The last two rows set the design. Jev reads musical intent well and computes
pitches badly, the same pattern as the count and arithmetic weaknesses TypeSafe
documents. Every option therefore reaches Jev as a description with an energy
tag. Code fills in the notes.

Jev also answered four design questions. It chose per-bar choices for layers
and 4-bar choices for harmony and structure (0.81). For the visual it chose a
lane score shown next to the rejected options and their probabilities (1.00).
It chose free-text steering plus buttons (0.99). It voted 0.67 to keep one key
and one style. The page overrides that vote because Oskar asked for less
constraint.

## The page

- **Catalog.** Six styles (house, techno, lo-fi, ambient, synthwave, dub), with
  2–7 patterns per layer plus a mute option. The layers are drums, bass,
  harmony, lead and texture. Patterns are Strudel code with chord placeholders:
  `{R2}` is the root in octave 2, `{V3}` a triad voiced from E3, `{SC4}` the
  key's scale. `fillTemplate` fills them per bar from the progression, so one
  bass line follows every chord in every key.
- **Decisions.** Each bar is one Jev call with five Choice questions. At the
  last bar of each phrase the same call also asks four phrase questions:
  section, progression, style, and the key for the phrase after next. The key
  is decided one phrase ahead so that progressions are always offered in a
  known key. Picks are sampled from Jev's probabilities at temperature
  "Adventure" (default 0.8). `jev-hymnal` found that argmax repeats itself.
- **Timing.** Decisions run 1–3 bars ahead, adapting to p90 reaction time, and
  calls are paced at least 1.25 s apart under the gateway's 50/min limit. A bar
  is locked 0.4 s before it starts (1.6 s when the tab is hidden). A late answer
  holds the previous parts over the new chord. A steer drops every decided bar
  that hasn't been locked and the next phrase decision, then those are
  re-decided with the request in the state for 12 bars.
- **Audio.** It runs real Strudel pattern code: `@strudel/core`, mini-notation
  and tonal 1.2.6, bundled with esbuild into an inline IIFE. `@kabelsalat/web`
  is stubbed because core 1.2.6 imports a `SalatRepl` that 0.4.1 doesn't
  export. A small evaluator parses the method-chain subset of JS the catalog
  uses, so the page never calls `eval`. Playback is a Web Audio synth written
  for the page: synthesized 808/909/707-flavoured drums, oscillators and a
  supersaw, ADSR, filters, a waveshaper, a convolver reverb bus and a delay
  bus. The artifact CSP blocks Strudel's sample and worker fetches. strudel.cc
  links carry the exact bar code.
- **Visual.** A canvas score with one lane per layer and notes from the queried
  haps. Bars already decided are drawn to the right of the playhead, and held
  bars get a dashed outline. The backdrop glows with RMS level, and its hue
  follows section energy. The "Why this bar" panel lists every option with its
  probability and energy dots.
- **Random dice.** The same catalog with uniform picks, for A/B listening, and
  the fallback when the connector is missing.

## Verification

- All 101,376 combinations (every option × 24 keys × every progression chord)
  compile, return haps, and yield valid MIDI notes, in headless Chromium
  (`python3 harness.py selftest`).
- Headless runs in random mode and with a mock connector (300–600 ms latency).
  The run decided 1–2 bars ahead with 0 held bars. A typed steer deleted the
  unlocked bars and the next phrase, and they were re-decided.
- Real Jev, on requests the page generated:
  - Opening: intro 0.99, Am–F–C–G 0.42, techno 0.51 (ambient 0.29), stay in
    A minor 0.90.
  - A dub outro bar: expected drum energy 1.02 without a request and 2.09 with
    "more energy, push it harder". Lead went from silent (0.78) to the melodica
    phrase (0.51). Jev settled between the request and the outro's energy
    target of 1.
  - About 1,400 input tokens per bar call, about $0.00006.
- **Not verified:** audio output and a live Jev call from inside claude.ai (no
  audio device or connector bridge in CCotw), reaction time through the
  artifact MCP path, and whether it sounds good. The earlier Prose Physics
  artifact used the same connector call shape.

## Files

`page.src.html` (the page, with a `/*__STRUDEL_LIB__*/` placeholder) ·
`build.py` (inlines `strudel-lib.js`) · `bundle/` (esbuild config for the
Strudel IIFE: `cd bundle && npm i && npm run build`) · `harness.py`
(selftest / random / mock-Jev runs) · `results/jev-probes.json` ·
`results/screenshot-mock.png`
