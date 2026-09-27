# Strudel FM: a more melodic lead

Oskar asked (2026-09-27) whether the tunes on [Strudel FM](https://austegard.com/fun-and-games/strudel-fm.html) could be more melodic. This directory holds the harness that measured the lead line before and after the rewrite, and the audio takes.

## What was wrong

The old lead drew a one-bar motif per phrase: onsets by coin flip on every sixteenth, pitches by a random walk. Alternate bars played the walk inverted. On-beat notes then snapped chromatically to the nearest chord tone within three semitones, and held notes were pushed off any rub with the voicing. The snapping broke the walk's shape: consecutive notes landed on the same chord tone or jumped between chord tones. The same bar rhythm repeated four times per phrase, and a new motif was drawn whenever the progression or section changed, so no tune came back.

## What changed

The rewrite is in `fun-and-games/strudel-fm.html`, in `writeLead` and its helpers:

- The station keeps two two-bar tunes per style, a hook (A) and a contrasting line (B). A rhythm is built from one-beat cells weighted per style, with no ties over the bar line, and ends on a held note or a rest. The contour is a scale-degree walk with an arch, rise, fall or wave shape, 60% steps, and a step back after any leap.
- Form: grooves and drops play A. It comes back plain every other phrase; in between it is ornamented with passing notes, restated a third higher, or answered by B in bars 3–4. Lifts play B. Breakdowns and Chill grooves play A thinned to its half-bar notes. After about ten phrases of A, B may become the new hook.
- Fitting: each bar of the tune is transposed diatonically (±3 degrees) to the position that puts the most on-beat and held notes on chord tones with no semitone or minor-ninth rub against the voicing. A note left on a non-chord tone on the beat moves one step in the direction the line is travelling, not onto the previous pitch.
- Cadence: bar 4 ends on a held chord tone, preferring the root, then the third. If the line is already on that note it is held rather than struck again.
- The lead now plays in Chill grooves (thinned). It is still silent in intros below energy 2.

## Measurements

`gen.mjs` loads the page headlessly, pins the seed, and writes 24 phrases (96 bars) for each of 7 stations × 3 energies × 4 seeds (8,064 bars per version). `metrics.py` reads the lead line and chord voicing from each bar's generated code. Full table: `metrics.txt`.

| all stations | before | after |
|---|---|---|
| bars with a lead line | 79.2% | 87.6% |
| intervals of 1–2 semitones (steps) | 22.5% | 38.7% |
| repeated notes | 29.1% | 16.6% |
| leaps over 4 semitones | 21.7% | 16.5% |
| leaps over 7 semitones | 6.1% | 3.4% |
| leaps followed by a step or small interval back | 20.6% | 58.6% |
| onsets on odd sixteenths | 22.4% | 13.1% |
| bars whose rhythm repeats inside the phrase | 85.7% | 73.7% |
| on-beat or held notes rubbing the voicing | 0.0% | 0.0% |
| phrase-final note on the chord root | 41.5% | 40.5% |
| compile errors | 0 | 0 |

Steps are counted in semitones, so a whole-tone step counts and a minor third does not. Ambient and sleep stay lowest on steps (32% and 22%): their tunes have two to four long notes per two bars and move between chord tones.

## Audio

`rec.mjs` records the page through its own Web Audio output (the listening-to-music tap) with seed 11 from 14 s after POWER, 45 s per take, before and after, for house, lo-fi, synthwave and ambient. The page loads its sample banks through the proxy, so the browser context sets `ignoreHTTPSErrors`; without it the drum and piano banks fail to load and those parts are silent. Takes are in `takes/` as MP3; each before/after pair is scaled by one gain, so loudness compares. House and synthwave peak near 1.7 before scaling in both versions, so those mixes may clip at full volume on the live page.

## Files

- `gen.mjs`: bar dump harness (injects a seed/reset hook at `window.__fm`).
- `metrics.py`: melody metrics over one or more dumps.
- `rec.mjs`, `record_all.sh`: audio takes.
- `before.html`, `after.html`: the page at each version (`*-seeded.html` add `window.__fmSetSeed`).
