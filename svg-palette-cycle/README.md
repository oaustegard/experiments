# svg-palette-cycle

![Four tiles of a cat, cycling palettes](cat-cycle.svg)

[`cat-cycle.svg`](cat-cycle.svg) is a Warhol 2×2 of one cat that keeps cycling four palettes (hot, jungle, ocean, grape), and at every moment the four tiles show four different ones. The file is 72 KB, with no JavaScript. `python3 build.py [source] [out.svg]` rebuilds it; the default source is MediaPipe's sample `cat.jpg`.

## How it is built

1. image-to-svg 1.9.0 traces the source once (`mode="illustration"`, `K=18`, `kuwahara:3`, upscaled 2× first), with a palette of six placeholder grays. Its luminance banding maps the 18 clusters onto those six tone bands, and its compact output gives each band one CSS class.
2. The traced paths go into a `<symbol>`. Each band class is restyled as `fill:var(--bN);stroke:var(--bN)`, and each `--bN` is registered with `@property` as a `<color>` so it can interpolate.
3. One `@keyframes cycle` steps through the four palettes, holding each for 70% of its slot and easing into the next. Four `<use>` tiles run it with delays of 0, −3, −6 and −9 s on a 12 s cycle. Since the delays are quarter-cycle rotations of the same sequence, the tiles never share a palette, including mid-fade.
4. Each tile also sets its palette statically, which is what `prefers-reduced-motion` shows.

The geometry is stored once and drawn four times, so the size is one tracing plus about 2 KB of CSS.

## Compared with the 2026-03-31 squirrel attempt

| approach | result |
|---|---|
| `feColorMatrix` hue-rotate, different speeds per tile | tiles converge to the same hue regularly |
| hue-rotate inside bounded 90° bands | neighbouring bands look alike; 3 of 4 tiles green at once |
| SMIL visibility swap of 4 baked palettes | always distinct, but hard cuts every 2 s |
| per-band `@property` colors, one keyframes, rotated delays (this) | always distinct, smooth crossfades, geometry stored once |

## Limits

- `@property` is needed for the crossfade: Chrome 85+, Safari 16.4+, Firefox 128+. Without it custom properties do not interpolate, and the palettes switch at the midpoint of each fade.
- `var()` only works through CSS, so the background rect gets `class="bg"` rather than `fill="var(...)"`.
- Luminance banding is what makes it read as a screen print, but green background and orange fur of similar brightness land in the same band.
