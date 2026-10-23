# svg-timeline

![Starry Night painting itself](starry-timeline.svg)

[`starry-timeline.svg`](starry-timeline.svg) is Van Gogh's *Starry Night* painting itself on a 48 s loop, written as one static SVG with CSS keyframes and no JavaScript. A charcoal underdrawing draws itself on, the paint lands shape by shape, the scene grades to deep night with the stars glowing and twinkling, then to dawn, then back. Finally the paint falls off the sketch and the sketch erases itself. The file is 849 KB, or 223 KB gzipped. `python3 build.py [source] [out.svg]` rebuilds it from `src/starry_night.jpg` in about 5 s.

It answers [saiashirwad/svjeez](https://github.com/saiashirwad/svjeez)'s `minecraft.svg`, which compiles a hand-authored TSX scene into the same kind of file. svjeez has no raster input, so everything in its scenes is built from primitives. Here the scene's material is a traced painting, and the authored part is the timeline.

## In the browser, on any picture

[`web/index.html`](web/index.html) is the same scene as a tool page: pick or drop a picture and it builds the animated SVG on the page and offers it as a download. It has a scrubber with the beats marked, and nothing is uploaded. It is one self-contained file. `python3 web/bundle.py` builds it from [`web/scene.js`](web/scene.js) (the timeline compiler and scene, ported from `timeline.py` and `build.py`), [`web/page.html`](web/page.html), and the tracer internals sliced from [`svg-in-browser/web/trace.js`](../svg-in-browser/web/trace.js). It is published as austegard.com/fun-and-games/self-painting.html.

The browser version has no night or dawn. Oskar's verdict on the color grade: "that just looks like a glitch in the rendering." Its loop is sketch, paint, hold, fall and erase, 34 s by default, and the output no longer needs `@property`. The Python build further down keeps the grade as it was first made.

The port changes these things:

- **Regions.** The browser tracer follows pixel edges with shared, once-simplified boundaries, so its regions tile the canvas exactly. Each 4-connected component is traced as its own label, which gives per-region shapes for the time slots with no pinholes, so the ground layer is gone. Paths are quadratic curves, not polygons.
- **No layer-forcing animation on the paint.** An animated `opacity` or `transform` on a group makes Chromium composite that group separately on every frame, even while the value holds at 1 or `none`. With 59 paint groups covering the canvas, night ran at 9 fps. The same file with those animations removed ran at 50–57. The paint groups now animate only inherited properties (`fill-opacity`, `stroke-opacity`, `stroke-width`). A patch lands as a fat stroke that thins to the 1-unit seam cover (9 → 1), and falls by thinning to nothing.
- **Glow, while it existed.** Before night was dropped, the glow was radial-gradient halos rather than blurred copies. On six small glow groups, plain `opacity` was the fast option: 56 fps, against 12 for custom properties multiplied in `calc()`.

| headless Chromium, SwiftShader (34 s loop) | t=8 paint | t=20 hold | t=26 fall | t=32 erase |
|---|---|---|---|---|
| Starry Night, 17,523 regions, 1,063 KB / 284 KB gz | 19 | 57 | 19 | 60 |
| dog photo, 10,434 regions, 836 KB / 224 KB gz | 25 | 58 | 22 | 60 |

The trace takes 2.5–3.5 s in JavaScript k-means at 1000 px on this container. WebGPU wasn't available to measure here.

## Pieces

| file | lines | what |
|---|---|---|
| `timeline.py` | 99 | The compiler. A `Motion` is a list of states on an absolute clock (`Motion(opacity=0).at(2).to(0.7, "out", opacity=1)`), and each segment's easing is stored on the keyframe where the segment starts. `Sheet` interns keyframe bodies and animation rules by their exact text, so elements that move alike share one class, and it merges keyframe percentages that have identical declarations. These are svjeez's `motion()` and interning ideas, ported. |
| `build.py` | 295 | The scene: a `BEAT` table of named times, plus the four layers below. |

The layers, bottom to top:

1. **Canvas and ground.** A primed-linen rect that darkens at night. Above it, a ground rect in the trace's background color fades in from 11 to 15 s, which hides the pinholes the trace leaves between shapes, and fades out as soon as the fall begins.
2. **Underdrawing.** 124 outlines: region borders of a blurred, 5-color posterization of the source, simplified, Chaikin-smoothed twice, with stretches along the canvas edge removed. Each outline has `pathLength="1"` and animates `stroke-dashoffset` 1→0 to draw on and 0→−1 to erase, both in a left-to-right sweep. Adjacent regions each draw their shared border, which gives the doubled line of a real sketch.
3. **Paint.** image-to-svg 1.9.0 (`painting`, `kuwahara:3`, integer grid) gives 4,751 shapes in 56 colors. Each shape's start time combines size and position: 55% size rank (big forms first, like an underpainting) and 45% a left-to-right, top-to-bottom sweep, plus a little jitter. That gives a slot on a 0.2 s grid. Each slot is one `<g>` that fades in while settling 10 px, and falls 40 px while fading out in the same order. Inside a slot, shapes of the same color merge into one path.
4. **Glow.** The 501 shapes whose color reads as starlight (bright and warm) are duplicated into 6 blurred groups with `mix-blend-mode:screen`. Each group twinkles on its own period (2.4, 3.2 or 4.0 s, phase-offset), and a 48 s envelope turns the whole glow layer on only for the night.

Color grading uses the cat's trick from [`svg-palette-cycle`](../svg-palette-cycle/README.md): each of the 56 colors is an `@property --cN` of type `<color>` on the root, and one `@keyframes` moves all of them between the original palette and two computed grades. The night grade divides colors into lights and everything else. Lights shift toward warm white. Everything else pulls its hue halfway toward 225° and drops to 42% lightness. The dawn grade blends 55% with a gradient map keyed on luminance (violet → rose → peach). Under `prefers-reduced-motion` the file shows the finished painting with no sketch or glow.

## Timeline (s)

| 0.3–3.2 | 2–17 | 17–22 | 22–29 | 29–34.5 | 34.5–37 | 38.5–45 | 45.2–47.2 |
|---|---|---|---|---|---|---|---|
| sketch draws on | paint lands | painting as-is | night, stars glow | dawn | back to day | paint falls | sketch erases |

## Rendering cost

A frame rate is only meaningful against a baseline on the same machine. These are `requestAnimationFrame` rates in headless Chromium with SwiftShader (software raster, 4 CPUs) at four points in the loop. Every frame repaints the whole SVG, so on real hardware all three files should run faster. Their relative order is the useful result.

| file | t=8 (painting) | t=20 (hold) | t=26 (night glow) | t=43 (falling) |
|---|---|---|---|---|
| svjeez `minecraft.svg` | 24 | — | 18 | 18 |
| one animation per shape (4,751 animated paths, 205 keyframes, 868 KB) | 7 | — | 1.3 | 1.0 |
| **this: one animated group per 0.2 s slot, merged by color (1,748 paths, 114 keyframes)** | **49** | **44** | **20** | **55** |

The per-shape version was slow because of the element count. Every element with a running animation, even one only holding a value, is restyled each frame. Grouping by time slot cut the animated elements from 4,751 to 76, and merging by color within a slot cut the paths to 1,748. Reordering shapes to group them changes stacking order. That doesn't show here, because a k-means trace is close to a partition of the canvas and holes are already cut with `evenodd`. The other big cost was the blur. Blurring each twinkle group inside an un-animated `<g filter>` and animating opacity on an outer group took night from 15 to 20–24 fps.

Tying each shape's fall time to its paint time took the keyframe count from 1,835 to 205. Independent start and fall times gave nearly every shape its own (start, fall) pair, and interning only pays off when pairs repeat.

## Compared with svjeez minecraft.svg

| | minecraft.svg | starry-timeline.svg |
|---|---|---|
| input | 839 lines of hand-written scene code | one JPEG + 295 lines of scene code |
| size raw / gz | 328 KB / 50 KB | 849 KB / 223 KB |
| keyframes | 911 | 114 |
| geometry | 1,102 `<use>` of textured voxel faces | 1,748 traced paths + 124 sketch lines |
| engine | 2,089 lines TS + bun | `timeline.py` 99 lines + image-to-svg |

Our file is 4.5× larger gzipped, and nearly all of that is traced path data, which repeats less than voxel faces do. The 114 keyframes and their classes are 48 KB of CSS.

## Prior art

In 2 searches I didn't find a tool that turns a raster image into a choreographed CSS-keyframes SVG. The components are common: `stroke-dashoffset` line drawing ([Jake Archibald, 2013](https://jakearchibald.com/2013/animated-line-drawing-svg/)), staggered reveals with per-group delays, and visual keyframe editors such as CSVG. What's new here is the composition: trace, derive an underdrawing, order by size and position, intern the timeline, and grade color through `@property`.

## Limits

- Needs `@property` for the color grades to interpolate (Chrome 85+, Safari 16.4+, Firefox 128+). Without it, the palettes switch at the midpoint.
- Performance on a real GPU browser has not been measured. The table above is software raster only.
- The sky's swirls don't move. Moving them would mean either per-region transforms, which the frame-rate result above argues against, or a displacement filter on the whole sky, which this rendering setup could not afford.
- The night/dawn grades are global per-color maps, so they can't localize light, for example warming only the horizon.
