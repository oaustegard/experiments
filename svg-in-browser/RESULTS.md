# svg-in-browser

[`web/index.html`](web/index.html) runs claude-skills `image-to-svg` in the page: drop an image in and get an SVG back without uploading anything. Served at austegard.com/experiments/svg-in-browser/web/.

## Answer

Can image-to-svg run in the page? Yes. k-means runs in WGSL on WebGPU (or in JS), and cleanup, tracing and output run in JS. Starry Night at 1000 px with K=56: 903 KB against the Python skill's 1,355 KB, error vs source 24.4/255 against 25.9. In JS it takes 3.2 s on this container, 2.6 s of it k-means.

No pixel is left uncovered. Each boundary between two colors is simplified once and shared by both regions: 0.000% uncovered on a dog photo even with the gap stroke off, against 0.81% before this fix and 1.98% for the Python skill.

The coarse-to-fine mode builds the picture the way a diffusion sampler sharpens one. It paints smooth blobs from a 1/8-scale image, then adds regions at 1/4, 1/2 and full scale where they lower the error. Starry Night ends at 634 KB / 23.97, better than flat on both counts. The dog ends at 379 KB / 7.79, smaller but slightly worse than flat's 506 KB / 7.16. The first layer alone is a 14–19 KB sketch.

Unmeasured: real-GPU k-means time (SwiftShader only here) and a sweep of the layer gain threshold.

## Findings

1. **Coarse-to-fine layers converge like a sampler, and the first layer is a usable preview.** Error after each cumulative layer: dog 13.85 → 12.07 → 9.34 → 7.79, Starry 33.41 → 31.11 → 26.99 → 23.97. Sizes: dog 14 → 30 → 116 → 379 KB, Starry 19 → 39 → 152 → 634 KB. The finer levels draw only 28–42% of their regions (`results/layers.txt`).
2. **Whether layers beat flat depends on the image.** Starry Night: smaller and more faithful (634 KB / 23.97 against 903 KB / 24.44). Dog: 25% smaller but slightly worse (379 KB / 7.79 against 506 KB / 7.16). The `gain` threshold (default 4/255) sets the trade-off and has not been swept (`results/layers.txt`).
3. **Simplifying each boundary once makes holes impossible.** On the dog photo, uncovered pixels drop from 0.811% to 0.000%, and from 1.678% to 0.000% with no stroke. The file also shrinks from 552 to 506 KB (`results/tears.txt`).
4. **The holes came from simplifying each region's loop on its own.** Each region was simplified with a tolerance of 0.2% of its length, which reached tens of pixels on long fur boundaries. The two sides of a shared edge then disagreed, leaving long thin slivers.
5. **Browser output is smaller than the Python skill's and slightly more faithful.** Starry Night: 903 KB against 1,355 KB, error 24.44 against 25.85 per 255. The browser writes one evenodd path per color (56), against one path per shape (6,344). Measured on one painting and one photo only (`results/starry-1000.txt`, `results/tears.txt`).
6. **The Python skill has the same kind of hole, with two causes.** It simplifies each shape's contour independently, and filtering leaves holes: shapes dropped by `min_area` or dark-shape gating leave pixels nothing covers. 1.98% uncovered on the dog photo.
7. **k-means is most of the CPU time.** 2.6 of 3.2 s. On SwiftShader WebGPU it takes 14 s, which says nothing about a real GPU (METHODS.md, SwiftShader).

## Method

`web/trace.js`, about 680 lines, no dependencies:

1. **Blur**: a 5-tap binomial blur, standing in for the pipeline's bilateral and Gaussian filters.
2. **k-means**: k-means++ seeding on 20k sampled pixels, then up to 20 Lloyd iterations. On WebGPU each iteration is one WGSL dispatch that assigns every pixel and accumulates per-cluster sums in workgroup memory, then K × 16 bytes read back for the center update. Otherwise the same loop runs in JS.
3. **Cleanup**: a 3×3 majority filter, then 4-connected components under 40 px absorbed into their most common neighbor. Nothing is deleted, so every pixel keeps a label.
4. **Tracing**: every label's boundary is followed along pixel edges with the region on the right, turning right at saddles (4-connectivity). Holes come out as loops of the opposite orientation, so one `fill-rule="evenodd"` path per color reproduces the region exactly.
5. **Shared-boundary simplification**: a junction is a vertex where three or more regions meet (the image edge counts as one), or a diagonal saddle. Whether a vertex is a junction depends only on its four pixels, so both regions along a boundary agree on it. Each loop is cut at its junctions into arcs. Each arc is simplified with Douglas–Peucker at fixed endpoints, with ε = 1 source pixel by default (`eps` option), through a cache keyed by the arc's first edge. The region on the other side walks the same arc backwards and gets the cached points reversed. A loop with no junctions anchors at its smallest vertex id, and a closed arc splits at its farthest vertex (ties to the smaller id), which both sides compute the same way.
6. **Output**: image-to-svg 1.9.0's compact format. A scoped class per color, relative `m`/`l` data on a 0.1-unit grid, the stroke=fill gap cover set once on the group, colors drawn largest area first.

Coarse-to-fine (`imageToSvgLayers`): each level box-downsamples the source (1/8, 1/4, 1/2, 1 with K = 8, 16, 32, 56), then quantizes, cleans and finds 4-connected regions as above. Each region is scored at full resolution: mean absolute error per channel against what earlier levels painted, versus against the region's own color. A region is drawn only when it improves the error by more than `gain` (default 4/255). Level 0 has nothing beneath it, so it draws everything and tiles the image. Regions not drawn become an empty label the tracer skips, and drawn regions are traced and simplified with shared arcs as above. Paths are smoothed into quadratic Béziers through edge midpoints, with junctions and the image corners kept as sharp corners. A quadratic traced backwards is the same curve, so the two sides of a smoothed arc still coincide. Each level is one `<g class="L{n}">`. `reveal` (seconds per layer) animates them in order. `revealMode: "fade"` fades each layer in. `"paint"` masks each layer with one back-and-forth brush stroke drawn on by animating `stroke-dashoffset`: 3 passes for the blobs, then 5, 8 and 13, each 1.3 bands wide so the passes overlap. `pathLength="1"` keeps the dash one unit long whatever the stroke's length, and a renderer without CSS animation shows the stroke fully drawn, so the image stays whole. The page downloads `.svgz` (the SVG gzipped by the browser's `CompressionStream`) or plain `.svg`. x and y scale separately so a rounded level grid still spans the viewBox.

Checks: `node run/check.js <out> [K] [side] [image] [flat|layers]` serves `web/`, runs both backends headless and writes SVGs and stats. `python3 run/holes.py <svg>...` repaints the background magenta and reports the uncovered share. Error vs source is the mean absolute difference per channel after rendering with rsvg-convert at the SVG's size, with the source resized by INTER_AREA.

Still missing compared with the Python skill: dark-shape gating, ImageMagick `smooth` presets, palette remapping, background detection and the compositional line pass. The majority filter would map to WGSL directly. Connected components and boundary following are sequential and take about 0.2 s in JS, so moving them to the GPU would not pay.

## Log

### 2026-10-23: prototype

Asked: "could we also make this a JavaScript (or better yet wgsl) compiler and run it in page?" Built `trace.js` with per-region Douglas–Peucker (ε = max(0.75, 0.2% of the loop)). Starry Night 1000 px K=56: 991 KB (235 KB gz) against Python 1,355 KB (294 KB gz). Error 24.53 against 25.85. WebGPU and JS agree: 56 paths each, sizes within 0.2%. JS 3.3 s, k-means 2.8 s, SwiftShader WebGPU 14.5 s.

### 2026-10-23: tears

Oskar sent a render with pale slivers where no shape was drawn. Measured by repainting the background magenta:

| output | size | uncovered | uncovered, no stroke |
|---|---|---|---|
| dog, per-region DP | 552 KB | 0.811% | 1.678% |
| dog, shared arcs | 506 KB | 0.000% | 0.000% |
| dog, Python skill 1.9.0 | 1,142 KB | 1.980% | — |
| starry, shared arcs | 903 KB | 0.000% | 0.000% |

Replaced per-region simplification with shared arcs (Method step 5). Starry Night error went from 24.53 to 24.44, and the dog's is 7.16. The Python skill's holes have a second cause the browser version never had: shapes the filters drop leave pixels uncovered. Porting the shared-arc tracer into the skill would fix both.

### 2026-10-23: coarse to fine

Asked: "the creation approach could still be more like diffusion, start with blobby outlines and then add progressively finer details on top." Built `imageToSvgLayers` (Method). The first run leaked 0.67% on the dog, all at the image edges. Two causes: the 1/8 grid has 98 rows, which times 8 is 784 against a 785-unit viewBox, and image corners were not junctions, so the curves cut them. Fixed with separate x/y scales and pinned corners, after which nothing is uncovered, with or without the stroke.

| image | L0 | L0–1 | L0–2 | L0–3 | flat (shared arcs) |
|---|---|---|---|---|---|
| dog, KB / error | 14 / 13.85 | 30 / 12.07 | 116 / 9.34 | 379 / 7.79 | 506 / 7.16 |
| starry, KB / error | 19 / 33.41 | 39 / 31.11 | 152 / 26.99 | 634 / 23.97 | 903 / 24.44 |

`web/dog-reveal.svg` is the dog with `reveal: 0.8`, a 3.2 s build-up from blobs to fur. The `gain` threshold and the level schedule are untuned.

### 2026-10-23: svgz and painting

Asked: download `.svgz` instead of `.svg`, and animate the "painting" of each layer. The page now offers both downloads. Its default layered and painted dog is 120 KB as `.svgz` against 409 KB as `.svg`, and the `.svgz` gunzips to the `.svg` byte for byte (headless Chromium, `fetch` on the download link). `web/dog-paint.svg` is that file: 1.6 s per layer, 6.4 s in all. Frames at 0.8, 2.4, 4.0, 5.6 and 7.0 s show the brush mid-pass on the blobs, then each finer layer swept over the last. A `.svgz` opens in browsers and editors from disk, but a web server has to send it with `Content-Encoding: gzip`, so pages here keep shipping `.svg` and let the server compress it. Painting masks thousands of paths every frame, and how smooth that is on real hardware is unmeasured.
