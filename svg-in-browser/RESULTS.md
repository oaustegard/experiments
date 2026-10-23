# svg-in-browser

[`web/index.html`](web/index.html) runs claude-skills `image-to-svg` in the page: drop an image in and get an SVG back without uploading anything. Served at austegard.com/experiments/svg-in-browser/web/.

## Answer

Can image-to-svg run in the page? Yes. k-means runs in WGSL on WebGPU (or in plain JS), and cleanup, tracing and output run in JS. On Starry Night at 1000 px with K=56 it writes 903 KB (Python skill: 1,355 KB), with error against the source of 24.4/255 (Python: 25.9). In JS it takes 3.2 s on this container, 2.6 s of it k-means.

The output has no holes. Every boundary between two colors is simplified once, and both regions use that same line, so no background shows between neighbors. On a dog photo, 0.000% of pixels are uncovered even with the gap-cover stroke turned off. Before this fix the figure was 0.81% (1.68% without the stroke), and the Python skill's output has 1.98%.

Real-GPU k-means time is unmeasured. This container's only WebGPU is SwiftShader, a software rasterizer. Opening the page on a Mac would measure it.

## Findings

1. **Simplifying each boundary once makes holes impossible.** On the dog photo, uncovered pixels drop from 0.811% to 0.000%, and from 1.678% to 0.000% with no stroke. The file also shrinks from 552 to 506 KB (`results/tears.txt`).
2. **The holes came from simplifying each region's loop on its own.** Each region was simplified with a tolerance of 0.2% of its length, which reached tens of pixels on long fur boundaries. The two sides of a shared edge then disagreed, leaving long thin slivers.
3. **Browser output is smaller than the Python skill's and slightly more faithful.** Starry Night: 903 KB against 1,355 KB, error 24.44 against 25.85 per 255. The browser writes one evenodd path per color (56), against one path per shape (6,344). Measured on one painting and one photo only (`results/starry-1000.txt`, `results/tears.txt`).
4. **The Python skill has the same kind of hole, with two causes.** It simplifies each shape's contour independently, and filtering leaves holes: shapes dropped by `min_area` or dark-shape gating leave pixels nothing covers. 1.98% uncovered on the dog photo.
5. **k-means is most of the CPU time.** 2.6 of 3.2 s. On SwiftShader WebGPU it takes 14 s, which says nothing about a real GPU (METHODS.md, SwiftShader).

## Method

`web/trace.js`, about 490 lines, no dependencies:

1. **Blur**: a 5-tap binomial blur, standing in for the pipeline's bilateral and Gaussian filters.
2. **k-means**: k-means++ seeding on 20k sampled pixels, then up to 20 Lloyd iterations. On WebGPU each iteration is one WGSL dispatch that assigns every pixel and accumulates per-cluster sums in workgroup memory, then K × 16 bytes read back for the center update. Otherwise the same loop runs in JS.
3. **Cleanup**: a 3×3 majority filter, then 4-connected components under 40 px absorbed into their most common neighbor. Nothing is deleted, so every pixel keeps a label.
4. **Tracing**: every label's boundary is followed along pixel edges with the region on the right, turning right at saddles (4-connectivity). Holes come out as loops of the opposite orientation, so one `fill-rule="evenodd"` path per color reproduces the region exactly.
5. **Shared-boundary simplification**: a junction is a vertex where three or more regions meet (the image edge counts as one), or a diagonal saddle. Whether a vertex is a junction depends only on its four pixels, so both regions along a boundary agree on it. Each loop is cut at its junctions into arcs. Each arc is simplified with Douglas–Peucker at fixed endpoints, with ε = 1 source pixel by default (`eps` option), through a cache keyed by the arc's first edge. The region on the other side walks the same arc backwards and gets the cached points reversed. A loop with no junctions anchors at its smallest vertex id, and a closed arc splits at its farthest vertex (ties to the smaller id), which both sides compute the same way.
6. **Output**: image-to-svg 1.9.0's compact format. A scoped class per color, relative `m`/`l` data on a 0.1-unit grid, the stroke=fill gap cover set once on the group, colors drawn largest area first.

Checks: `node run/check.js <out> [K] [side] [image]` serves `web/`, runs both backends headless and writes SVGs and stats. `python3 run/holes.py <svg>...` repaints the background magenta and reports the uncovered share. Error vs source is the mean absolute difference per channel after rendering with rsvg-convert at the SVG's size, with the source resized by INTER_AREA.

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
