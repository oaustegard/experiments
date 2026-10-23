# svg-in-browser

[`web/index.html`](web/index.html) runs claude-skills `image-to-svg` in the page: drop an image in and get an SVG back without uploading anything. Served at austegard.com/experiments/svg-in-browser/web/.

## Result

On Starry Night at 1000×625 with K=56 (the skill's painting mode), the browser version writes a 991 KB SVG (235 KB gzipped). The Python skill writes 1,355 KB (294 KB gzipped). Rendered and compared with the source, the browser output is slightly closer: mean error 24.5/255 against 25.9. That is one image, and the two pipelines differ in more than speed (below).

Timing on this container's CPU in headless Chromium: 3.3 s total, of which k-means is 2.8 s and blur, cleanup, tracing and output together are 0.6 s. The WebGPU path took 14.5 s for k-means. That number is meaningless: the only WebGPU adapter in this container is SwiftShader, a software rasterizer (METHODS.md, "SwiftShader timings can reverse a GPU conclusion"). How fast the WGSL k-means is on a real GPU is still unmeasured; opening the page on a Mac reports it in the stats box. The two backends agree: 56 paths each, sizes within 0.2%, error equal to two decimals. They are not identical, because the GPU sums in integers and the JS loop in floats.

## How it works

`web/trace.js`, about 370 lines, no dependencies:

1. **Blur**: a 5-tap binomial blur, standing in for the pipeline's bilateral and Gaussian filters.
2. **k-means**: k-means++ seeding on 20k sampled pixels, then up to 20 Lloyd iterations. On WebGPU each iteration is one WGSL dispatch that assigns every pixel and accumulates per-cluster sums in workgroup memory, then 896 bytes (K × 16) read back for the center update. Otherwise the same loop runs in JS.
3. **Cleanup**: a 3×3 majority filter, then 4-connected components under 40 px absorbed into their most common neighbor (the skill's `min_area`).
4. **Tracing**: every label's boundary is followed along pixel edges with the region on the right, turning right at saddles (4-connectivity). Holes come out as loops of the opposite orientation, so one `fill-rule="evenodd"` path per color reproduces the region exactly. No contour hierarchy is needed, and no shape can overlap another of its own color.
5. **Simplification and output**: Douglas–Peucker per loop (ε = max(0.75 px, 0.2% of the loop)), then image-to-svg 1.9.0's compact format: a scoped class per color, relative `m`/`l` data on a 0.1-unit grid, and the stroke=fill gap cover set once on the group. Colors are drawn largest area first.

## Differences from the Python skill

- One path per color (56) instead of one per shape (6,344 in Python's output). This is the main reason for the smaller file, alongside exact tiling, which leaves no overlapping contours to store.
- No dark-shape gating (compactness and edge density) and no isolation filter. Thin dark brushstrokes the Python version drops survive here as hairlines, visible in the moon.
- No ImageMagick `smooth` presets, palette remapping, compositional line pass or background detection. Palette remapping and background detection are short ports. The line pass needs Hough lines and skeletonization.

## What could also move to WGSL

The majority filter is per-pixel and maps to WGSL directly. Connected components and boundary following are sequential walks: GPU versions exist (label propagation, list ranking), but they total 0.25 s in JS, so they would not pay. On a real GPU the remaining cost is the 0.6 s of JS.

## Files

- `web/index.html`: the page. Shows the source and SVG side by side, with a download link and stage timings.
- `web/trace.js`: the module, `imageToSvg(imageData, {K, width, precision, minArea, iters, device, seed})`.
- `run/check.js`: headless check. Serves `web/`, runs both backends, writes the SVGs and stats (`node run/check.js <out> [K] [side]`).
- `results/starry-1000.txt`: the numbers above.
