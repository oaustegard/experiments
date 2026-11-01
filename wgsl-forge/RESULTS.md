# wgsl-forge: results

The question: can we compile shapes, described in Python or as SVG, into a WGSL shader, and
can we build a WGSL optimizer that shrinks shaders as far as possible without changing what
they compute?

## Answer

Yes, both. The compiler turns a scene of shapes (circles, rects, ellipses, curved paths,
gradients, transforms, and from Python also smooth unions and time-animated values) into one
fragment shader that computes every shape's distance at each pixel.
On 26 SVGs, 22 of them real icons, its images reach a median PSNR of 34.7 dB against Chromium's own SVG rendering, <!-- @evidence[[results/diff_svg.json#summary.median_psnr]] -->
a root-mean-square error of about 4.7 of 255 levels, almost all on shape edges.

The minifier cuts 86 shaders (55 naga compiler tests, 31 WebGPU samples) to 34% of their bytes. <!-- @evidence[[results/minify_bench.json#ratios.full.raw]] -->
Every output passes naga, the WGSL validator in wgpu and Firefox.
Of 37 compute entry points runnable with buffers alone, 30 gave bit-identical buffers after minification; <!-- @evidence[[results/diff_compute.json#counts.identical]] -->
the other 7 do not compile in Chrome even unminified.
On the 49 shaders LucentFlux's wgsl-minifier also handles, ours leaves 39% of the bytes <!-- @evidence[[results/minify_bench.json#baseline.ours_ratio]] -->
and wgsl-minifier leaves 56%. <!-- @evidence[[results/minify_bench.json#baseline.baseline_ratio]] -->

A scene with a path compiles to at least 2000 bytes, <!-- @evidence[[results/overhead.json#one_line_path_fill.min]] -->
mostly shared distance and winding code, so a small icon's shader outweighs its SVG.
Everything ran on SwiftShader, Chromium's CPU stand-in for a GPU; per-pixel time on real
GPUs, where each pixel loops over every path segment, is the main unmeasured cost.

## Findings

1. **Minifier output size.** The full minifier leaves 34% of the corpus's raw bytes <!-- @evidence[[results/minify_bench.json#ratios.full.raw]] -->
   and 50% of its gzip bytes; <!-- @evidence[[results/minify_bench.json#ratios.full.gz]] -->
   removing whitespace and comments alone leaves 65%. <!-- @evidence[[results/minify_bench.json#ratios.whitespace_only.raw]] -->
   The WebGPU samples on their own come to 33% raw <!-- @evidence[[results/minify_bench.json#by_source.samples.ratio]] -->
   and 50% gzip, <!-- @evidence[[results/minify_bench.json#by_source.samples.gz_ratio]] -->
   so the naga test files do not inflate the headline. All 86 outputs pass naga 30.0.1
   (`results/minify_bench.json`, `full.invalid` is empty).
2. **What was executed, and what was only compiled.** 12 corpus files use textures or 64-bit integers and were not executed. <!-- @evidence[[results/diff_compute.json#counts.files_skipped]] -->
   The rest hold 37 compute entry points, <!-- @evidence[[results/diff_compute.json#counts.entry_points]] -->
   and 30 of them returned bit-identical buffers on random inputs; <!-- @evidence[[results/diff_compute.json#counts.identical]] -->
   the other 7 do not compile in Chrome unminified. <!-- @evidence[[results/diff_compute.json#counts.error]] -->
   Every minified scene shader drew the same pixels as its source in all 26 SVG renders and
   all 6 Python frames (`results/diff_svg.json`, `results/diff_python.json`). The corpus's
   vertex and fragment shaders were validated and compiled, not executed.
3. **Compiling in Chrome.** Tint, Chrome's WGSL compiler, accepts 73 shaders both before and after minification. <!-- @evidence[[results/tint_compile.json#counts.both]] -->
   11 fail both ways, mostly on extensions Chrome does not enable here (f16, ray queries, 64-bit integers). <!-- @evidence[[results/tint_compile.json#counts.neither]] -->
   2 compile only after minification, <!-- @evidence[[results/tint_compile.json#counts.minified_only]] -->
   because dead-code removal drops an unused `const` that Tint rejects (a float literal just
   past the f32 range, and a `u32 + f32` in an unused expression). None compiles before but
   not after.
4. **Against wgsl-minifier 0.7.0.** It produced no output for 36 of the 86 files: <!-- @evidence[[results/minify_bench.json#baseline.rejected_or_crashed]] -->
   26 stopped with a parse or validation error from its embedded, older naga, <!-- @evidence[[results/minify_bench.json#baseline.failures.error]] -->
   and 10 panicked. <!-- @evidence[[results/minify_bench.json#baseline.failures.panic]] -->
   One of its outputs fails validation. On the 49 files both handle, ours leaves 39% of the
   bytes and it leaves 56% (`results/minify_bench.json`, `baseline`).
   After gzip, ours leaves 58% <!-- @evidence[[results/minify_bench.json#baseline.ours_gz_ratio]] -->
   and it leaves 73%. <!-- @evidence[[results/minify_bench.json#baseline.baseline_gz_ratio]] -->
   Ours is smaller on 45 of the 49. <!-- @evidence[[results/minify_bench.json#baseline.ours_smaller]] -->
5. **Which passes matter.** On a total output of 63114 bytes, <!-- @evidence[[results/minify_bench.json#full.raw]] -->
   switching off renaming adds 37389 bytes. <!-- @evidence[[results/minify_bench.json#ablation.rename.raw_cost]] -->
   Switching off dead-code removal adds 10569, <!-- @evidence[[results/minify_bench.json#ablation.dce.raw_cost]] -->
   much of it naga test code that no entry point reaches.
   Writing `vec3<f32>` as `vec3f` saves 2333. <!-- @evidence[[results/minify_bench.json#ablation.aliases.raw_cost]] -->
   Type aliases (`alias a=vec4f;`) save 1075 raw bytes <!-- @evidence[[results/minify_bench.json#ablation.type_aliases.raw_cost]] -->
   but add 275 gzip bytes, so `--no-aliases` suits shaders served compressed (`tests/bench_minify.py`).
6. **SVG fidelity.** Against Chromium at 192 px wide, the worst image is off by 1.143 levels per channel on average. <!-- @evidence[[results/diff_svg.json#summary.max_mae]] -->
   At most 1.711% of an image's pixels differ by more than 32 levels, <!-- @evidence[[results/diff_svg.json#summary.max_bad_px]] -->
   all on shape edges (`results/svg/*.png`, where the third panel is the difference magnified 4×).
7. **Line segments against quadratic curves.** Ink is how much darker than white an image is, summed over pixels; the ratio divides the shader's ink by Chromium's.
   With curves flattened to line segments at the SVG's own size, the Feather settings icon had an ink ratio of 0.9824 at 512 px (Round 1). <!-- @evidence[[results/round1_lines.json#ink_ratio_512px.fe_settings]] -->
   With each cubic split into quadratic Béziers, the same icon at 512 px has 1.0022 (Round 2). <!-- @evidence[[results/scale_check.json#fe_settings.512.ink_ratio]] -->
   Round 1's median PSNR was 28.93 dB. <!-- @evidence[[results/round1_lines.json#diff_svg_192px.median_psnr]] -->
   In Round 2 the error falls with resolution, from 4.577 levels at 48 px <!-- @evidence[[results/scale_check.json#fe_settings.48.mae]] -->
   to 0.377 at 768 px, <!-- @evidence[[results/scale_check.json#fe_settings.768.mae]] -->
   which is what antialiasing error does and geometry error does not. At 48 px the shader
   draws slightly heavy (`scale_check.json`, ink ratio 1.03), from its one-pixel coverage ramp.
8. **Animation.** A Python scene driven by `T` matches its own SVG export, evaluated in Python at the same time, at 47.05 dB for t=1.3. <!-- @evidence[[results/diff_python.json#/sunrise/1.3/psnr]] -->
   While an operator-precedence bug was live, frames at t=0 and t=4.7 still matched closely,
   because the animated term was near zero at both, and t=1.3 visibly diverged
   ([ERRORS.md](ERRORS.md) #2). The SDF-only operations (smooth union, subtract, intersect,
   `rounded`, `onion`, `soft`) have no SVG equivalent and so no reference image; they were
   checked by eye (`results/python/blobs_*.png`) and for minified output drawing identically.
9. **Compiled scene shader size.** A one-circle scene minifies to 466 bytes. <!-- @evidence[[results/overhead.json#one_circle.min]] -->
   A scene with one path needs 2000, <!-- @evidence[[results/overhead.json#one_line_path_fill.min]] -->
   and 2461 once the path is also stroked. <!-- @evidence[[results/overhead.json#one_curve_path_fill_and_stroke.min]] -->
   The minified scene shaders are a median 4.9 times the bytes of their SVGs. <!-- @evidence[[results/diff_svg.json#summary.median_min_over_svg]] -->
10. **A naga quirk.** naga 30 rejects `array<e,2>`, where `e` aliases `i32`, as a different
    type from `array<i32,2>`; the alias pass leaves template arguments alone for that reason
    ([ERRORS.md](ERRORS.md) #5).

## Method

**Corpus.** 55 WGSL files from naga's test inputs (`gfx-rs/wgpu`, `naga/tests/in/wgsl`) and 31
from `webgpu/webgpu-samples`, with the three Cornell-box shaders concatenated with their
`common.wgsl` and one template placeholder filled in (`corpus/README.md`). All 86 pass naga.

**Checks.** `naga` (30.0.1, `cargo install naga-cli`) validates every minifier output. Chrome's
Tint compiles originals and outputs (`tests/tint_compile.py`). `harness/gpu.mjs` runs jobs in
headless Chromium on SwiftShader WebGPU: compute dispatches, full-screen renders, and
Chromium's own SVG rasteriser through a canvas. `tests/diff_compute.py` fills every buffer
binding with typed random values laid out from the module's own types, sets any override that
has no default to 1, runs each compute entry point twice unminified (to rule out races) and
once minified, and compares bytes. `tests/diff_svg.py` and `tests/diff_python.py` composite
both images on white and report the mean absolute error per channel (0–255), the share of
pixels off by more than 32 in their worst channel, and PSNR; no pass threshold is set.
`tests/scale_check.py` renders four SVGs at 48–768 px and reports the ink ratio.
`tests/bench_minify.py` measures raw and gzip-9 bytes per pass and runs the baseline,
LucentFlux `wgsl-minifier` 0.7.0 (`cargo install wgsl-minifier --features bin`), logging why
each file failed. `tests/overhead.py` compiles the smallest scenes of each kind.

**Curves.** Each cubic is split into n quadratics, with n chosen so the quadratic stays within
0.02 nominal pixels of the cubic (the error bound falls as 1/n³); quadratics are then split at
their highest or lowest point so each rises or falls monotonically. Per pixel, the shader
solves the cubic equation for the closest point on each quadratic, so distances are exact to
the quadratics, which are within that tolerance of the original cubics.

**Controls.** Unminified against minified isolates the minifier. Chromium's SVG renderer is
the reference for the compiler; it uses Skia's coverage antialiasing, while the shader uses a
one-pixel ramp on the signed distance, so edge pixels differ by design.

**Cost.** Everything ran on CPU. Minifying all 86 files takes 0.69 s <!-- @evidence[[results/minify_bench.json#minify_only_seconds]] -->
(1.23 s with naga validating each output); the whole benchmark with every ablation takes
about 15 s. Each GPU suite takes 6–15 s on SwiftShader.

## Log

### Round 1 — 2026-11-01: parser, minifier, line-segment paths

Oskar asked for a WGSL compiler from Python or SVG input and a size-first WGSL optimizer.
Asked whether "Python rendering" meant a per-pixel shading function or drawing calls, he
chose input that describes shapes rather than pixels. Built the lexer (with the spec's
template-list discovery), parser, printer and scope resolver; all 86 corpus files round-trip.
The minifier's first version reached 34.6% raw / 50.2% gzip with every output naga-valid on
the first run, and 29 compute entry points bit-identical (30 once the harness set override values, Round 2). The shape compiler first
flattened every curve to line segments within 0.2 nominal pixels: 26 SVGs at 192 px had a
median PSNR of 28.93 dB and worst MAE 4.85. The ink ratio at 512 px stayed at 0.982–0.984 for
curvy icons, a geometric error that resolution would never fix (`results/round1_lines.json`).

### Round 2 — 2026-11-01: quadratic segments, more passes, Python animation

Cubics now become y-monotone quadratic Béziers (error falls as n⁻³). The shader finds exact
distance by solving the closest-point cubic and the winding number by solving the quadratic
at the pixel's y. Median PSNR rose to 34.74 dB; MAE now falls with size (4.58 → 0.38 from 48
to 768 px). Single-use `let` inlining and splat constructors took the minifier from 34.6% to 34.1% raw
(50.2% → 49.7% gzip); type aliases then took raw to 33.6% and gzip back up to 50.1%. The alias pass first aliased the generic constructors `vec2`
and `array` and broke 9 outputs, which naga caught; it then hit naga's alias-in-template quirk.
The Python front-end gained `T`-driven animation; one bug (a missing parenthesis around an
animated radius) passed at t=0 and t=4.7 and failed at t=1.3. Gallery page built
(`web/`), all 8 shaders compile and draw in headless Chromium. A fresh reader's review led to
measuring what the first writeup asserted: the per-source split, the baseline's failure
reasons, the fixed shader overhead, override values in the compute harness, and the 512 px
point for comparing the two rounds at one size.
