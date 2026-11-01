# wgsl-forge: results

The question: can we compile shapes, described in Python or as SVG, into a WGSL shader, and
can we build a WGSL optimizer that shrinks shaders as far as possible without changing what
they compute?

## Answer

Yes, both. The compiler turns a shape scene (circles, rects, ellipses, paths with curves,
gradients, transforms, and in Python also smooth unions, subtraction and time-animated
values) into one fragment shader that computes every shape's distance at each pixel.
On 26 SVGs, 22 of them real icons, its images reach a median PSNR of 34.7 dB against Chromium's own SVG rendering. <!-- @evidence[[results/diff_svg.json#summary.median_psnr]] -->
The remaining error is edge antialiasing, which shrinks as resolution grows.

The minifier cuts 86 real shaders (naga's WGSL test inputs and the WebGPU samples) to 34% of their bytes. <!-- @evidence[[results/minify_bench.json#ratios.full.raw]] -->
After gzip they are 50% of the original gzip size. <!-- @evidence[[results/minify_bench.json#ratios.full.gz]] -->
Every output validates in naga, and all 29 runnable compute entry points produced bit-identical buffers. <!-- @evidence[[results/diff_compute.json#counts.identical]] -->
On the 49 shaders that LucentFlux's wgsl-minifier also handles, ours leaves 39% of the bytes <!-- @evidence[[results/minify_bench.json#baseline.ours_ratio]] -->
and wgsl-minifier leaves 56%. <!-- @evidence[[results/minify_bench.json#baseline.baseline_ratio]] -->

The compiled shaders are bigger than their SVGs: the minified output is a median 4.9 times the SVG's bytes, <!-- @evidence[[results/diff_svg.json#summary.median_min_over_svg]] -->
because about 2 KB of distance and winding code ships with every scene. Everything ran on
SwiftShader, so per-pixel cost on real phones and laptops is unmeasured; that test would
change the picture most.

## Findings

1. **Size.** The full minifier leaves 34% of the corpus's raw bytes <!-- @evidence[[results/minify_bench.json#ratios.full.raw]] -->
   and 50% of its gzip bytes; <!-- @evidence[[results/minify_bench.json#ratios.full.gz]] -->
   whitespace and comment removal alone leaves 65%. <!-- @evidence[[results/minify_bench.json#ratios.whitespace_only.raw]] -->
   All 86 outputs pass naga 30.0.1 (`results/minify_bench.json`, `full.invalid` is empty).
2. **Behaviour preserved.** 29 of 29 runnable compute entry points return bit-identical buffers on random inputs. <!-- @evidence[[results/diff_compute.json#counts.identical]] -->
   The other 8 do not compile in Chromium even unminified.
   73 shaders compile in Tint both before and after minification, <!-- @evidence[[results/tint_compile.json#counts.both]] -->
   and none compiles before but not after. Every minified scene shader draws the same pixels
   as its source, in all 26 SVG renders and all 6 Python frames (`results/diff_svg.json`,
   `results/diff_python.json`).
3. **Against wgsl-minifier 0.7.0.** It rejected or crashed on 36 of the 86 files, <!-- @evidence[[results/minify_bench.json#baseline.rejected_or_crashed]] -->
   since it embeds an older naga, and one of its outputs fails validation.
   On the 49 files both handle, ours leaves 39% of the bytes and it leaves 56% (`results/minify_bench.json`, `baseline`).
   After gzip, ours leaves 58% <!-- @evidence[[results/minify_bench.json#baseline.ours_gz_ratio]] -->
   and it leaves 73%. <!-- @evidence[[results/minify_bench.json#baseline.baseline_gz_ratio]] -->
   Ours is smaller on 45 of the 49. <!-- @evidence[[results/minify_bench.json#baseline.ours_smaller]] -->
4. **Which passes matter.** Switching off renaming costs 37389 bytes. <!-- @evidence[[results/minify_bench.json#ablation.rename.raw_cost]] -->
   Switching off dead-code removal costs 10569, <!-- @evidence[[results/minify_bench.json#ablation.dce.raw_cost]] -->
   high for this corpus because naga's tests carry much unreachable code.
   Writing `vec3<f32>` as `vec3f` saves 2333. <!-- @evidence[[results/minify_bench.json#ablation.aliases.raw_cost]] -->
   Type aliases (`alias a=vec4f;`) save 1075 raw bytes <!-- @evidence[[results/minify_bench.json#ablation.type_aliases.raw_cost]] -->
   but add 275 gzip bytes, so `--no-aliases` suits shaders served compressed (`tests/bench_minify.py`).
5. **SVG fidelity.** Against Chromium at 192 px wide, the worst image is off by 1.143 levels per channel on average. <!-- @evidence[[results/diff_svg.json#summary.max_mae]] -->
   At most 1.711% of an image's pixels differ by more than 32 levels, <!-- @evidence[[results/diff_svg.json#summary.max_bad_px]] -->
   all of them on shape edges (`results/svg/*.png`, third panel ×4).
6. **Curves have to stay curves.** Flattening paths to line segments at the SVG's own size gave an ink ratio of 0.9824 at 512 px (Round 1). <!-- @evidence[[results/round1_lines.json#ink_ratio_512px.fe_settings]] -->
   That version's median PSNR was 28.93 dB. <!-- @evidence[[results/round1_lines.json#diff_svg_192px.median_psnr]] -->
   With exact distance to quadratic Béziers (Round 2), the error falls with resolution, from 4.577 levels at 48 px <!-- @evidence[[results/scale_check.json#fe_settings.48.mae]] -->
   to 0.377 at 768 px, <!-- @evidence[[results/scale_check.json#fe_settings.768.mae]] -->
   where the ink ratio is 1.0012. <!-- @evidence[[results/scale_check.json#fe_settings.768.ink_ratio]] -->
7. **Animation.** A Python scene driven by `T` matches its own SVG export, evaluated in Python at the same time, at 47.05 dB for t=1.3. <!-- @evidence[[results/diff_python.json#/sunrise/1.3/psnr]] -->
   Frames at t=0 and t=4.7 passed while an operator-precedence bug was live, because the
   animated term was near zero at both ([ERRORS.md](ERRORS.md) #2).
8. **Output size.** The median minified scene shader is 2778 bytes. <!-- @evidence[[results/diff_svg.json#summary.median_min_bytes]] -->
   Path data is delta-encoded integers, so most of a small icon's shader is the fixed
   helper code; the compiler does not compress SVG.
9. **A naga quirk.** naga 30 rejects `array<e,2>`, where `e` aliases `i32`, as a different
   type from `array<i32,2>`; the alias pass leaves template arguments alone for that reason
   ([ERRORS.md](ERRORS.md) #5).

## Method

**Corpus.** 55 WGSL files from naga's test inputs (`gfx-rs/wgpu`, `naga/tests/in/wgsl`) and 31
from `webgpu/webgpu-samples`, with the three Cornell-box shaders concatenated with their
`common.wgsl` and one template placeholder filled in (`corpus/README.md`). All 86 pass naga.

**Checks.** `naga` (30.0.1, `cargo install naga-cli`) validates every minifier output. Chromium's
Tint compiles originals and outputs (`tests/tint_compile.py`). `harness/gpu.mjs` runs jobs
in headless Chromium on SwiftShader WebGPU: compute dispatches, full-screen renders, and
Chromium's own SVG rasteriser through a canvas. `tests/diff_compute.py` fills every buffer
binding with typed random values from the module's own layout, runs each compute entry point
twice unminified (to rule out races) and once minified, and compares bytes.
`tests/diff_svg.py` and `tests/diff_python.py` composite both images on white and report the
mean absolute error per channel (0–255), the share of pixels off by more than 32 in their worst
channel, and PSNR. `tests/scale_check.py` renders four SVGs at 48–768 px and reports
shader ink divided by reference ink. `tests/bench_minify.py` measures raw and gzip-9 bytes per
pass and runs the baseline, LucentFlux `wgsl-minifier` 0.7.0 (`cargo install wgsl-minifier
--features bin`).

**Controls.** Unminified versus minified isolates the minifier. Chromium's SVG renderer is
the reference for the compiler; it uses Skia's coverage antialiasing, while the shader uses a
one-pixel ramp on the signed distance, so edge pixels differ by design.

**Cost.** The whole experiment ran on CPU: the minifier takes 13.7 s for the 86 files plus
naga validation, and the GPU suites take 6–15 s each on SwiftShader.

## Log

### Round 1 — 2026-11-01: parser, minifier, line-segment paths

Oskar asked for a WGSL compiler from Python or SVG input and a size-first WGSL optimizer,
and on the follow-up chose shapes over pixels. Built the lexer (with the spec's
template-list discovery), parser, printer and scope resolver; all 86 corpus files round-trip.
The minifier's first version reached 34.6% raw / 50.2% gzip with every output naga-valid on
the first run, and 29/29 compute entry points bit-identical. The shape compiler first
flattened every curve to line segments within 0.2 nominal pixels: 26 SVGs at 192 px had a
median PSNR of 28.93 dB and worst MAE 4.85. The ink ratio at 512 px stayed at 0.982–0.984 for
curvy icons, a geometric error that resolution would never fix (`results/round1_lines.json`).

### Round 2 — 2026-11-01: quadratic segments, more passes, Python animation

Cubics now become y-monotone quadratic Béziers (error falls as n⁻³). The shader finds exact
distance by solving the closest-point cubic and the winding number by solving the quadratic
at the pixel's y. Median PSNR rose to 34.74 dB; MAE now falls with size (4.58 → 0.38 from 48
to 768 px). Added single-use `let` inlining, splat constructors and type aliases (34.1% →
33.6% raw, gzip 49.7% → 50.1%). The alias pass first aliased the generic constructors `vec2`
and `array` and broke 9 outputs, which naga caught; it then hit naga's alias-in-template quirk.
The Python front-end gained `T`-driven animation; one bug (a missing parenthesis around an
animated radius) passed at t=0 and t=4.7 and failed at t=1.3. Gallery page built
(`web/`), all 8 shaders compile and draw in headless Chromium.
