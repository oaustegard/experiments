# wgsl-forge

A WGSL toolchain in plain Python: a shape compiler that turns SVG files or Python-built scenes
into a single WebGPU fragment shader, and a minifier for any WGSL module. On 86 test shaders
every minified output validates, and every one that could be executed (30 compute entry
points, 32 scene renders) computes bit-identical results. Results and how they were measured: [RESULTS.md](RESULTS.md).
Live gallery: [`web/index.html`](web/index.html) ([austegard.com/experiments/wgsl-forge/web/](https://austegard.com/experiments/wgsl-forge/web/)).

```bash
python3 -m forge svg drawing.svg --minify -o drawing.wgsl     # SVG -> shader
python3 -m forge py examples/sunrise.py -o sunrise.wgsl       # Python scene -> shader
python3 -m forge minify shader.wgsl -o shader.min.wgsl        # any WGSL module
python3 -m forge fmt shader.min.wgsl                          # pretty-print
python3 -m forge html drawing.svg --minify -o preview.html    # self-contained live preview
```

Needs Python 3.10+ and numpy. The checks also use `naga` (`cargo install naga-cli`) and
Node with Playwright's Chromium for WebGPU.

## The shape compiler

Both front-ends build the same `Scene` (`forge/scene.py`): an ordered list of shapes, each a
geometry with a fill and/or stroke paint and a transform. `forge/codegen.py` emits one shader
that, for each pixel, carries the point into each shape's local coordinates, evaluates a
signed distance there, turns it into one pixel's worth of coverage and composites back to
front. Nothing is rasterised ahead of time, so edges stay sharp at any resolution.

- **Primitives** use closed-form distances: circle, ellipse, rounded rect.
- **Paths** keep their curves. Each cubic becomes n quadratic Béziers, with n chosen to stay
  within 0.02 nominal pixels (the error falls as 1/n³), split so each one only rises or only
  falls. They are stored delta-encoded as integers in one `const` array. A loop computes the
  distance to each quadratic (solving a cubic equation for the closest point), the winding
  number for `nonzero`/`evenodd` fills, and stroke distance with butt, square or round caps.
- **Paints** are solid colours and linear or radial gradients, in user or bounding-box
  units, with `gradientTransform`.
- **Transforms** that never change are folded into one matrix; a path under one is baked
  into user space, so skew and non-uniform scale stay exact.

The shader's interface is fixed: `@group(0) @binding(0) var<uniform> u: U` with
`struct U { r: vec2f, t: f32 }` (output size in pixels, time in seconds), a vertex entry
`vs` that draws a full-screen triangle (`draw(3)`), and a fragment entry `fs` that returns
premultiplied RGBA. `python3 -m forge html` shows the host code needed.

### SVG

`forge/svg.py` reads `svg`, `g`, `use`/`symbol`, `rect`, `circle`, `ellipse`, `line`,
`polyline`, `polygon` and `path` (every command, arcs included), presentation attributes,
`style=""`, simple `<style>` rules (class, id and element selectors), gradients with `href`
inheritance, `fill-rule`, the opacities, `stroke-linecap` and `currentColor`.

Some features are only approximated:

- **Group opacity** multiplies into each child. Real group opacity flattens the group into one layer first.
- **Stroke joins** are always round, because joins come from distance to the centre line.
- **A radial gradient's focal point** is treated as its centre.

It warns on and skips `text`, `image`, `clipPath`, `mask`, `filter`, `pattern` and `marker`.

### Python

```python
from forge.scene import Scene, T, sin, radial, smooth_union, Circle

s = Scene(320, 200, background="#101418")
s.circle(160, 100, 40 + 4 * sin(T * 2), fill=radial(160, 100, 50, ["#fff7d6", "#ff8a3d"]))
with s.group(translate=(160, 100), rotate=T * 30):          # animated transform
    for k in range(12):
        s.rect(-2, -80, 4, 20, rx=2, fill="#ffd36e", rotate=k * 30)
s.add(smooth_union(Circle(60, 150, 25), Circle(90 + 20 * sin(T), 150, 20), k=12), fill="hotpink")
s.path("M10 190 Q 160 120 310 190", fill=None, stroke="white", stroke_width=3, linecap="round")
wgsl = s.compile(minify=True)
svg = s.to_svg(t=1.5)       # the scene at t=1.5 as SVG; raises on SDF-only features
```

Shapes: `circle`, `ellipse`, `rect` (`rx`/`ry`), `line`, `polyline`, `polygon`, `path` (SVG path
syntax) and `add(geometry)`. Geometries can be combined with `union`/`smooth_union`,
`subtract` and `intersect` (sharp, or smooth with `k`), and changed with `rounded` and
`onion`. These SDF-only features, and `soft`, have no SVG equivalent, so `to_svg` raises on
them and no reference image checks them. Style keywords: `fill`, `stroke`, `stroke_width`, `opacity`, `fill_opacity`,
`stroke_opacity`, `fill_rule`, `linecap`, `soft` (extra edge blur, for glows). Transform
keywords on shapes and `group()`: `translate`, `rotate` (degrees, or `(deg, cx, cy)`),
`scale`, `transform` (an SVG transform string). Any number can be an expression in `T`
(seconds) built with `+ - * /`, `sin`, `cos`, `fract`, `clamp`, `mix` and `smoothstep`;
it compiles to WGSL over the time uniform and evaluates in Python for `to_svg(t)`.

## The minifier

`forge/wgsl.py` lexes WGSL, using the spec's template-list discovery to tell `a < b` from
`vec3<f32>`, and parses the full grammar. `forge/printer.py` prints the tree back with the
fewest spaces and parentheses that re-parse to the same tree. If a compact print would
create a template list that was not there, as in `f((a<b), c>d)`, it re-lexes and keeps the
parentheses. `forge/scope.py` binds every name to its declaration, and `forge/types.py`
infers types where that is cheap.

`forge/minify.py` runs these passes in order:

| pass | what it does |
|---|---|
| dce | drops module declarations no entry point reaches, and unused local `let`/`var` with side-effect-free initialisers |
| inline | replaces a literal `const` with its value where that is shorter overall |
| rewrite | `x = x + y` → `x += y`; `i += 1` → `i++` on integers; drops a trailing `return;` and empty `else {}` |
| inline_lets | inlines a `let` used once, if its initialiser is pure, concretely typed, does not move into a loop, and nothing it reads is written in between |
| splat | `vec3f(x, x, x)` → `vec3f(x)` |
| untype | drops `: T` from `let`/`var` when the initialiser's inferred type is exactly `T` |
| literals | the shortest spelling of the same value and type: `1.0`→`1.`, `0.5`→`.5`, `100000.0`→`1e5`, `1.0f`→`1f` |
| aliases | `vec3<f32>` → `vec3f` and the other predeclared aliases |
| type_aliases | `alias a = vec4f;` for concrete types used often enough to pay (`--no-aliases` turns it off; it costs gzip bytes) |
| rename | scope-aware renaming, most-referenced first: module names, then per-function locals reusing the same short names, then struct members |

Some names are never changed: entry-point names, `override` names (the host sets them by
name), binding numbers and attributes, directives, and struct members spelled like swizzles
(`x`, `rgb`) or like builtin result fields (`fract`, `exp`, `whole`, `old_value`, `exchanged`).
`--keep a,b` protects more.

## Checks

| script | checks |
|---|---|
| `tests/test_forge.py` | parser round trips, printer edge cases, literal spelling, renaming hazards, idempotence (no GPU) |
| `tests/bench_minify.py` | bytes per pass, naga validation, and the comparison with wgsl-minifier |
| `tests/tint_compile.py` | Chromium's Tint compiles each corpus shader before and after |
| `tests/diff_compute.py` | compute entry points: bit-identical buffers before and after, on random inputs |
| `tests/diff_svg.py` | SVG → shader against Chromium's SVG rendering, and minified against unminified pixels |
| `tests/diff_python.py` | an animated Python scene against its own SVG export at three times |
| `tests/scale_check.py` | error and ink ratio from 48 to 768 px |
| `recheck.py` | under 5 minutes: unit tests, minifier ratios against `results/`, evidence refs; `--gpu` adds the GPU suites |

`harness/gpu.mjs` is the WebGPU runner the GPU checks share. It runs headless Chromium on
SwiftShader, so it checks correctness but not speed.

## Layout

```
forge/        wgsl.py ast.py printer.py scope.py types.py names.py minify.py   (WGSL)
              scene.py codegen.py svg.py geom.py color.py                      (shapes)
              gpu.py reflect.py __main__.py
harness/      gpu.mjs
corpus/       86 third-party WGSL files (see corpus/README.md)
examples/     sunrise.py blobs.py
tests/        the checks above; tests/svg/ test SVGs, tests/svg/real/ icons (attribution inside)
results/      JSON per check; svg/ and python/ side-by-side images
web/          gallery page and its build script
```
