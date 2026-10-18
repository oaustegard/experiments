# arms/wgsl: pykernel, a Python-subset to WGSL compiler

A prototype that takes a Python function marked `@kernel`, reads its AST, and emits a
WGSL compute shader. The function body is the work of one invocation. The arm runs the
shaders in headless Chromium's WebGPU and checks them against a numpy float32 model.
`SKIP` keeps `bench.py` from running it, since f32 cannot meet bench.py's 1e-9 check.

## Files

| file | role |
|---|---|
| `pykernel.py` | the translator (stdlib `ast`, `inspect`, `textwrap`) |
| `test_pykernel.py` | 30 checks: inference, buffer access, rejections, golden output, real kernels |
| `kernels_gpu.py` | `mandelbrot` and `matmul` as `@kernel` functions, plus host-side inputs and checksums |
| `build.py`, `build.sh` | translate to `out/<k>.wgsl`, write `out/<k>.json` (manifest), write f32 inputs |
| `run_gpu.js` | Node runner: playwright-core launches Chromium and serves `page/` on localhost |
| `page/index.html` | `window.runKernel`: adapter, `getCompilationInfo()` gate, dispatch, readback, timing |
| `verify.py` | builds, runs the GPU, compares with numpy float32 and the f64 kernels |

Run order: `python3 test_pykernel.py`, then `python3 verify.py`. `verify.py` calls
`build.sh` and `run_gpu.js` itself.

## The subset

Signature: the first parameter is the invocation index, `gid: int`. Each other
parameter is one of:

| annotation | WGSL |
|---|---|
| `list[float]` | `var<storage, read>` or `read_write` `array<f32>`. `read_write` if the body assigns into it |
| `int` | a field of the uniform struct `Params`, as `i32` |
| `float` | a field of the uniform struct `Params`, as `f32` (read-only inside the kernel) |

Every kernel also gets a uniform field `pk_count`: the invocation count. The generated
`main` returns early when `gid >= pk_count`, so the host can round the dispatch up.

Statements: assignment to a name or `list[index]`, augmented assignment, annotated
assignment (`x: float = 0`), `if`/`elif`/`else`, `while`, `for i in range(stop)` and
`for i in range(start, stop)`, `break`, `continue`, bare `return`, `pass`, and a
docstring.

Expressions: int and float literals, bool literals, names, `+ - * /` (`/` is Python
true division and always gives f32), `//` and `%` on int operands only, unary `- + not`,
single comparisons `< <= > >= == !=`, `and`/`or` over bool operands, `list[index]`
loads, and the calls `min(a, b)`, `max(a, b)`, `abs(x)`, `float(x)` and `int(x)`, plus the
math functions `sqrt`, `sin`, `cos`, `tan`, `exp`, `log`, `atan2(y, x)`, `pow(x, y)` (all f32) and
`floor(x)`, which returns an int as `math.floor` does. A kernel module imports the math names from
`math` so it still runs as plain Python (added 2026-10-18 for the austegard.com demos).

Beyond the list in the task: `float()` and `int()` casts, `//` and `%` on ints,
`continue`, and annotated locals. `int()` truncates toward zero, as Python does.

Rejected with `KernelError("line N: unsupported node <NodeType>: ...")`: lambdas, lists,
tuples, comprehensions, strings, attribute access, calls to anything else, `**`, bitwise
operators, chained comparisons, conditional expressions, `while`/`for` with `else`,
`for` over anything but `range` with 1 or 2 arguments, return values, use before
assignment, bool arithmetic, truthiness of numbers, float indices, nested lists,
parameters without a type, names starting with `pk_` or `__`, and a missing `@kernel`.

## Design

1. **Type inference is a fixpoint.** The translator makes passes over the body until
   local types stop changing. `int` joins `float` to `float`, so `x = 0; x = x + 0.5`
   makes `x` an f32 and converts the literal where it lands. Any other mismatch, such as
   `bool` with a number, is an error that names both types.
2. **Definite assignment is checked after inference, in source order.** A read is legal
   only after a textual assignment. An `if` counts only when both branches assign. A
   loop body does not count after the loop. This matches Python's NameError behavior,
   and it keeps the zero that a hoisted declaration gives from being read silently.
3. **All locals are hoisted** to the top of `main` with zero initializers. WGSL block
   scoping is then never a problem, and the Python function scope carries over.
4. **`for` loops evaluate the bound once.** The stop expression goes in a `let`, and the
   WGSL counter is separate from the Python name. A body assignment to `i` therefore does
   not change the iteration, as in Python.
5. **Python int semantics.** `//` and `%` go through `pk_floordiv` and `pk_mod`, which
   take the sign of the divisor as Python does. The helpers are emitted only when used.
6. **Naming.** Python names that are WGSL reserved words (`select`, `min`, `type`, and
   others) become `pk_<name>`. Generated names all start with `pk_`, and user names may
   not, so the two cannot collide.
7. **Output contract (manifest).** `translate()` returns the WGSL plus `buffers`
   (binding, access), `uniforms` (field, type, byte offset), `uniform_binding` and
   `uniform_size`. The host creates an explicit bind-group layout from these, not
   `layout: 'auto'`, which would drop unused bindings.
8. **Dispatch.** `workgroup_size` comes from `@kernel(workgroup_size=N)`, default 64.
   The grid is 1-D, `gid = row * n + col` for the two kernels here.

### Kernels

- `mandelbrot`: one invocation per pixel. It writes `float(k)` into `out[gid]`. The host
  computes `inside * 1000000 + total_iters % 1000000` from the buffer, as the Python
  kernel does.
- `matmul`: one invocation per output element, `c[gid] = sum_k a[i*n+k] * b[k*n+j]`.
  The inputs are `kernels/matmul.py`'s own `make()`, flattened, cast to f32 at
  packing, and uploaded as storage buffers. The sum runs in the same k order as the
  Python loop.

### Runner

The page does all the WebGPU work and reports failures in a result object rather than
throwing. Each failure it can name returns `{ok:false, stage, error, navigator_gpu, adapter}`. The stages are `navigator.gpu`,
`requestAdapter`, `requestDevice`, `getCompilationInfo` (any error-severity message),
`createComputePipeline`, `dispatch` and `device`. Each run is one dispatch, a copy into a
MAP_READ buffer, submit, and `mapAsync`. The timer is `performance.now()` from before
submit until `mapAsync` resolves. There is one untimed warm-up, then five timed runs, and
the median is reported. Chromium is launched with `executablePath` from
`/opt/pw-browsers/chromium-1194` and the four flags from the task. The page is served
over http, because a `setContent` page had no `navigator.gpu` in this container.

## Results (SwiftShader, software WebGPU, 4 vCPU container)

| kernel | N | verdict | escape-count match / checksum | GPU median |
|---|---|---|---|---|
| mandelbrot | 500 | PASS (>= 99.9%) | 100.0000%, 0 pixels differ | ~27 to 49 ms (5 runs, two sessions) |
| matmul | 200 | PASS (<= 1e-4) | max element rel err 0, checksum rel err 0 vs numpy f32 | ~47 to 114 ms (5 runs, two sessions) |

Against the f64 checksum from `python3.11 kernels/<k>.py`:
mandelbrot 4.0e-5 (the GPU and numpy-f32 agree exactly, so the gap is f32 against f64:
f32 counts four more pixels inside the set, and the iteration total differs by 30), matmul
4.3e-8. The timing ranges are from
two verify runs, and the medians move with the software renderer.

Failure path, checked: a shader with an undefined identifier stops at the
`getCompilationInfo` gate with `getCompilationInfo: 43:16 unresolved value 'undefined_thing'`.

## Limits of the prototype

- **Not GPU timing.** SwiftShader is a software rasterizer on the CPU. The numbers show
  the shaders run and answer correctly, not how fast a GPU would run them.
- **Out-of-bounds indexing is not checked.** WebGPU's robust buffer access clamps or
  discards such accesses. It does not throw, and the translator does not add a check.
- **Integer arithmetic wraps at 32 bits.** Python ints are unbounded. Integer division by
  zero is defined in WGSL (it returns the numerator); Python raises.
- **Floats are f32.** The f32 path matches numpy's float32 elementwise, but not the f64
  checksums (see above). No fused multiply-add is assumed, and WGSL does not guarantee
  one.
- **1-D grid only.** `dispatchWorkgroups(ceil(n*n / wg))` is capped by the 65535
  per-dimension dispatch limit. That gives about 4.2M invocations at workgroup size 64,
  and larger grids need a 2-D dispatch.
- **No `list[int]`, no nested lists, no tuples, no structs, no vectors, no workgroup
  memory or barriers.** The subset is scalar code over flat f32 buffers.
- **Definite assignment does not follow control flow.** A `return` in one branch does
  not make the other branch's assignments count. This is conservative and can reject
  valid code.
- **Bench integration.** `bench.py` only lists arms that have a `run.sh`. This arm has
  none, and `SKIP` is present as the task asked. `run.sh` would need the GPU result to
  print a checksum for all four bench kernels, and `dictwords` and `nbody` have no
  `@kernel` form.
- **Nothing is committed.** `arms/wgsl/.gitignore` excludes `node_modules/`, `out/` and
  `__pycache__/`.

## What a production version would need

- **A typed IR between the AST and WGSL.** It would hold the type rules and let the
  translator emit the same program to WGSL, SPIR-V or Metal. The pass structure would
  stay: fixpoint typing, then definite assignment, then codegen.
- **Real control-flow analysis** for definite assignment, and a proper scope model in
  place of hoisting.
- **Full Python integer semantics,** or an explicit contract for wrap-around, with
  division by zero trapping on the host side.
- **Bounds-checked indexing as an option,** for debugging, and a kernel-level
  `robustness` setting.
- **More of the type system:** `u32`, `vec2/3/4`, `list[list[...]]` as strided views, and
  structs as buffer records. Shared memory and `workgroupBarrier`, so reductions and
  tiled matmul become expressible.
- **Multi-dimensional dispatch** and a launch model with `grid` and `block` as the
  user sees them, plus automatic splitting past the dispatch limit.
- **Validation without a browser.** Naga or Tint offline checks, plus a differential
  fuzzer that compares random kernels against CPython, so the translator is tested
  beyond the two kernels here.
- **Host runtime:** buffer pooling, pipeline caching, several dispatches in flight, and
  timestamp queries when the adapter allows them, in place of `performance.now()`
  around `mapAsync`.
- **Measurement on real hardware,** with warm-up, cold-start separated from steady
  state, and the transfer cost counted in the comparison.
