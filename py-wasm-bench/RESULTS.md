# py-wasm-bench results

Run 2026-10-18 on a CCotw container (4 vCPU, nothing else running), 5 reps per cell,
median reported. Every timed cell first matched the native CPython 3.11 checksum to
1e-9 relative. Full tables: `results/report.md`; raw: `results/wasmtime.json`,
`results/node.json`.

## Speed against native CPython 3.11

Seconds inside the program around `run(n)` (startup excluded); `x` = speed relative to
native CPython 3.11. Wasm rows are wasmtime 39 / node 22 (V8).

| arm | nbody | mandelbrot | matmul | dictwords |
|---|---|---|---|---|
| native CPython 3.11 | 1.64s (1.00x) | 0.86s (1.00x) | 0.50s (1.00x) | 0.70s (1.00x) |
| cpython-wasi, wasmtime | 3.80s (0.43x) | 1.88s (0.45x) | 1.25s (0.40x) | 2.34s (0.30x) |
| cpython-wasi, node | 4.87s (0.34x) | 2.70s (0.32x) | 1.77s (0.28x) | 2.06s (0.34x) |
| py2wasm, wasmtime | 3.69s (0.45x) | 1.49s (0.58x) | 1.47s (0.34x) | 2.14s (0.33x) |
| py2wasm, node | 3.38s (0.49x) | 1.49s (0.57x) | 1.36s (0.37x) | 1.93s (0.36x) |
| cython-wasi, wasmtime | 1.91s (0.86x) | 0.18s (4.74x) | 1.42s (0.35x) | 1.69s (0.41x) |
| mypyc-wasi, wasmtime | 2.67s (0.62x) | 0.042s (20.3x) | 0.64s (0.77x) | 1.62s (0.43x) |
| mypyc-wasi, node | 2.14s (0.77x) | 0.044s (19.5x) | 0.69s (0.72x) | 1.64s (0.43x) |
| c-ceiling, wasmtime | 0.125s (13.2x) | 0.027s (32.0x) | 0.012s (41.9x) | 0.031s (22.4x) |
| c-ceiling, node | 0.099s (16.6x) | 0.033s (26.3x) | 0.018s (27.3x) | 0.034s (20.5x) |
| c-native (gcc -O2) | 0.072s (22.7x) | 0.026s (32.6x) | 0.010s (50.7x) | 0.016s (43.1x) |

Shipped size per kernel: c-ceiling 33–43 KB; cpython-wasi, mypyc-wasi and cython-wasi
6.8–6.9 MB plus a stdlib tree that is mapped in, not counted; py2wasm 27 MB,
self-contained.

## Findings

1. **py2wasm is 0.85–1.26x the speed of plain CPython compiled to wasm, not 2.5–3x.**
   Against the same `libpython3.11.a` that py2wasm itself links, at the same `-O2`, it
   wins on mandelbrot (1.26x) and loses on matmul (0.85x under wasmtime). Wasmer's
   figure was pystone against their own CPython-wasm build; it does not carry to these
   kernels. Both land at 0.3–0.6x native CPython, and py2wasm's modules are 4x larger.
2. **Running CPython inside wasm costs about 2.3–3.3x** on these kernels (native vs
   cpython-wasi under wasmtime). V8 is slower than wasmtime for the interpreter
   (0.70–0.78x on three kernels) and mixed for compiled C (1.26x on nbody, 0.67x on
   matmul).
3. **Type annotations alone buy speed only where values stay in locals.** mypyc gives
   20x native CPython on mandelbrot, where every value is a local int or float it can
   unbox. On nbody and matmul, whose floats live in `list[float]`, it stays below
   native CPython (0.62x, 0.77x): list elements remain boxed `PyFloat` objects behind
   `PyList`. Cython on unmodified annotated Python (no `cdef`) shows the same shape at
   smaller magnitude (4.7x on mandelbrot).
4. **The ceiling is 30–106x above cpython-wasi.** Hand-written C of the same algorithms,
   compiled to wasm, runs 13–42x native CPython and within 1.0–1.9x of native C under
   wasmtime. The gap between mypyc and that ceiling on nbody and matmul (about 20–55x)
   is value representation: flat `f64` arrays instead of lists of boxed floats.
   dictwords' ceiling is optimistic, since the C uses a purpose-built hash table; a
   compiler would need a general `str`/`dict` runtime there.
5. **Python → WGSL works for the data-parallel kernels.** `arms/wgsl/pykernel.py`
   (~800 lines, stdlib `ast`) translates `@kernel` functions to compute shaders. In
   headless Chromium on SwiftShader, mandelbrot matched a numpy float32 model on all
   250,000 pixels and matmul matched it exactly. Relative error against the f64 Python
   checksum is 4.0e-5 and 4.3e-8 (f32 precision). The 27–114 ms GPU times are software
   rendering and say nothing about real-GPU speed.

The route that closes the gap is a compiler whose typed subset lowers `list[float]` to
a flat typed buffer and `int` to `i64`, and that does not link libpython for the hot
code. That is SPy's design; mypyc and Cython keep the CPython object model and inherit
its cost.

## Not measured

- **SPy.** Blocked: SPy's dependencies must come from PyPI outside the jail (it has no
  network), and the `jail_route` guard refuses any `python`/`pip` command naming
  `/tmp/jail/`, which contradicts `scripts/jail.sh`'s documented install step.
  `arms/spy/NOTES.md` holds what was learned from reading its source.
- Real-GPU WebGPU timing; browser-side wasm startup and download cost.
- Cython with `cdef` typed memoryviews, and mypyc with `native_class`/`i64`
  annotations. Both would narrow finding 3, at the price of code that is no longer
  plain Python.

## Gotchas that cost time

- **clang rewrites `pow(x, 0.5)` to `sqrt` without fast-math.** That moved nbody's
  result by 5e-14 and made it about 3x faster than the Python it models.
  `-fno-builtin-pow` keeps the libm call (`arms/c-ceiling/NOTES.md`).
- **mypyc cross-compiling to wasm32 emits 8-byte pointer strides.** It reads
  `SIZEOF_SIZE_T` from the build host, so list literals silently held garbage
  (`[1.0, 2.0][1]` returned -113.0). Pin `SIZEOF_SIZE_T=4` before importing mypyc
  (`arms/mypyc-wasi/gen_c.py`).
- **py2wasm downloads wasi-sdk 21 (75 MB) on any invocation, even `--help`,** unless
  `WASI_SDK_DIR` is set. Its anti-bloat drops `_json`. Its prebuilt
  `libpython3.11.a` reports 3.11.8+ and works as a stock CPython-for-WASI link target.
- **A CPython guest writes `.pyc` into a mapped stdlib dir** even with
  `PYTHONDONTWRITEBYTECODE=1`; `PYTHONPYCACHEPREFIX` redirects it.
- **Headless Chromium exposes `navigator.gpu` only on an http(s) page,** not on a
  `setContent` page.
