# py-wasm-bench

Which route from Python to WebAssembly (and WebGPU) closes the speed gap, measured
rather than argued. Oskar, 2026-10-18: *"Could we improve on current platform tooling
for converting code to run under wasm and/or webgpu? … more like py2wasm but
optimized."* The working hypothesis from that conversation: py2wasm's ceiling is
CPython's object model, so the gain lives in type-specialised compilation of hot
functions (wasm) and a `@kernel` → WGSL path (WebGPU), not in tuning Nuitka.

Status: **in progress**. Results go in `RESULTS.md` once every arm has run.

## Kernels

`kernels/*.py`, plain typed Python, each exposing `run(n) -> float` and a `__main__`
that prints one JSON line `{"kernel","n","checksum","seconds"}` with `seconds` timed
around `run(n)` only. Defaults are sized for ~0.7–1.5 s on native CPython 3.11.

| kernel | stresses |
|---|---|
| `nbody` | float arithmetic through list indexing |
| `mandelbrot` | pure scalar float/int loops |
| `matmul` | nested list-of-lists float loops |
| `dictwords` | strings, dicts, sorting: dynamic glue a typed compiler helps least |

## Arms

One directory per arm under `arms/`, contract in the `bench.py` docstring: `build.sh`,
`run.sh KERNEL N` (last stdout line is the kernel's JSON), optional `artifact.txt`
(shipped bytes) and `SKIP` (reason the arm cannot run). Wasm arms run through
`tools/run_wasi.sh`, which takes `RUNTIME=wasmtime` (default) or `RUNTIME=node`
(V8, Chrome's engine).

| arm | route |
|---|---|
| `cpython-native-3.11`, `-3.13` | reference speed and checksums |
| `cpython-wasi` | stock CPython 3.11 interpreter compiled to wasm32-wasi |
| `py2wasm` | Wasmer's Nuitka fork, Python → C → wasm |
| `mypyc-wasi`, `cython-wasi` | typed Python → C extension, statically linked into CPython-wasi |
| `spy` | SPy (static Python dialect) → C → wasm |
| `c-ceiling` | hand-written C of the same algorithms → wasm: what an ideal typed-subset compiler reaches |
| `wgsl` | AST → WGSL prototype for the data-parallel kernels, run in headless Chromium |

`bench.py` checks every result against a native CPython checksum (relative tolerance
1e-9) before it records a time, so an arm that computes the wrong thing fails loudly.

## Environment

CCotw container, 4 vCPU. wasi-sdk 21 (the version py2wasm's prebuilt
`libpython3.11.a` was built with) and wasmtime 39.0.1 under `/opt/pwb`, downloaded from
GitHub release assets. Node 22.22. Headless Chromium's WebGPU here is SwiftShader, so
the `wgsl` arm settles correctness, not GPU speed.
