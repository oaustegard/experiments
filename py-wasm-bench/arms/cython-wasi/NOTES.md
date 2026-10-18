# cython-wasi

Unmodified kernel -> `cython -3` C -> statically linked into py2wasm's WASI CPython 3.11, one `out/<kernel>.wasm` per kernel.

## Build recipe

1. `build.sh` copies the kernel unmodified into `build/<k>/<k>.py`, so the module is named `<k>`. Then it runs `/opt/pwb/venv-mypyc/bin/cython -3 -o <k>.c <k>.py`. The `-3` flag is Python 3 semantics, and annotation typing is Cython 3's default for `.py` sources.
2. `main.c` (`-DKERNEL=<k>`): `PyImport_AppendInittab("<k>", PyInit_<k>)`, then `Py_BytesMain`. This is the same file as the mypyc arm.
3. Compile `<k>.c` and `main.c` with `/opt/pwb/wasi-sdk-21/bin/clang --target=wasm32-wasi -O2 -w -fwrapv -D_WASI_EMULATED_{MMAN,GETPID,SIGNAL,PROCESS_CLOCKS}`, against `py2wasm-2.6.3-src/nuitka/wasi-python/include/python3.11`.
4. Link as in the mypyc arm: `libpython3.11.a libmpdec.a libexpat.a`, the four `-lwasi-emulated-*` libs, `-ldl`, a stack of 8 MB, and `--strip-all`.
5. `driver/driver.py`, `run.sh`, and `env.sh` are the same as in the mypyc arm. The stderr line shows `run=cython_function_or_method`, which confirms the compiled module is the one imported.

## Versions

- Cython 3.3.0 in `/opt/pwb/venv-mypyc` (CPython 3.11.17)
- wasi-sdk 21.0 (clang 17.0.6), CPython 3.11 for WASI from py2wasm 2.6.3 (3.11.8+)
- wasmtime 39.0.1, node 22.22.0

## Gotchas

- None required a workaround. Cython's generated C has no host pointer-size assumptions, so it compiled and ran without patching.
- Stripped: 6.8 MB per kernel. Unstripped: 20.4 MB, the same as mypyc, since the bulk is libpython.
- Measured, not tuned: `nbody.c` has 24 `double` locals and 25 `PyObject *` locals. The annotation typing did not make the arithmetic fully unboxed.
- The nbody checksum differs from native CPython by 3.3e-16 relative. That is inside the 1e-9 tolerance; the cause was not investigated.

## Verified (small N, both runtimes, checksum vs native CPython 3.11)

nbody 1000, mandelbrot 60, matmul 30, dictwords 5000: all pass in both runtimes. The `bench.py` run is still to be done by the parent.
