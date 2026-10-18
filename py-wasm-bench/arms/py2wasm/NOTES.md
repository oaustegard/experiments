# py2wasm

One standalone wasm module per kernel: `out/<kernel>.wasm`, compiled by py2wasm 2.6.3 from PyPI.
`run.sh KERNEL N` runs it through `tools/run_wasi.sh` with N as argv[1].

## Build recipe (`build.sh`)

- venv: `python3.11 -m venv /opt/pwb/venv-py2wasm`, then `pip install py2wasm==2.6.3 patchelf`.
- `export WASI_SDK_DIR=/opt/pwb/wasi-sdk-21`. This skips py2wasm's own downloader.
- For each kernel: `py2wasm kernels/<k>.py -o arms/py2wasm/out/<k>.wasm`, run from `build/`.
  Each build takes about 1.5 to 2.5 minutes. `build.sh` skips kernels whose `.wasm` exists.
- Nuitka's standalone mode on Linux needs `patchelf` on PATH, so it is installed in the venv.

## Versions

- py2wasm 2.6.3 (PyPI), Nuitka 2.6.3, Python 3.11.17 venv.
- wasi-sdk 21.0 (clang 17.0.6). Nuitka's `--static-libpython=yes --lto=yes` build.
- wasmtime 39.0.1, node 22.22.0.

## Artifact size

Each `out/<k>.wasm` is about 27 MB (nbody 27,074,088; mandelbrot 27,034,025; matmul 27,046,428;
dictwords 27,048,498 bytes). This is self-contained: the modules run with no `--map` and no
stdlib files on disk.

## Gotchas

- **Downloader trap.** Running `py2wasm` without `WASI_SDK_DIR` (even `--help`) downloads
  a 75 MB wasi-sdk into the venv. Always export the variable first.
- **patchelf missing.** `FATAL: Error, standalone mode on Linux requires 'patchelf'`. Install it.
- **`_json` dropped by anti-bloat.** Nuitka prints `Not including '_json' automatically ... may slow
  down by using fallback implementation`, so `json` uses its pure-Python fallback. None of the kernels
  use `json` inside the timed `run(n)`, so this should not affect the numbers. Re-check it if a kernel
  starts to use `json` in its hot path.
- `wasm-ld: warning: unknown -z value: noexecstack` is harmless.
- **Side effect on the parent's copy.** While py2wasm runs, Nuitka writes `.pyc` files into the
  bundled `wasi-python/lib/python3.11` tree. The venv's copy is the one it uses, but the parent's
  `/opt/pwb/py2wasm-2.6.3-src` copy also received five cache files during compiles. Those
  were deleted, and the stray-write count is 0 at the end.
- Checksums verified at nbody 1000, mandelbrot 60, matmul 30, dictwords 5000 under both runtimes: exact match with CPython 3.11.
