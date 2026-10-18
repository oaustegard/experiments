# cpython-wasi

Stock CPython 3.11 interpreter as a WASI command (`out/python.wasm`), running the four kernels
from `../../kernels/` through `../../tools/run_wasi.sh`.

## Build recipe (`build.sh`)

- Stub `build/main.c`: `#include <Python.h>` and `return Py_BytesMain(argc, argv);`.
- Compile with `/opt/pwb/wasi-sdk-21/bin/clang --target=wasm32-wasi --sysroot=.../wasi-sysroot -O2 -w`,
  `-I$W/include/python3.11`, plus the `-D_WASI_EMULATED_{MMAN,GETPID,SIGNAL,PROCESS_CLOCKS}` defines
  from `nuitka/build/SconsCompilerSettings.py`.
- Link with `$W/lib/libpython3.11.a $W/lib/libmpdec.a $W/lib/libexpat.a`, then
  `-lwasi-emulated-{mman,getpid,signal,process-clocks} -ldl -lm`, and `-Wl,--strip-all`
  (20 MB unstripped, 6.8 MB stripped). `$W` is `/opt/pwb/py2wasm-2.6.3-src/nuitka/wasi-python`.
- `Py_BytesMain`, `PyInit__json`, `PyInit__decimal`, `PyInit_pyexpat` and the other `Setup.stdlib`
  modules are statically linked into `libpython3.11.a`. No dynamic loading is needed.

## Run recipe (`run.sh KERNEL N`)

`tools/run_wasi.sh out/python.wasm` with:
- `--map $W/lib/python3.11::/pyhome/lib/python3.11` (stdlib), `--env PYTHONHOME=/pyhome`
- `--map kernels::/kernels`, then runs `/kernels/KERNEL.py N`
- `--map build/pycache::/pycache --env PYTHONPYCACHEPREFIX=/pycache`

## Versions

- Interpreter: py2wasm 2.6.3's prebuilt `libpython3.11.a`. `sys.version` reports
  `3.11.8+ (heads/3.11-dirty, Mar 12 2024)`, built by Clang 17.0.6. It is **not** the native 3.11.17
  used as the reference. Checksums still match exactly.
- wasi-sdk 21.0 (clang 17.0.6), from the GitHub release tarball into `/opt/pwb/wasi-sdk-21`.
- wasmtime 39.0.1 (`/usr/local/bin/wasmtime`); node 22.22.0.

## Artifact size

- `out/python.wasm`: 6,799,810 bytes (`artifact.txt` counts this file only).
- Not counted: the stdlib, which is mapped from py2wasm's tree (233 MB including `test/`, but
  only the modules a kernel imports are read). A shipped app would need a pruned copy, so the true
  shipped size is larger than the number above.

## Gotchas

- **Bytecode writes.** The guest writes `.pyc` files into any mapped stdlib dir, including `opt-1` and
  `opt-2` variants, even with `PYTHONDONTWRITEBYTECODE=1`. `run.sh` therefore redirects them with
  `PYTHONPYCACHEPREFIX` into `build/pycache`. The first run is slower, and startup is not timed.
- **Cold start.** wasmtime spends about 3 s compiling `python.wasm` on each run, in wall time only.
  Node takes about 0.4 s wall. The timed `seconds` value excludes both.
- **Stack.** wasmtime is run with `-W max-wasm-stack=8388608` from `run_wasi.sh`, and node with
  `--stack-size=8000`. These were not tuned for this arm.
- Checksums verified at nbody 1000, mandelbrot 60, matmul 30, dictwords 5000 under both runtimes: exact match with CPython 3.11.
