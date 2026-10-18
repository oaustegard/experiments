# mypyc-wasi

Typed kernel -> mypyc C -> statically linked into py2wasm's WASI CPython 3.11, one `out/<kernel>.wasm` per kernel.

## Build recipe

1. `gen_c.py KERNEL_SRC BUILD_DIR` copies the unmodified kernel into `build/<k>/` (so the module is named `<k>`), then calls `mypyc.build.mypycify([<k>.py], target_dir="gen", opt_level="2")`. That emits `gen/__native.c` only; it does not invoke a host compiler. The lib-rt `.c` files are `#include`d from `__native.c`, so the only include path needed is `mypyc.build.include_dir()`.
2. `main.c` (`-DKERNEL=<k>`): `PyImport_AppendInittab("<k>", PyInit_<k>)`, then `Py_BytesMain`.
3. Compile with `/opt/pwb/wasi-sdk-21/bin/clang --target=wasm32-wasi -O2 -w -fwrapv -D_WASI_EMULATED_{MMAN,GETPID,SIGNAL,PROCESS_CLOCKS}`, against `py2wasm-2.6.3-src/nuitka/wasi-python/include/python3.11`.
4. Link `main.o native.o libpython3.11.a libmpdec.a libexpat.a -lwasi-emulated-{mman,getpid,signal,process-clocks} -ldl`, with `-Wl,-z,stack-size=8388608` and `-Wl,--strip-all`.
5. `driver/driver.py KERNEL N` runs inside the guest. It imports the builtin module, asserts `__spec__.origin == "built-in"`, times `run(n)` with `perf_counter`, and prints the kernel's JSON line. The stderr line shows `run=builtin_function_or_method`, which is the mypyc-compiled function. A `.py` function would show `function`.
6. `run.sh` calls `tools/run_wasi.sh` with `--map <stdlib>::/pyhome/lib/python3.11`, `--map arms/mypyc-wasi/driver::/drv`, `--env PYTHONHOME=/pyhome`, and `PYTHONUTF8=1`.

## Versions

- mypy 2.4.0, mypyc from that wheel, in `/opt/pwb/venv-mypyc` (CPython 3.11.17, native build host)
- wasi-sdk 21.0 (clang 17.0.6)
- CPython 3.11 for WASI from py2wasm 2.6.3 (`patchlevel.h`: 3.11.8+)
- wasmtime 39.0.1, node 22.22.0

## Gotchas

- **Host pointer size leaks into mypyc output (fixed in `gen_c.py`).** mypyc takes `PLATFORM_SIZE` (and the `c_pyssize_t` / tagged-int widths) from `sysconfig SIZEOF_SIZE_T` on the build host, which is 8. `lower/list_ops.py` then writes list-literal items at `ob_item + 8*i`. On wasm32 that stride is wrong. The symptom was silent corruption: `[1.0, 2.0][1]` came back as `-113.0`, the mypyc float error sentinel, and `nbody` aborted in `_Py_Dealloc` with "uninitialized element". `gen_c.py` makes `sysconfig.get_config_var("SIZEOF_SIZE_T")` return 4 before mypyc is imported. That is the ILP32 path mypyc already has for 32-bit hosts.
- `__native.c` includes the lib-rt `.c` files directly, so there is one translation unit per kernel. No lib-rt object files are built separately.
- Not stripped: 20.4 MB per kernel. Stripped with `--strip-all`: 6.8 MB. The arm ships the stripped file.
- The stdlib mapped into the guest is 231 MB (`lib/python3.11` including tests). It is mapped, not shipped, and is not in `artifact.txt`. The shipped size is the `.wasm` alone, which includes the CPython core.
- The `.py` source is never in the guest path. The driver's origin assertion checks this.

## Verified (small N, both runtimes, checksum vs native CPython 3.11)

nbody 1000, mandelbrot 60, matmul 30, dictwords 5000: all pass, relative error 0 in both runtimes. The `bench.py` run is still to be done by the parent.
