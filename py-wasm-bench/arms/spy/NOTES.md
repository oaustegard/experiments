# SPy arm: notes

**Status: built and verified at DEFAULT_N for all four kernels.** Source is SPy
commit `a39950cab36e7adeca4808fb0e75081526ee44c8` (2026-10-12, "Introduce the SIMD
type (#734)"), shallow clone. Nothing committed.

## Layout

- `src/<kernel>.spy`: the four ports.
- `build.sh`: builds `out/<kernel>.wasm` (wasm32-wasi). Idempotent in output.
- `run.sh KERNEL N`: runs the module under `tools/run_wasi.sh`, RUNTIME=wasmtime only.
- `artifact.txt`: `out/{kernel}.wasm`.
- `build/` (gitignored): `bin/python` shim and the copies used for probing.

## Setup (what was done, once)

1. `GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://github.com/spylang/spy /tmp/jail/spy`,
   then `chown -R nobody:nogroup /tmp/jail/spy`.
2. Pinned wheels from `pyproject.toml` into `/tmp/jail/spy/.deps`, with the form the
   jail_route guard admits: `python3.12 -I -m pip install --only-binary=:all: --target
   /tmp/jail/spy/.deps build~=1.3 click==8.1.8 fixedint==0.2.0 ninja==1.13.2 py==1.11.0
   typer==0.15.1 pexpect==4.9.0 wasmtime==48.0.0 ziglang==0.13.0`, run while the checkout
   was still root-owned. The checkout was then chowned to nobody.
3. `build.sh` does the rest, with every SPy process inside `scripts/jail.sh`.

## Build recipe

Per run, `build.sh` does:

1. libspy for wasi, both flavours, through the jail:
   - `make -C spy/libspy TARGET=wasi BUILD_TYPE=debug OUTPUT_KIND=testlib`. The compiler
     loads `build/wasi/debug-testlib/libspy.wasm` at import time.
   - `make -C spy/libspy TARGET=wasi BUILD_TYPE=release OUTPUT_KIND=exe`, giving
     `build/wasi/release/libspy.a`.
2. Per kernel, stage `src/<k>.spy` into `/tmp/jail/spy/stage/<k>/` (nobody-owned).
   `spy build` writes its C and `build.ninja` into the default `build/` beside the source,
   so the staging dir has to be writable by nobody.
3. Inside the jail:
   `python -m spy build --no-spyc --no-compile --target wasi --output-kind exe --release stage/<k>/<k>.spy`,
   then `sed -i 's/ -flto//g' build/build.ninja`, then `ninja -C build`.
4. Copy `build/<k>.wasm` to `out/`, after checking the `\0asm` magic.

Environment the jail needs (passed with `-e`):
- `PATH` with `arms/spy/build/bin` first. Its `python` is a shim to `/usr/bin/python3.12`,
  because SPy pins `requires-python ~=3.12.0` and the jail's bare `python` is 3.13.
- `PYTHONPATH=/tmp/jail/spy/.deps:/tmp/jail/spy`.

## Toolchain facts found on the way

- Options must come BEFORE the filename. `spy build FILE [ARGV]...` takes everything
  after FILE as program argv, so `spy build FILE --target wasi` silently builds the
  native target.
- `-O LEVEL` does not reach the C flags: `spy/cli/commands/build.py` never passes
  `opt_level` to `BuildConfig`. The C optimisation level comes from `--release` alone
  (`-O3 -flto` in `spy/build/flags.py`).
- `--release` link fails. The generated C and the release libspy are compiled with
  `-flto`, and zig's wasm-ld prints "function signature mismatch" warnings for the
  compiler-rt long-double helpers, then aborts (SIGABRT). Removing `-flto` from the
  generated `build.ninja` links cleanly, and the outputs match the Python checksums,
  so the build strips it. The release libspy is linked without `-flto`, which lld
  accepts here.
- The Makefile and `spy build` call bare `python` (`$(shell python -m spy.build.flags ...)`, `python -m ziglang cc`).
  The shim keeps them on the 3.12 the wheels were installed for (3.13 was not tried).
- `str(f64)` matches CPython's `repr(float)` (shortest round-trip, `1e-05` style), so
  the checksum prints at full precision and the JSON carries the same double the
  Python kernel printed.

## Dialect changes per kernel

Applied to every kernel: `from time import time` for the timing; `def main(argv:
list[str]) -> i32` reads N with `int(argv[1])`; the JSON line is built with `+` and
`str()`. `** 0.5` stays `** 0.5` (f64 pow, backed by libm `pow`).

- **nbody**: `list[list[f64]]` becomes one flat `list[f64]` with 7 fields per body. SPy
  rejects augmented assignment through a nested subscript (`nested_list_augassign`), so
  `b[3] -= x` becomes `bodies[7*i + 3] -= x`. Every expression keeps the Python operand
  order. Module-level `SOLAR_MASS` etc. are plain consts.
- **mandelbrot**: `inside` and `total_iters` are f64, not i64. SPy has no `f64(i64)`,
  and `i64 * i32` is rejected, so the i64 path cannot reach the f64 return. Both
  counters stay below 2**53 (`inside` <= n*n, `total` <= 50*n*n), so each addition is
  integer-exact. The result `inside*1e6 + total % 1e6` equals Python's `float(...)`.
- **matmul**: `list[list[f64]]` becomes one row-major `list[f64]` (element (i, j) at
  `i*n + j`), for the same reason as nbody. The i, k, j accumulation order is unchanged.
- **dictwords**:
  - LCG: `(x * 1103515245 + 12345) % 2**31` runs in `u32` with wraparound and a
    `0x7fffffff` mask. That is exact: the low 31 bits of a product depend only on the
    low 32 bits of its operands, and 2**31 divides 2**32. The state is below 2**31 after
    each step, so `i32(x % 12)` is a valid index. The i64 route was rejected: there is no
    `i32(i64)` or `u32(i64)`, and list indexing needs an `i32`.
  - Module-level `SYLLABLES` list: SPy cannot yet redshift a prebuilt module-level list
    ("This prebuilt constant cannot be redshifted (yet)"). The list is built in `run()`
    and passed to `make_word`.
  - `sorted(counts.items(), key=(-count, word))[:10]`: no `sorted` or `items()` in the
    stdlib at this commit. It becomes a hand-written top-10 insertion that keeps the list
    ordered by (count desc, word asc). That order is total over distinct words, so the
    first ten entries equal Python's. Keys come from `for w in counts` (dict key
    iteration), values from `counts[w]`.
  - `acc * 100000 + len(counts)`: computed in f64 (`f64(acc) * 100000.0 + f64(len)`),
    exact because the values are small integers.

Inexpressible: nothing. Every kernel ran at its default N with the stated workarounds.

## Verification (DEFAULT_N, against `python3.11 kernels/<k>.py`)

| kernel | N | SPy checksum | Python checksum | rel. diff |
|---|---|---|---|---|
| nbody | 200000 | -0.16908371256962862 | -0.1690837125696371 | 4e-14 |
| mandelbrot | 500 | 99378102374.0 | 99378102374.0 | 0 |
| matmul | 200 | 5876827.206929504 | 5876827.206929504 | 0 |
| dictwords | 500000 | 44734605653.0 | 44734605653.0 | 0 |

The nbody difference is at the last few ulps after 200000 steps. The likely source is
libm `pow` (musl in the wasi sysroot, glibc in CPython), since the operation order
matches. This is within the 1e-9 tolerance but was not isolated further.

`python3 bench.py --reps 1 --arms spy` (in-program seconds, wall in parentheses):

```
spy                      nbody          0.185s  wall    0.219s
spy                      mandelbrot     0.026s  wall    0.057s
spy                      matmul         0.147s  wall    0.179s
spy                      dictwords      0.304s  wall    0.339s
```

Artifact sizes (`out/*.wasm`): nbody 1074592, mandelbrot 1032707, matmul 1062749,
dictwords 1098873 bytes. These are whole modules, including the statically linked
libspy and musl pieces.

## Runtime notes

- `run.sh` refuses RUNTIME=node with `unsupported: node runtime not used for
  jailed-compiler output` and exit 3. Any other RUNTIME value is an error.
- The modules import only `wasi_snapshot_preview1` functions. Checked by string scan
  on each output. The other strings in the binaries are zig's build-path debug info.
- SPy's runtime is built with `SPY_GC_NONE`, so nothing is freed. dictwords allocates a
  word and a parts list per iteration with no GC. Peak memory was not measured.
- Each run is a single rep here. Use `--reps 5` for the matrix.
