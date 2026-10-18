# c-ceiling: hand-translated C, compiled to wasm32-wasi

The ceiling an ideal type-specialising compiler could reach: each kernel in
`kernels/` written by hand in C with the same algorithm and the same
floating-point operation order, compiled with clang to WASI.
`arms/c-native/` builds the same sources with gcc as a native reference.

## Layout

- `src/<kernel>.c`: one file per kernel. `main` takes N from argv[1] (default
  `DEFAULT_N` from the Python file) and prints the contract JSON. `seconds` is
  `clock_gettime(CLOCK_MONOTONIC)` around `run(n)`, which holds the same work
  Python's `run(n)` does (setup, kernel, final reduction), not process startup.
- `build.sh` writes `out/<kernel>.wasm`. `run.sh KERNEL N` runs it through
  `tools/run_wasi.sh`, so `RUNTIME=wasmtime` (default) or `RUNTIME=node` works.
- `artifact.txt`: `out/{kernel}.wasm`.

## Toolchains

| tool | version | where |
|---|---|---|
| wasi-sdk 21.0 (clang 17.0.6, wasi-libc, wasm-ld) | release tarball | `/opt/pwb/wasi-sdk-21` |
| wasmtime | 39.0.1 | `/usr/local/bin/wasmtime` |
| node (V8, WASI via `node:wasi`) | v22.22.0 | system |
| gcc (native reference) | 13.3.0, Ubuntu 24.04, glibc | system |
| CPython reference | 3.11.17 | `python3.11 kernels/<k>.py` |

## Flags

- wasm: `clang -O2 --target=wasm32-wasi -ffp-contract=off -fno-builtin-pow -Wall ... -lm`
- native: `gcc -O2 -ffp-contract=off -fno-builtin-pow -Wall ... -lm`
- No `-ffast-math`, no `-march` flags, so no FMA and no `-funsafe-math`.
- `-ffp-contract=off` stops a*b+c being fused. Neither target emits FMA at this
  baseline, so this is a guard, not a change to the result.
- `-fno-builtin-pow` is required. Without it, clang 17 rewrites
  `pow(x, 0.5)` to `llvm.sqrt.f64` even without fast-math (confirmed in the
  LLVM IR). `sqrt` is not what CPython computes (CPython calls libm `pow`).
  The first build did exactly that: nbody's wasm checksum diverged from native
  at relative 5e-14, and the wasm nbody ran about 3x faster than native, which
  is the sqrt. With the flag, the call stays `pow` and nbody matches native
  bit-for-bit.

## Verification at DEFAULT_N

Checksums compared with `python3.11 kernels/<k>.py`. Tolerance 1e-9 relative;
mandelbrot and dictwords are exact.

| kernel (N) | Python | native | wasmtime | node |
|---|---|---|---|---|
| nbody (200000) | -0.1690837125696371 | match, rel 0 | match, rel 0 | match, rel 0 |
| mandelbrot (500) | 99378102374.0 | exact | exact | exact |
| matmul (200) | 5876827.206929504 | match, rel 0 | match, rel 0 | match, rel 0 |
| dictwords (500000) | 44734605653.0 | exact | exact | exact |

Median of 5 runs, `seconds` field, 4-vCPU container (noisy; single runs
varied up to about 2x):

| kernel | native gcc | wasmtime | node |
|---|---|---|---|
| nbody | 0.0895 s | 0.0895 s | 0.0952 s |
| mandelbrot | 0.0250 s | 0.0247 s | 0.0359 s |
| matmul | 0.0078 s | 0.0091 s | 0.0188 s |
| dictwords | 0.0289 s | 0.0240 s | 0.0493 s |

Wasm module sizes (raw `wasm-ld` output, no `wasm-opt`, no strip; includes
wasi-libc's printf, malloc and clock_gettime):
nbody 42548 B, mandelbrot 33041 B, matmul 33599 B, dictwords 36338 B.

## Deviations from the Python algorithm

None in the algorithm, the operation order, or the constants. These differences
are representation only:

1. **Output format.** `%.17g` prints integral checksums without a `.0`
   (`99378102374`, not `99378102374.0`). The values are the same doubles.
2. **nbody constants.** Python computes `1.66e-03 * DAYS_PER_YEAR` and similar
   at run time. The C literals are the same expressions, folded by the compiler
   with the same IEEE round-to-nearest result. `SOLAR_MASS` is `(4.0 * PI) * PI`,
   the same association as Python.
3. **matmul data layout.** Flat row-major `double` arrays instead of
   lists of lists. Loop order i-k-j and the accumulation `ci[j] += aik * bk[j]`
   are unchanged, and `ci` is explicitly zeroed as `[0.0] * n` is.
4. **dictwords hash map.** Python's dict is replaced by an open-addressing table
   (FNV-1a, linear probing, 2^16 slots, each word stored inline, at most 8 bytes).
   The number of distinct words is bounded by 22620 (words are 1 to 4 syllables
   from 12, so 12 + 144 + 1728 + 20736), so the table never fills.
5. **dictwords sort.** `qsort` over pointers with comparator
   `(-count, word)`: count descending, then `memcmp` bytewise, with a shorter
   prefix first. That matches Python's tuple order for ASCII. Keys are unique,
   so `qsort` instability cannot change the top 10.
6. **Integer `%` and `//`.** Python floors on negatives, C truncates. Every
   operand here is non-negative: the LCG state is less than 2^31, so
   `x * 1103515245 + 12345` is under 2^62 and stays in int64. `x // 7` and
   `% 12`, `% 4`, `% 2147483648` are non-negative. `total_iters % 1000000` is
   non-negative. Checked by range reasoning, not by a test.
7. **`pow` is libm, not bit-identical across libms.** The arms call the libm
   `pow` for `** 0.5`, as CPython does. wasi-libc (musl) and glibc `pow`
   are not guaranteed bit-identical. A probe of 2,000,000 inputs
   (`x = 1e-3 + i * 0.00137`) gave different bit hashes for musl (wasm) and
   glibc (native and CPython). The nbody checksum still matched at the default N
   on all runtimes. Any future N could drift in the last bits, and bench.py's
   1e-9 tolerance absorbs that.

## Not done

- `bench.py` was not run (instructed). `arms/c-native` also builds the artifact
  at `out/<kernel>`, a native binary whose size is not a shipped size.
- No commit, no push.
