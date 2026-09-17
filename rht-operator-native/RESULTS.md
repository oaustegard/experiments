# rht-operator-native — compiled structured RHT against dense `sgemm`

**Status: done — positive.** A 60-line C kernel that applies remex's randomized
Hadamard rotation directly (permute, sign, block FWHT) beats the materialized
dense matrix on every call shape from d=768 up, on x86, ARM and Apple Silicon,
at one thread and at all threads. Its output also hashes identically on all
four machines measured, where the dense path's output differs on every one. Getting reproducible **codes** needed a
second change, in the Lloyd-Max codebook, found along the way.

Origin: a conversation about arXiv:2609.15083 (SL(n) representation learning)
led to the question of what remex could still get out of its rotation.
[remex#72](https://github.com/oaustegard/remex/issues/72) had already
measured a NumPy operator form and closed it as won't-do (dense won batches
7–35x); its closing comment named a compiled FWHT as the lever it had not
tried. That is the whole of this experiment.

## Prior art

- **remex#72** (closed, not planned): NumPy structural apply loses to `sgemm`
  on every batch; wins only single queries at d ≥ 2048; `ivf` search is not
  matvec-dominated. Storage win called real but resident-only.
- **experiments#10** (`remex-vs-higgs-ablation`): a BLAS-bound FWHT crosses
  Haar at d ≈ 1024.
- **remax#60**: single-round RHT loses stack independence on anisotropic input
  in remax; remex keeps one round when d is a power of two, and this kernel keeps
  remex's round count unchanged.
- **FFHT** (FALCONN, Andoni et al.) is the established AVX C implementation,
  with a pip fork (`ffht-unofficial`) that adds OpenMP 2-D support. It
  handles power-of-two lengths only; remex's construction needs any even d
  (block-diagonal FWHT plus permutations), which is why this kernel was
  written rather than wrapped. It does not try to beat FFHT's inner loop.
- **remax `_native.py`** is the in-house precedent for compile-at-import C
  with a NumPy fallback.

## Method

`rht_kernel.c` replays `remex.rotation.rht_rotation`'s plan from the same
PCG64 stream (`rhtop.plan`), row by row, two d-length buffers per thread,
OpenMP across rows above 2^18 input floats. Built `-O3 -ffp-contract=off`
(the flag matters, see below). `ci_run.py` measures, on each machine:

1. max |operator − dense| and bit-equality across `-O0`, compiler-default ISA,
   `-march=native`, 1 and N threads, and a NumPy reference
   (`check_bitexact.numpy_ref`) that performs the same elementwise operations;
2. SHA-256 fingerprints of R, rotated vectors, centroids, boundaries and
   codes at 2/4/8 bits, d=768 and d=3072, on an identical input;
3. best-of-k time for dense `X @ R.T` / `R @ q` / `X @ R` and the operator,
   n ∈ {1, 64, 10,000}, d from 384 to 4096, with BLAS and OpenMP limited to
   the same thread count, and the two arms timed in separate blocks (ERRORS #1).

Machines: the 1-vCPU authoring container (Xeon, AVX-512), and three GitHub
runners driven by `.github/workflows/rht-operator-native.yml`, which commits
each machine's JSON to `ci/`. Every number below comes from `make_tables.py`
over those files; `recheck.py` asserts each claim against them.

## Correctness

The operator matches the dense matrix to float32 rounding (max abs 2–3e-6 on
unit-scale input, relative 1–6e-7) at every d from 6 to 4096, both
directions, and round-trips to 1e-6 (`check_correct.py`). Retrieval is
unaffected: across d ∈ {384, 768, 1536, 3072} × isotropic/anisotropic × 2/4/8
bits, recall@10 was identical in 23 of 24 cells and +0.0005 in the last,
where 1 of 200 queries changed its top-10 (`check_codes.py`). Operator codes
differ from dense codes in 1e-7–4e-7 of coordinates at 2-bit, ~1e-6 at 4-bit
and ~1e-5 at 8-bit.

## Speed

Ratios are operator time / dense time; below 1 the operator is faster.

**Batch encode, n = 10,000, one thread**

| machine | threads | d=384 | d=768 | d=1024 | d=1536 | d=2048 | d=3072 | d=4096 |
|---|---|---|---|---|---|---|---|---|
| Xeon AVX-512, 1 vCPU (authoring container) | 1 | 1.20 | 0.64 | 0.30 | 0.35 | 0.15 | 0.18 | 0.08 |
| GitHub ubuntu-24.04 x64 | 1 | 0.93 | 0.55 | 0.20 | 0.25 | 0.10 | 0.13 | 0.05 |
| GitHub ubuntu-24.04-arm (Neoverse V2) | 1 | 0.37 | 0.20 | 0.08 | 0.10 | 0.04 | 0.05 | 0.02 |
| GitHub macos-15 (Apple Silicon, Accelerate) | 1 | 1.22 | 0.78 | 0.31 | 0.44 | 0.21 | 0.21 | 0.09 |

**Batch encode, n = 10,000, all threads**

| machine | threads | d=384 | d=768 | d=1024 | d=1536 | d=2048 | d=3072 | d=4096 |
|---|---|---|---|---|---|---|---|---|
| GitHub ubuntu-24.04 x64 | 4 | 0.80 | 0.49 | 0.16 | 0.22 | 0.09 | 0.11 | 0.05 |
| GitHub ubuntu-24.04-arm (Neoverse V2) | 4 | 0.37 | 0.20 | 0.08 | 0.10 | 0.04 | 0.05 | 0.02 |
| GitHub macos-15 (Apple Silicon, Accelerate) | 3 | 1.01 | 0.87 | 0.28 | 0.42 | 0.19 | 0.21 | 0.09 |

**Single query (`R @ q`), one thread**

| machine | threads | d=384 | d=768 | d=1024 | d=1536 | d=2048 | d=3072 | d=4096 |
|---|---|---|---|---|---|---|---|---|
| Xeon AVX-512, 1 vCPU (authoring container) | 1 | 0.79 | 0.18 | 0.08 | 0.04 | 0.02 | 0.02 | 0.01 |
| GitHub ubuntu-24.04 x64 | 1 | 1.11 | 0.40 | 0.16 | 0.14 | 0.05 | 0.04 | 0.01 |
| GitHub ubuntu-24.04-arm (Neoverse V2) | 1 | 0.55 | 0.19 | 0.09 | 0.06 | 0.03 | 0.03 | 0.01 |
| GitHub macos-15 (Apple Silicon, Accelerate) | 1 | 0.53 | 0.22 | 0.11 | 0.08 | 0.04 | 0.03 | 0.01 |

**Batch of 64, all threads**

| machine | threads | d=384 | d=768 | d=1024 | d=1536 | d=2048 | d=3072 | d=4096 |
|---|---|---|---|---|---|---|---|---|
| GitHub ubuntu-24.04 x64 | 4 | 1.55 | 0.74 | 0.30 | 0.39 | 0.16 | 0.20 | 0.03 |
| GitHub ubuntu-24.04-arm (Neoverse V2) | 4 | 1.30 | 0.70 | 0.28 | 0.36 | 0.14 | 0.19 | 0.02 |
| GitHub macos-15 (Apple Silicon, Accelerate) | 3 | 1.49 | 0.76 | 0.24 | 0.32 | 0.13 | 0.16 | 0.07 |

Decode (`X @ R`) tracks encode within a few percent everywhere.

- From d=1024 up the operator is 3–50x faster on every machine and thread
  count measured. At d=768 it is 1.3–5x faster on Linux; on macOS it is a
  narrow win (0.78 one-thread, 0.87 three-thread).
- **d=384 is a split.** Dense wins batch encode on the AVX-512 box (1.20) and
  on macOS (1.22); the operator wins on ARM (0.37) and at 4 threads on x64
  (0.80). Batches of 64 at small d stay on the serial path by design and lose
  to multi-threaded `sgemm` (1.3–1.55 at d=384).
- Thread scaling at 4 threads matches `sgemm`'s: 2.0x on the x64 runner
  (two physical cores), 3.9x on ARM.
- **macOS is not a controlled comparison.** Apple clang has no `-fopenmp`, so
  the kernel ran serially there, and `threadpoolctl` cannot see Accelerate,
  so the dense arm's threads were not limited in either row.
- Construction (authoring box): 29 ms → 0.08 ms at d=768, 741 ms → 0.16 ms at
  d=3072. Resident rotation state: 2.4 MB → 12 KB, 37.7 MB → 49 KB, 67 MB →
  33 KB at d=4096.

The x64 runner's 1-thread column at d=1024 moved from 0.27 to 0.20 between
two runs on different OpenBLAS kernels (SkylakeX, then Haswell): read single
cells to about ±0.1.

## Reproducibility

Every step of the operator is an elementwise IEEE operation in a fixed order
(gather, multiply, pairwise add/sub), so its output does not depend on the
instruction set, the thread count, or NumPy vs C. It hashed identically across
all build variants and the NumPy reference on all four machines.
`-ffp-contract=off` keeps a compiler from fusing a multiply into an add; no
such pattern exists in the kernel today, but the flag makes it a guarantee.

The dense path is not reproducible. On the authoring box alone, forcing four
OpenBLAS kernel families (`OPENBLAS_CORETYPE` = SkylakeX, Haswell,
Sandybridge, Prescott) gives four different code hashes at d=768 and d=3072,
flip rates 3e-7 to 1e-6 (`check_blas_drift.py`).

That fixes the rotation. The codes still differed: the second CI run's x64
runner produced different 4-bit operator codes from the other three machines
while agreeing on every rotated vector. Reproduced locally with
`NPY_DISABLE_CPU_FEATURES=X86_V4` and bisected (`check_simd_bisect.py`) to the
**codebook**. `lloyd_max_codebook` iterates in float64 through
`scipy.stats.norm`, whose `cdf`/`pdf` differ by an ulp across NumPy's SIMD
dispatch levels. The float32 centroid table it returns absorbs that; the
boundaries do not, because they are computed as midpoints of the float64
centroids before the cast. At 4 bits the middle boundary comes out 0.0 at one
dispatch level and 3.6e-17 at another. Taking the boundaries as midpoints of
the float32 centroids instead (`check_simd_codebook.py`) removes the
dependence. Fingerprints at d=768 (d=3072 in the same shape, same verdicts):

| field | local-xeon-avx512-1cpu | gha-ubuntu24-x64 | gha-ubuntu24-arm64 | gha-macos15-arm64 | agree |
|---|---|---|---|---|---|
| `R` | `f8e83129` | `f8e83129` | `f8e83129` | `f8e83129` | **yes** |
| `op_rot` | `6147b47e` | `6147b47e` | `6147b47e` | `6147b47e` | **yes** |
| `dense_rot` | `45672793` | `65ac7490` | `e0bc6abe` | `a89191d7` | no (4 distinct) |
| `cents2` | `2cda3067` | `2cda3067` | `2cda3067` | `2cda3067` | **yes** |
| `bounds2` | `af4e57c4` | `af4e57c4` | `af4e57c4` | `af4e57c4` | **yes** |
| `bounds2_f32mid` | `239f914e` | `239f914e` | `239f914e` | `239f914e` | **yes** |
| `dense_codes2` | `63a2a33a` | `608dbd19` | `63a2a33a` | `608dbd19` | no (2 distinct) |
| `op_codes2` | `63a2a33a` | `63a2a33a` | `63a2a33a` | `63a2a33a` | **yes** |
| `op_codes2_f32mid` | `63a2a33a` | `63a2a33a` | `63a2a33a` | `63a2a33a` | **yes** |
| `cents4` | `8a0ff63a` | `8a0ff63a` | `8a0ff63a` | `8a0ff63a` | **yes** |
| `bounds4` | `bb40fae6` | `13a37d73` | `a2f45808` | `f8102a2e` | no (4 distinct) |
| `bounds4_f32mid` | `d45806cf` | `d45806cf` | `d45806cf` | `d45806cf` | **yes** |
| `dense_codes4` | `b4dd566c` | `75c8c955` | `c02b2da8` | `44ceb088` | no (4 distinct) |
| `op_codes4` | `44ceb088` | `c02b2da8` | `44ceb088` | `44ceb088` | no (2 distinct) |
| `op_codes4_f32mid` | `c02b2da8` | `c02b2da8` | `c02b2da8` | `c02b2da8` | **yes** |
| `cents8` | `ef2a746b` | `ef2a746b` | `ef2a746b` | `ef2a746b` | **yes** |
| `bounds8` | `0281de7c` | `29ebb8ac` | `21eb60ca` | `0fba5608` | no (4 distinct) |
| `bounds8_f32mid` | `b697300d` | `b697300d` | `b697300d` | `b697300d` | **yes** |
| `dense_codes8` | `ffb0bb33` | `61893fa8` | `5efdde01` | `5e2652d3` | no (4 distinct) |
| `op_codes8` | `41cd1a9b` | `6c348e31` | `6c348e31` | `41cd1a9b` | no (2 distinct) |
| `op_codes8_f32mid` | `327fcd5d` | `327fcd5d` | `327fcd5d` | `327fcd5d` | **yes** |

With both changes, codes hash identically on all four machines at 2, 4 and
8 bits, at both dimensions. With only the operator, 4- and 8-bit codes split
into two hash groups at d=768; with only the new boundaries, the dense
rotation still gives four.

This bears on two existing remex claims:

- `CLAUDE.md` promises identical results for the same `(d, bits, seed)`.
  That holds within a machine. Across machines it fails at 4 and 8 bits for
  **both** rotations, because the codebook drift is independent of the
  rotation.
- The Mojo port's byte-parity table (1–4 bits exact, 8-bit ~99.8%) is
  measured against one Python build. The Python side's own 4-bit boundaries
  differ across the four machines here, so that parity holds for particular
  machine pairs.

## Proposed remex changes (not applied)

Each item below needs a maintainer decision before it goes into remex.

1. **Apply `rht` through the operator.** Faster from d=768 up, reproducible,
   and removes the resident matrix. `Quantizer.R` can stay a lazy property
   returning `rht_rotation(d, seed)` unchanged, so `gpu.py`, `save_params` and
   the Mojo R-parity test are untouched. Existing `rht` indexes would see
   ~1e-6 of codes move, the same size as moving them to another CPU today.
2. **Compile at import.** remax `_native.py` is the pattern. The compiler-default
   build is within ~15% of `-march=native` and avoids SIGILL on shared caches.
   Without a compiler, the NumPy fallback (same hashes) is 3–9x slower than
   today's dense batch path; falling back to dense instead keeps the speed and
   loses the reproducibility.
3. **Boundaries from the float32 centroids.** A one-line change in
   `codebook.py`, independent of the rotation. Moves 1–65 boundaries by an ulp
   (65 of 255 at 8 bits) and therefore a similar ~1e-6 share of codes for
   **every** existing index, Haar included. Worth pairing with (1) in one
   release so the drift happens once.
4. **Mojo.** Parity becomes guaranteed only if the Mojo port also applies the
   operator and derives boundaries the same way. The authoring container has
   no Mojo toolchain, so the Mojo side was neither built nor timed.
5. **Default rotation.** With the operator, `rht` is ahead of Haar on build
   time, memory, apply speed from d=768 up, and reproducibility, at measured
   recall parity (experiments#11). The rotation is recorded on disk and
   `rotation_identity_gate.py` covers a default flip.

## Caveats

- Speed at d=384 is machine-dependent and not a win on x86 or macOS.
- The x64 runner has two physical cores; no many-core machine was measured.
- macOS: serial kernel, uncontrolled Accelerate threads (above).
- Only 2,000 vectors per fingerprint; agreement is an observed property of
  these inputs plus the argument from the operations above; other inputs
  were not hashed.
  NumPy's float64 norm step agreed everywhere here and was not changed.
- `ivf` search latency is not re-profiled; remex#72 found the query matvec to
  be at most 14% of an `ivf` search at d=3072, so the single-query win there
  is bounded accordingly.
- The kernel does not implement remax's multi-rotation stacked encode.

## Files

`rht_kernel.c`, `build.sh`, `rhtop.py` (kernel and plan);
`check_correct.py`, `check_bitexact.py`, `check_codes.py`,
`check_blas_drift.py`, `check_simd_bisect.py`, `check_simd_codebook.py`;
`bench_apply.py`, `bench_variants.py` (local benches); `ci_run.py`,
`ci/*.json`, `make_tables.py`, `recheck.py`; `ERRORS.md`.
