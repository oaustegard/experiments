# rht-operator-native — compiled structured RHT against dense `sgemm`

**Status: done — positive.** A 60-line C kernel that applies remex's randomized
Hadamard rotation directly (permute, sign, block FWHT) beats the materialized
dense matrix on every call shape measured from d=1024 up (one query, batches
of 64 to 10,000, decode), on x86, ARM and Apple Silicon, at one thread and at
all threads. d=768 is a win outside a band of small batches that a mis-set
serial cutoff hands back to `sgemm`; d=384 is a split. The operator's output
hashes identically on all four machines measured, where dense output takes
three or four distinct values. Getting reproducible **codes** needed a
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

<!-- table: batch encode, n=10,000, 1 thread -->
| machine | threads | d=384 | d=768 | d=1024 | d=1536 | d=2048 | d=3072 | d=4096 |
|---|---|---|---|---|---|---|---|---|
| Xeon AVX-512, 1 vCPU (authoring container) | 1 | 1.20 | 0.64 | 0.30 | 0.35 | 0.15 | 0.18 | 0.08 |
| GitHub ubuntu-24.04 x64 | 1 | 0.93 | 0.54 | 0.20 | 0.25 | 0.10 | 0.12 | 0.05 |
| GitHub ubuntu-24.04-arm (Neoverse V2) | 1 | 0.37 | 0.19 | 0.08 | 0.10 | 0.04 | 0.05 | 0.02 |
| GitHub macos-15 (Apple Silicon, Accelerate) | 1 | 1.11 | 0.76 | 0.33 | 0.41 | 0.17 | 0.22 | 0.09 |
<!-- /table -->

**Batch encode, n = 10,000, all threads**

<!-- table: batch encode, n=10,000, all threads -->
| machine | threads | d=384 | d=768 | d=1024 | d=1536 | d=2048 | d=3072 | d=4096 |
|---|---|---|---|---|---|---|---|---|
| GitHub ubuntu-24.04 x64 | 4 | 0.79 | 0.49 | 0.16 | 0.21 | 0.09 | 0.11 | 0.05 |
| GitHub ubuntu-24.04-arm (Neoverse V2) | 4 | 0.37 | 0.19 | 0.08 | 0.10 | 0.04 | 0.05 | 0.02 |
| GitHub macos-15 (Apple Silicon, Accelerate) | 3 | 1.17 | 0.75 | 0.33 | 0.40 | 0.18 | 0.22 | 0.09 |
<!-- /table -->

**Single query (`R @ q`), one thread**

<!-- table: single query, 1 thread -->
| machine | threads | d=384 | d=768 | d=1024 | d=1536 | d=2048 | d=3072 | d=4096 |
|---|---|---|---|---|---|---|---|---|
| Xeon AVX-512, 1 vCPU (authoring container) | 1 | 0.79 | 0.18 | 0.08 | 0.04 | 0.02 | 0.02 | 0.01 |
| GitHub ubuntu-24.04 x64 | 1 | 0.81 | 0.29 | 0.15 | 0.11 | 0.05 | 0.04 | 0.02 |
| GitHub ubuntu-24.04-arm (Neoverse V2) | 1 | 0.60 | 0.19 | 0.09 | 0.06 | 0.03 | 0.03 | 0.01 |
| GitHub macos-15 (Apple Silicon, Accelerate) | 1 | 0.46 | 0.40 | 0.16 | 0.08 | 0.04 | 0.03 | 0.03 |
<!-- /table -->

**Small and medium batches, all threads**

n = 64:

<!-- table: batch 64, all threads -->
| machine | threads | d=384 | d=768 | d=1024 | d=1536 | d=2048 | d=3072 | d=4096 |
|---|---|---|---|---|---|---|---|---|
| GitHub ubuntu-24.04 x64 | 4 | 1.54 | 0.76 | 0.30 | 0.39 | 0.15 | 0.24 | 0.05 |
| GitHub ubuntu-24.04-arm (Neoverse V2) | 4 | 1.30 | 0.68 | 0.28 | 0.36 | 0.14 | 0.19 | 0.02 |
| GitHub macos-15 (Apple Silicon, Accelerate) | 3 | 1.11 | 0.73 | 0.43 | 0.48 | 0.14 | 0.11 | 0.06 |
<!-- /table -->

n = 128:

<!-- table: batch 128, all threads -->
| machine | threads | d=384 | d=768 | d=1024 | d=1536 | d=2048 | d=3072 | d=4096 |
|---|---|---|---|---|---|---|---|---|
| GitHub ubuntu-24.04 x64 | 4 | 1.65 | 0.82 | 0.33 | 0.49 | 0.08 | 0.10 | 0.05 |
| GitHub ubuntu-24.04-arm (Neoverse V2) | 4 | 1.38 | 0.73 | 0.29 | 0.38 | 0.04 | 0.05 | 0.02 |
| GitHub macos-15 (Apple Silicon, Accelerate) | 3 | 1.31 | 0.80 | 0.39 | 0.46 | 0.14 | 0.11 | 0.07 |
<!-- /table -->

n = 256:

<!-- table: batch 256, all threads -->
| machine | threads | d=384 | d=768 | d=1024 | d=1536 | d=2048 | d=3072 | d=4096 |
|---|---|---|---|---|---|---|---|---|
| GitHub ubuntu-24.04 x64 | 4 | 1.69 | 0.88 | 0.16 | 0.24 | 0.08 | 0.13 | 0.05 |
| GitHub ubuntu-24.04-arm (Neoverse V2) | 4 | 1.39 | 0.74 | 0.08 | 0.24 | 0.04 | 0.05 | 0.02 |
| GitHub macos-15 (Apple Silicon, Accelerate) | 3 | 1.46 | 0.89 | 0.42 | 0.37 | 0.15 | 0.06 | 0.08 |
<!-- /table -->

n = 1024:

<!-- table: batch 1024, all threads -->
| machine | threads | d=384 | d=768 | d=1024 | d=1536 | d=2048 | d=3072 | d=4096 |
|---|---|---|---|---|---|---|---|---|
| GitHub ubuntu-24.04 x64 | 4 | 0.76 | 0.39 | 0.15 | 0.20 | 0.08 | 0.11 | 0.04 |
| GitHub ubuntu-24.04-arm (Neoverse V2) | 4 | 0.37 | 0.19 | 0.08 | 0.19 | 0.04 | 0.05 | 0.02 |
| GitHub macos-15 (Apple Silicon, Accelerate) | 3 | 1.45 | 0.83 | 0.35 | 0.38 | 0.18 | 0.23 | 0.08 |
<!-- /table -->

Decode (`X @ R`) tracks encode within a few percent everywhere. The tables
are the latest CI run (7be5fce). Ranges below span all four runs after the
timing fix (6b638f2, 9bdd544, 567ec45, 7be5fce), archived under
`ci/history/`; the first of those predates the n = 128–1024 cells.

- **From d=1024 up the operator is faster in every cell measured**: every
  machine, every run, every shape and thread count, at 0.007–0.49 of dense
  time. The slowest cells are x64 at 4 threads, d=1536, n=128.
- **d=768 is a win except where the serial cutoff bites.** Batch 10,000 runs
  at 0.19–0.87 of dense and single queries at 0.17–0.40 (one thread). At 4
  threads on x64, batches of 128 and 256 come in at 0.82–1.04: at d=768 the
  cutoff keeps every batch below 341 rows serial while `sgemm` uses all
  cores. macOS is the weakest machine at d=768 (0.66–0.87 batch).
- **d=384 is a split.** Batches of 64–256 lose to dense on every multi-core
  machine in every run (1.11–1.81). Batch 10,000 loses on AVX-512 x86 (1.20
  on the authoring box, 1.16–1.29 on the one x64 run that drew SkylakeX) and
  is mixed on macOS (0.74–1.22). It wins on ARM (0.37 in every run) and on
  the x64 runs that drew Haswell (0.77–0.93).
- **The x64 runner's numbers depend on which OpenBLAS kernel it draws**:
  Haswell in three runs, SkylakeX (AVX-512) in one. SkylakeX `sgemm` is
  faster, so that run's ratios are the least favourable to the operator.
  On ARM, batch ratios moved by at most 0.01 between runs and single-query
  ratios by up to 0.11.
- **The serial cutoff is set too high.** On ARM at d=384, n=256 (serial)
  costs 1.30–1.40x dense while n=1024 (parallel) costs 0.37x. A cutoff near
  2^15–2^16 input floats would likely recover most of the small-batch cells;
  that is a prediction and was not run.
- Thread scaling at 4 threads matches `sgemm`'s: about 2x on the x64 runner
  (two physical cores), 3.9x on ARM.
- **macOS is not a controlled comparison.** Apple clang has no `-fopenmp`, so
  the kernel ran serially there, and `threadpoolctl` cannot see Accelerate,
  so the dense arm's threads were not limited in either row.
- Construction (authoring box): 29 ms → 0.08 ms at d=768, 741 ms → 0.16 ms at
  d=3072. Resident rotation state: 2.4 MB → 12 KB, 37.7 MB → 49 KB, 67 MB →
  33 KB at d=4096.

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

<!-- table: fingerprints d=768 -->
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
<!-- /table -->

With both changes, codes hash identically on all four machines at 2, 4 and
8 bits, at both dimensions. With only the operator, 4- and 8-bit codes split
into two hash groups at d=768; with only the new boundaries, the dense
rotation still gives three or four (the x64 runner matches the authoring
box whenever it draws the same SkylakeX kernel).

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

1. **Apply `rht` through the operator, with a lower serial cutoff.** Faster
   from d=1024 up and at d=768 outside the small-batch band, reproducible,
   and removes the resident matrix. `Quantizer.R` can stay a lazy property
   returning `rht_rotation(d, seed)` unchanged, so `gpu.py`, `save_params` and
   the Mojo R-parity test are untouched. Existing `rht` indexes would see
   ~1e-6 of codes move, the same size as moving them to another CPU today.
2. **Compile at import, or ship wheels.** remax `_native.py` is the in-house
   pattern for compiling at import. The compiler-default
   build is within ~15% of `-march=native` and avoids SIGILL on shared caches.
   Without a compiler, the NumPy fallback (same hashes) is 3–9x slower than
   today's dense batch path; falling back to dense instead keeps the speed and
   loses the reproducibility. Locked-down deployments (distroless images,
   read-only home, serverless) would hit that fallback silently, which argues
   for prebuilt wheels (cibuildwheel), or at least a logged warning and a
   setting to choose the fallback.
3. **Boundaries from the float32 centroids.** A one-line change in
   `codebook.py`, independent of the rotation. Moves 1–65 boundaries by an ulp
   (65 of 255 at 8 bits). The centroid table does not change, so every
   existing code, Haar included, decodes exactly as before; only newly encoded
   coordinates within an ulp of a moved boundary can land in the neighbouring
   cell, a ~1e-6 share, which moving an index to another CPU already causes
   today, so the container formats need no new version. Worth pairing with
   (1) in one release so the change happens once.
4. **Mojo.** Parity becomes guaranteed only if the Mojo port also applies the
   operator and derives boundaries the same way. The authoring container has
   no Mojo toolchain, so the Mojo side was neither built nor timed.
5. **Default rotation.** With the operator, `rht` is ahead of Haar on build
   time, memory, apply speed from d=1024 up, and reproducibility, at measured
   recall parity (experiments#11). The rotation is recorded on disk and
   `rotation_identity_gate.py` covers a default flip.

## Caveats

- Speed at d=384 is machine-dependent and not a win on AVX-512 x86 or macOS.
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
- libgomp is not fork-safe once a parallel region has run: a pre-forking
  server (gunicorn `--preload`, Celery prefork) that encodes before forking
  can hang its children. A remex integration should default to one thread
  or document this. This experiment did not test fork behaviour.

## Files

`rht_kernel.c`, `build.sh`, `rhtop.py` (kernel and plan);
`check_correct.py`, `check_bitexact.py`, `check_codes.py`,
`check_blas_drift.py`, `check_simd_bisect.py`, `check_simd_codebook.py`;
`bench_apply.py`, `bench_variants.py` (local benches); `ci_run.py`,
`ci/*.json`, `ci/history/` (every post-fix run), `history.py`, `make_tables.py`,
`regen_tables.py`, `recheck.py`; `ERRORS.md`.
