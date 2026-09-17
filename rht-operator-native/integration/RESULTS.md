# integration — remex `feat/rht-operator` against v0.7.0, before merge

**Status: done — merge criteria met.** remex#86 items 1 (operator-form `rht`)
and 3 (boundaries from float32 centroids), implemented in
[remex#87](https://github.com/oaustegard/remex/pull/87), measured through the
public API against remex `main` (v0.7.0) on GitHub x64, ARM, macOS and
Windows runners. The operator-level study this follows is
[`../RESULTS.md`](../RESULTS.md).

## Method

`run_all.sh` installs `main` and the branch into separate `--target`
directories and runs `bench.py` three times: main, branch, main again (the
noise control). `bench.py` times `Quantizer` construction, `encode` for
n = 1 to 10,000, `search`, `search_adc` and `search_batch`, fingerprints
boundaries and codes at 2/4/8 bits for `rht` and `haar`, and on the branch
sweeps the operator's serial/parallel cutoff. `codediff.py` reduces the
codes to shares that differ, on the runner. The workflow also runs the
branch's rotation tests on each platform. `compare.py ci/<sha>` prints the
tables.

Two runs: `ci/96c831c` (x64, ARM, macOS; before the per-call overhead fix)
and `ci/debedb8` (all four platforms, after it). The macOS control in the
second run swung 0.6–1.7, so that run's macOS timings are excluded from the
ranges below.

## Results

**Reproducibility.** On the branch, boundaries and `rht` codes hash
identically on all four platforms at d ∈ {384, 768, 3072} and 2/4/8 bits. On
`main`, 4- and 8-bit boundaries take four distinct values across the four
platforms and `rht` codes up to three. Haar codes still vary on the branch,
as expected: they go through a dense matmul, and only their boundaries
changed.

**Code changes, main → branch.** At most 1.6e-6 of coordinates at 2 and 4
bits and 2.3e-5 at 8 bits, on every platform; the main → main control
changed none. Haar codes change by at most 3.3e-6 (boundaries only).

**Compiled kernel.** Built and loaded on all four platforms, Windows
included; no runner fell back to NumPy. The branch's rotation, operator and
scalar-mode tests passed on all four (136–137 passed).

**Speed, branch time / main time:**

| | d=384 | d=768 | d=1024 | d=1536 | d=3072 |
|---|---|---|---|---|---|
| construction | 0.83–0.94 | 0.61–0.76 | 0.52–0.72 | 0.22–0.40 | 0.06–0.11 |
| encode, 1 vector | 0.93–1.78 | 0.23–1.03 | 0.20–0.76 | 0.19–0.55 | 0.09–0.29 |
| encode, 64 | 0.93–1.15 | 0.81–0.91 | 0.62–0.78 | 0.62–0.74 | 0.40–0.51 |
| encode, 10,000 | 0.84–0.97 | 0.76–0.93 | 0.67–0.85 | 0.59–0.87 | 0.41–0.67 |
| search | 0.82–1.15 | 0.91–1.02 | 0.98–1.25 | 0.87–1.03 | 0.84–0.95 |
| search_adc | 0.96–1.00 | 0.62–1.00 | 0.66–1.04 | 0.75–1.08 | 0.81–1.02 |

- Search is unchanged within its noise (the control ranged up to 1.20 in the
  same cells): the scan dominates, not the query rotation.
- **d=384 small calls are slower on two platforms.** Single-vector encode:
  +4 µs on ARM (16 → 20 µs) and +27 µs on Windows (34 → 61 µs) after the
  overhead fix; x64 is at parity. Batches of 64–256 are up to 15% slower on
  Windows. ctypes calls are expensive on Windows; packing the fixed arguments
  into a struct and reading one fewer array pointer would save an estimated
  1–1.5 µs of a ~9.5 µs call on Linux, and was not done.
- **The first run found the d=384 overhead** (1.32x on x64, 1.61x on ARM):
  the default thread count was reread from the environment and CPU affinity
  on every call. Cached in debedb8.
- **Encode is now bounded by `np.searchsorted`**, about two-thirds of
  `encode` at d=768 and d=3072 on the authoring box. That is the next target.
- **Cutoff sweep.** The serial/parallel crossover falls between 2^14.6 and
  2^15.6 input floats on every platform, so the 2^15 default stands.

## Errors in this stage

1. The authoring session's editable install made `from remex import _native`
   resolve to the branch while measuring `main`, so a local `main` run
   reported `native: true`. Timings were unaffected (`core.py` came from
   `main`); `bench.py` now checks for `RHTOperator` first. CI uninstalls the
   editable package before benchmarking.
2. The first run saved every code array into the uploaded artifact and
   committed 278 MB of `.npy` under `ci/96c831c/codes`. Diffs are now
   computed on the runner and the arrays were removed from the tree; they
   remain in git history.
3. The local bench's `out/` directory (two JSONs, 30 arrays, ~60 MB) was
   committed before `.gitignore` existed. Every runner re-uploaded it and the
   collect job's `cp` failed on the colliding copies, twice. The job logs are
   unreachable from the authoring session; an `ERR` trap that writes the
   failing command to a check-run annotation found it. The local JSONs moved
   to `ci/local-xeon-1cpu/`.
4. The Windows job first failed at checkout; the full `experiments` tree has
   paths Windows cannot create. The job now uses a sparse checkout.
