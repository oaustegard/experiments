# ERRORS — rht-operator-native

What was wrong, how it was caught, and which way it pushed the conclusion.

| # | Error | Caught by | Direction |
|---|---|---|---|
| 1 | **Interleaved timing.** `ci_run.py` timed each dense cell and then its operator cell back to back. OpenBLAS workers and the libgomp team both spin-wait after a call, so each arm ran against the other's spinning threads. At 4 threads on the first CI run the operator scaled 0.91x (x64) and 1.40x (ARM) at d=384 while dense scaled 2.0x and 4.0x. | Implausible scaling at the smallest d on both architectures; nothing else in the kernel changes with d that way. | Against the operator at small d (x64 d=384 went from 1.69 to 0.77 op/dense after the fix). Fixed in 6b638f2: dense cells and operator cells are timed in separate blocks with a sleep between. |
| 2 | **OpenMP build regressed the serial path 37%.** Wrapping the row loop in `#pragma omp parallel` slowed single-thread d=3072 batch encode from 239 ms to 327 ms, even with the team disabled by an `if` clause. | Comparing the OpenMP and non-OpenMP builds after the first local rerun came out slower than the one before it. | Against the operator. Fixed with an explicit serial branch outside the parallel region (226 ms). |
| 3 | **Thread-team wake-up on small batches.** n=64 at 4 threads on ARM: operator 290 us against 106 us dense at d=384. | First CI run. | Against the operator. Fixed: serial below 2^18 input floats. |
| 4 | **ctypes `ndpointer` validation dominated single queries.** About 10 us of a 14 us call at d=384. | First CI run's single-query column had a d-independent floor. | Against the operator. Fixed: raw pointers, with dtype/contiguity/shape checked once in Python. d=384 single query went from 14 us to 7 us locally. |
| 5 | **macOS crash.** `rhtop.Op` called `os.sched_getaffinity`, which does not exist on macOS. First run's macOS job failed; logs are unreadable from the authoring container (the proxy blocks the log blob redirect), so the cause was inferred from the code and confirmed by the next run passing. | CI job status. | No effect on numbers; one machine missing from the first run. |
| 6 | **Attributed code drift to the rotation, then to the norms; it was the codebook.** The second CI run showed the operator's codes at d=768/4-bit differing on the x64 runner, which had drawn a Haswell OpenBLAS kernel. I first suspected `np.sum` in the norm step. Reproduced with `NPY_DISABLE_CPU_FEATURES=X86_V4` and bisected (`check_simd_bisect.py`): norms, unit vectors and rotated vectors identical, boundaries not. | Fingerprint table plus a local reproduction, before any claim was written. | Would have overstated the reproducibility result: "operator codes are bit-identical across machines" was true for 3 of 4 machines by luck of which SIMD level NumPy dispatched. The claim is now restricted to what was measured, and the codebook drift is reported as its own finding. |
| 7 | **repo-index push race.** This experiment's CI commits results to `main` while `repo-index.yml` is rebuilding; repo-index pushes without rebasing and failed twice (bf272d7, 6b638f2). | Actions run list. | No effect on results. Fixed by rebasing in `repo-index.yml` before its push. |
| 8 | **Table regenerator deleted half the write-up.** Its regex let an empty `<!-- table -->` block match across the next marker; everything from the speed bullets to the fingerprint table vanished. | Section list after the run. | None on results. Restored from the committed copy; the regex now refuses to cross a marker and asserts the marker count is unchanged. |
| 9 | **Claims written against one run.** The write-up quoted ranges from the latest run; a doc-tooling commit retriggered the benchmark, the x64 runner drew a different OpenBLAS kernel, and five recheck claims failed. | `recheck.py` on the new data. | Mixed: the new run was more favourable to the operator at d=384 and less at d=768 batch-128. Every run is now archived in `ci/history/`, claims are ranges across all of them, and doc tooling no longer triggers the benchmark. |

Base rate: 9 errors, 4 of which biased the comparison **against** the arm under test. None of the fixes changed which arm was faster above d=768.

**History rewrite, 2026-09-17.** The code arrays under
`integration/ci/96c831c/codes` and the local `integration/out` were removed
from git history with `git filter-repo` (pack 84 → 66 MiB). Every commit from
their first appearance onward has a new hash; the run identifiers in
`RESULTS.md`, `ERRORS.md`, `ci/history/ORDER` and the `ci/history/` directory
names were remapped through filter-repo's commit map. `repo-index/index.npy`
and the other experiments' `.npy` data were left alone: only those two paths
were dropped.
