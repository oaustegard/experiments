# lowbit-scan-crossover — a reported scale gate that was a constant term

**Question (from Oskar):** challenge the inevitability of BLAS outperforming
low-bit storage at smaller corpus sizes, as recorded in memory `dab41dd6`.

**Verdict:** The crossover is real as measured and is not a property of corpus
size. `dab41dd6`'s own published table fits `t = 4.108 ms + 32.40 ns·n`; the
same numpy expression here fits `t = −0.78 ms + 33.98 ns·n`. **Per-row cost
agrees to 5% across the two machines — the entire crossover is a 4.1 ms
constant in that measurement's harness.** Solving `n* = a/(b_f32 − b_ham)`
recovers **n\* ≈ 68,000**, which is the reported gate, derived rather than
interpolated between two rows. With no constant term the equation has no
solution, and no crossover was observed here at any n from 100 to 1e6, at any
ISA level from SSE4.2 up, cold or warm, single-query or batch-1024.

## Prior art — read this before the results

**The C kernel here is a reimplementation, not a contribution.**
`remax/src/remax/_native.py` already contains a `__builtin_popcountll` Hamming
scan (8 bytes per iteration, compiled at import with `-O3`, ctypes-loaded,
user-private cache, graceful NumPy fallback), and
`remax.packing.hamming_distances` already dispatches to it. Its docstring
already reports **25–35x over the NumPy path**, already identifies the
100k–1M cache falloff, and already warns against quoting the ratio as a single
number because it moves with both `n` and `d`. This experiment's 37x warm /
19x cold reproduces that on different hardware; it does not discover it.

Also already known in-account: **a gather cannot use the popcount kernel** —
`remax/core.py` and `bench/results/QUERY_PATH_SPEED.md` put asymmetric scoring
at 38–45x a Hamming scan for exactly that reason. The "family model predicts
ADC is gather-bound" note below is therefore a restatement, not a prediction.

Externally, the kernel is textbook: faiss's `HammingComputer32` is literally
four `uint64`s XORed and popcounted, and what gcc emits at `-march=x86-64-v3`
is Muła/Kurz/Lemire's AVX2 Harley-Seal. Bit planes are bit-slicing, a database
indexing technique long predating this.

`hamkern.c` is kept because a standalone kernel with no ctypes cache and no
fallback isolates kernel cost cleanly, which is what the roofline table needs.
It should not be shipped; `_native.py` is the implementation.

**How this happened:** `experiments/CLAUDE.md` mandates an account-wide `xr`
check before building, precisely to prevent this. It was skipped. Run
afterwards it answers immediately:

```
$ xr.py "hamming distance popcount kernel in C over packed binary codes"
remax/src/remax/packing.py:1          <- the dispatcher
remax_kb/remax_kb/_hamming.py:1
...
remax/src/remax/_native.py:271        <- the kernel

$ xr.py -r remax "SIMD popcount hamming scan native C extension"
remax/src/remax/packing.py:1
remax/src/remax/_native.py:91         <- rank 2 when scoped
remax/CHANGELOG.md:21
remax/tests/test_native.py:1
```

A first attempt raised `ModuleNotFoundError: remex` and was written up here as
"`xr` is unavailable in this container" — an infrastructure finding — instead
of being fixed with `pip install --break-system-packages remex onnxruntime
tokenizers`, which takes well under a minute (`remex` is on PyPI, published by
the same author). Warm calls are 175 ms. **An ImportError in a mandated check
is a missing dependency, not a broken check.** The wrong diagnosis reached
`METHODS.md`, this file, and a PR body before it was caught. Corrected in both
places; `mcp__github__search_code user:oaustegard` remains the no-local-index
fallback.

Two things fall out that are worth more than the retraction:

1. **`.sum(axis=1)` over a narrow inner axis is the whole slowdown.**
   `np.bitwise_count` runs at 14.7 GB/s; `.sum(axis=1)` over a 4-wide axis runs
   at **1.9 GB/s** — 62% of the kernel. Storing the W words as W contiguous
   columns (bit planes) turns the row reduction into W−1 whole-array adds and
   recovers **5.2x** at k=256. Pure numpy, no compiled dependency.
2. **The correction in `dab41dd6` refuted a claim measured in a different
   configuration.** `remax-hamming-speedup` ships d=512 k=4 → 2048-bit codes,
   **32 words/row**. `dab41dd6` measured k=256 → **4 words/row**. The narrow
   reduction is ~3x worse per byte (1.20 vs 3.33 GB/s), so the pathology that
   produced the crossover is specific to the narrow config. In the shipped
   config the original METHODS.md claim reproduces here.

## The generative move

`generative-thinking`, **family traversal**, after two revisions had converged
on "the gate is the kernel" and stopped moving. All three instances on the
table (ADC gather, sign-bit popcount, float32 BLAS) are scans:

```
    t(n) = a + (bytes_per_candidate × n) / (bandwidth × efficiency)
```

Walking to the limit: `n` occurs in exactly one place, as a multiplier on the
linear term, identically for every member. **Two members can cross only if at
least one has `a > 0`.** That reorganises the question — it is not about scale,
it is about a constant term wearing scale's clothes — and it is checkable
against `dab41dd6`'s own table, which is what `fit.py` does.

An earlier attempt labelled `inversion` was written over a result already in
hand and did not fire; it is recorded here because the failure mode (retrofit
a move onto ordinary engineering suspicion, then keep the label) is the one
worth catching.

## Setup

k=256 centered sign bits packed to uint64 (32 B/vec) unless stated, float32
`(n,d) @ q` baseline, single-threaded (`OMP_NUM_THREADS=1`), numpy 2.4.6. Warm
rows are min-of-9; cold rows are median-of-9 with a 300 MB scratch stream
evicting L3 between reps. 4 vCPU Xeon @2.10GHz, AVX-512 incl. VPOPCNTDQ, 260 MB
L3, 16 GB RAM. `dab41dd6` measured on a 1 vCPU container.

`C vpopcnt` is the identical arithmetic in `hamkern.c` at `-O3 -march=native`
(objdump confirms 47 `vpopcnt`) — a reimplementation of `remax/_native.py`, see
Prior art. `xor read` is a pure streaming read of the same
buffer — achievable bandwidth *at that residency*, a cache roofline at small n,
not a DRAM roofline.

## 1. The kernel gap (`roofline.py`)

Throughput is over each kernel's own bytes: two streaming kernels both at
bandwidth differ in wall time by exactly their storage ratio.

| n | f32 sgemv | np popcount | C vpopcnt | xor read |
|--:|--:|--:|--:|--:|
| 5,000 | 0.211 ms / 24.3 GB/s | 0.115 / 1.4 (1.84x) | **0.007 / 24.0 (31.6x)** | 0.003 / 62.2 |
| 42,500 | 1.950 / 22.3 | 1.152 / 1.2 (1.69x) | **0.052 / 26.0 (37.3x)** | 0.013 / 103.4 |
| 250,000 | 19.859 / 12.9 | 7.087 / 1.1 (2.80x) | **0.395 / 20.2 (50.2x)** | 0.309 / 25.9 |
| 1,000,000 | 77.869 / 13.2 | 33.329 / 1.0 (2.34x) | **1.456 / 22.0 (53.5x)** | 1.189 / 26.9 |

sgemv runs at bandwidth; the numpy expression runs ~20x under it. A crossover
between a saturated kernel and a 20x-off kernel separated by 32x in bytes lands
in the low hundreds of thousands. Descending to where per-call overhead should
decide it (`steelman.py`): n=100 → 3.19x, n=1,000 → 9.21x, n=5,000 → 33.19x.
No crossover two decimal orders below the claimed gate.

## 2. Where the time goes, and the layout fix (`layout.py`)

n=42,500, 1.36 MB of codes:

```
  xor only                0.283 ms      4.8 GB/s
  bitwise_count only      0.093 ms     14.7 GB/s
  sum(axis=1) only        0.703 ms      1.9 GB/s   <-- 62% of the kernel
  full expression         1.142 ms      1.2 GB/s
```

Nothing about bit-packing is slow; a reduction *shape* is slow. Bit planes:

| config | n | naive u64 expr | SoA bit planes | SoA/naive |
|---|--:|--:|--:|--:|
| k=256, 4 words/row, 32x | 42,500 | 1.129 ms (1.76x vs BLAS) | **0.219 ms (9.08x)** | **5.16x** |
| k=256, 4 words/row, 32x | 250,000 | 6.975 (2.92x) | **2.011 (10.15x)** | 3.47x |
| d=512 k=4, 32 words/row, 12x | 10,000 | 0.769 (1.59x) | **0.391 (3.12x)** | 1.97x |
| d=512 k=4, 32 words/row, 12x | 50,000 | 4.716 (2.58x) | **1.794 (6.78x)** | 2.63x |
| d=512 k=4, 32 words/row, 12x | 250,000 | 48.173 (1.20x) | **20.023 (2.89x)** | 2.41x |

The 32-word naive kernel sustains 3.33 GB/s against the 4-word kernel's
1.20 GB/s — the per-row reduction overhead is the same either way and there is
8x more payload to hide it behind. The 50k row reproduces
`remax-hamming-speedup`'s published 2.43x at 2.58x.

Run-to-run variance on this container is real: repeats put SoA/naive at k=256
n=42,500 between **5.2x and 6.6x**, and the BLAS baseline between 1.66 and
1.99 ms. The naive and SoA GB/s figures are stable to ~5%; the ratios inherit
the baseline's noise. Nothing here turns on a difference smaller than 2x.

Reproduce: `gcc -O3 -march=native -funroll-loops -shared -fPIC -o hamkern.so
hamkern.c` (needed by `roofline.py` and `steelman.py`; `arms.py` builds its own
per-`-march` variants), then run each script with `OMP_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1`.

## 3. Both disconfirming arms failed to disconfirm (`arms.py`)

The `challenging` pass (analysis profile, self path, verdict REVISE) named
VPOPCNTDQ as the load-bearing feature and cold cache as the untested regime.
Same C source at three `-march` levels, BLAS keeping AVX-512 in every row:

| n=42,500 | warm | cold (L3 evicted per rep) |
|---|--:|--:|
| f32 sgemv | 1.941 ms — 1.00x | 4.181 ms — 1.00x |
| C v4+vpopcnt `[47 vpopcnt]` | 0.052 — **37.02x** | 0.215 — **19.47x** |
| C v3 AVX2 `[0 vpopcnt]` | 0.056 — 34.68x | 0.247 — 16.92x |
| C v2 SSE4.2 `[0 vpopcnt]` | 0.056 — 34.78x | 0.232 — 18.00x |
| np popcount | 1.115 — 1.74x | 1.485 — 2.82x |

`[n]` = `vpopcnt` instructions in the object. Zero at v3/v2 — gcc emits a
table/Harley-Seal popcount — and it costs **6%**. The ISA hypothesis is dead;
so is any attribution of the original result to the 1 vCPU container's
microarchitecture. Cold costs the compressed side proportionally more (fewer
bytes to amortise DRAM latency over): 37x → 19x, never approaching 1.0.

## 4. Batching — the real narrowing (`steelman.py`)

Multi-query turns sgemv into GEMM, which amortises the corpus read and goes
compute-bound, while a naive per-query Hamming scan re-streams the codes.
n=42,500, ms/query: batch 1 → 1.9414 vs 0.0542 (35.8x); batch 8 → 1.0637 vs
0.0550 (19.3x); batch 64 → 0.3212 vs 0.0615 (5.2x); batch 1024 → 0.2509 vs
0.0594 (**4.2x**). BLAS gains 7.7x from batching and still loses. 4.2x is the
pessimistic bound — the Hamming side is unblocked.

## 5. 100M measured directly — and top-k, not the scan, is the risk

3.2 GB of codes fits in this container's 16 GB, so the blog post's figure needs
no extrapolation at all. `dab41dd6` raised a latency worry against
*"single-threaded brute force returns top-100 candidates in well under a
second"* on the strength of a ~4.7 s/query estimate derived from the numpy
kernel. Measured, single-threaded, `hamkern.c`:

| n | codes | scan (best of 5) | GB/s |
|--:|--:|--:|--:|
| 16,000,000 | 0.51 GB | 47.0 ms | 10.9 |
| 100,000,000 | 3.20 GB | 298–316 ms | 10.7 |

10.7 GB/s over 3.2 GB is the DRAM-streaming regime, between the L2-resident
22–26 GB/s and the L3-evicted 6.9 GB/s measured in § 1 and § 3. The **scan** is
comfortably sub-second.

Selection is not:

```
n=100,000,000, warm, single-threaded
  scan                                316.0 ms
  + np.argpartition top-100          +713.7 ms  ->  1.030 s   <- over budget
  + counting-sort top-100            +707.9 ms  ->  1.024 s   <- no better
  fused scan+threshold, one pass      394.8 ms  ->  0.395 s   <- 2.6x
```

The counting sort does not help because it still makes two more full passes
over the 200 MB score array (`bincount`, then `flatnonzero`); the pass count is
what costs, not the comparison strategy. The fix is not a better top-k, it is
**not materialising the score array at all** — `ham_scan_thresh` emits candidate
ids under a distance threshold in the same pass as the scan, and never writes
100M scores.

That path returns an **exact** top-k, verified at n=20M: the sorted distance
multiset of the fused top-100 is identical to `np.argpartition`'s. The *index*
sets differ, legitimately — 45 rows tie at the boundary distance, so which 100
you name is arbitrary and both answers are correct.

**So the blog claim holds, conditionally.** True for the scan (316 ms) and for
a fused implementation (395 ms); **false** for scan + naive numpy top-k
(1.03 s). `dab41dd6` was wrong about the stage but right that there was a risk
— and the risk is real for anything that does the obvious thing.

*Methodology note, because it nearly shipped:* the first run of this measured
`np.argpartition` at **5391 ms** — a single unwarmed call, dominated by page
faults on the freshly-written score array and the 800 MB index allocation.
Warm it is 714 ms, 7.6x less. One cold call reported as a result is the same
error this whole experiment exists to document, made once more at the last
step.

## What to change

- **Retract** "compression is SCALE-GATED and below ~150k rows its win is
  negative" (`dab41dd6`).
- **Do not restore** `README.md`'s "beats BLAS float cosine at every N"
  unscoped — it is unscoped in the same way its correction was. State the two
  measured regimes and the configuration each belongs to.
- **Stop attributing the original result to the 1 vCPU container.** Fitted
  per-row costs agree to 5%. It is a 4.1 ms constant, not a slower core, not a
  smaller cache, not a missing instruction.
- **Wire `remax_kb` to the native kernel it already depends on.**
  `remax_kb/_hamming.py` imports `from remax.packing import stable_top_k`, so
  `remax` is already a hard dependency — but `_popcount_rows` implements the
  NumPy path inline and never calls `remax.packing.hamming_distances`, which
  would dispatch to `_native`. The compiled kernel exists, ships, falls back
  cleanly, and is one import away from the shipped scan. This is a wiring gap,
  not a research finding, and it is the highest-value item here.
- **Ship the bit-plane layout** where a compiled kernel is genuinely
  unavailable — the `js/kb-reader.js` browser/Worker reader, and any NumPy-only
  fallback: 2.4x at the shipped config, 5.2–6.6x at k=256, no compiler. Note
  the limit: bit planes win on *full* scans and lose on filtered subsets (W
  scattered accesses per row instead of one), so this is a full-scan
  optimisation.
- **Retire "a compiled SIMD kernel is unnecessary."** That ruling in
  `remax-hamming-speedup` was made against the NumPy idiom at 1.7x over BLAS.
  The kernel it declared unnecessary was already written, in `remax`, at 28x
  per that repo's own CHANGELOG.
- **The 100M figure is now measured, not extrapolated — and the bottleneck is
  top-k, not the scan.** See § 5.

## Reusable

- **`fit.py` — fit `a + b·n` before reporting any crossover.** If `a ≈ 0` on
  both sides, a reported crossover is an artifact of the two `n` values that
  bracket it. If `a > 0`, the constant is the finding and it belongs to the
  harness, not the corpus. `n*` is meaningful only inside the fitted range;
  the script flags extrapolations below it.
- **Bit-plane (SoA) layout for packed-code scans** (`layout.py::soa_vs_naive`)
  — 2–5x over the row-major `np.bitwise_count(C ^ q).sum(axis=1)` idiom that
  `METHODS.md` currently records as the fast kernel. The win scales inversely
  with code width, so it is largest exactly where the codes are smallest.

## Caveats

- One machine, one microarchitecture family. ARM/NEON untested — though with
  SSE4.2 costing 6%, the popcount instruction is not where the risk lives. The
  `omni-macos` Table 7 cross-reference in `dab41dd6` is neither confirmed nor
  refuted here.
- Single-threaded throughout. Both kernels thread; sgemv threads via a tuned
  library, the scan would need doing by hand. Real deployment cost, unmeasured.
- Warm rows are min-of-9, a best-case estimator; cold rows are median. Both
  kernels are measured identically, so the ratio is the robust quantity.
- Latency only. R@100 = 0.926 at k=256 is untouched, and the operational
  question is latency *at fixed recall* — a scan needing a float32 rerank buys
  less than the headline end-to-end.
- Index maintenance (packbits rebuild, incremental insert) is not costed.
- **`dab41dd6`'s fit is over 3 points** after dropping its DRAM-superlinear
  4M/16M tail. `a = 4.108 ms` with residuals ±0.5 ms is a good fit but a
  3-point one, and the *cause* of the 4.1 ms is not identified — only its
  existence and size.
- The family model puts ADC's 13.5x in the gather regime rather than the
  streaming one. `remax/core.py` and `QUERY_PATH_SPEED.md` already say this
  (asymmetric scoring at 38–45x a Hamming scan, "cannot use the SIMD popcount
  kernel"), so it is a restatement in different vocabulary, not a prediction.
  The bit-width leg remains untested, and nothing measured here transfers to
  the remex ADC path.
- **Novelty accounting.** New here: the constant-term reconciliation of
  `dab41dd6` with `remax-hamming-speedup` (§ fit); the `.sum(axis=1)`
  narrow-axis diagnosis, which is about the `bitwise_count` path and not the
  LUT path `_native.py`'s docstring analyses; the bit-plane numbers; the 6%
  `-march` measurement; and the `remax_kb` wiring gap. Not new: the C kernel,
  the ~30x NumPy-vs-C ratio, the cache falloff, and the gather-vs-stream
  distinction — all already in `remax`.
