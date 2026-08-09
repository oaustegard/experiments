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
(objdump confirms 47 `vpopcnt`). `xor read` is a pure streaming read of the same
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

## What to change

- **Retract** "compression is SCALE-GATED and below ~150k rows its win is
  negative" (`dab41dd6`).
- **Do not restore** `README.md`'s "beats BLAS float cosine at every N"
  unscoped — it is unscoped in the same way its correction was. State the two
  measured regimes and the configuration each belongs to.
- **Stop attributing the original result to the 1 vCPU container.** Fitted
  per-row costs agree to 5%. It is a 4.1 ms constant, not a slower core, not a
  smaller cache, not a missing instruction.
- **Ship the bit-plane layout** on the remax scan path: 2.4x at the shipped
  config, 5.2x at k=256, pure numpy.
- **Re-measure, don't revise, the 100M figure.** `dab41dd6`'s ~4.7 s/query
  single-core extrapolation came from the numpy kernel; candidates here span
  ~150 ms (warm 22 GB/s) to ~460 ms (cold 6.9 GB/s) over 3.2 GB. Three values
  an order apart is a reason to measure at 16M and extrapolate once. This is
  the number the blog-post latency worry rests on.

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
- The family model predicts ADC's 13.5x is gather latency (a cache line and a
  dependent load per subquantizer per candidate), hence flat in `n` and
  insensitive to bit width. Flat in `n` matches `901e3c06`. **The bit-width leg
  is untested** — this is a prediction, not a result, and nothing here
  transfers to the remex ADC path.
