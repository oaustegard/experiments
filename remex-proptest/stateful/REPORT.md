# remex area C: stateful indexes and caches (final report)

Revision under test: remex 2.2.0, unmodified. Nothing committed. Hypothesis 6.170.15.

## Run results (this container)

| Run | Result |
|---|---|
| Cold full run, `results/pytest_full_run.txt` | 54 s. 15 failed (exactly the confirmed regression tests), 28 passed, 12 skipped (mutant tests, env-gated), 6 xfailed (2 KnownBug machines + 4 ambiguous). |
| Machines in that run | RemexIndexMachine 60 passing, CacheMachine 60, GpuIvfMachine 60; rht thread-history 40; encode block-size 40; chunk-boundary tests pass. |
| Mutation calibration, `results/mutation_results.{md,json}` | Baseline clean on all three machines (153-156 examples). 16 of 17 mutants caught; M13 survived (see below). |
| 500 x 40 exploratory, `results/exploratory_500.txt` | RemexIndexMachine and CacheMachine pass (500 examples x 40 steps, no shrinking). GpuIvfMachine failed once, a harness false positive (below). |

Coverage events are in `results/coverage_events.json`. Examples: precision_switch 264, xhat_warm_at_call 108 / cold 419, search_after_drop_cache 52, memo_repeat_checked 90, adds_between_searches 1/2/3 = 358/228/373, guard_bits_warm 146, warm_then_derive 42. Weak spots: gpuivf:gpu_xhat_warm only 8 and cpu_xhat_warm 4.

## Mutants

| id | defect | caught after (examples) |
|---|---|---|
| M01 | add skips `_decoded` reset | 4 |
| M02 | add ids from 0 | 4 |
| M03 | mean re-derived per batch | 4 |
| M04 | load drops mean | 4 |
| M05 | reduceat offsets not rebased | 19 |
| M06 | drop_cache leaves `_dir_len` (answer-equivalent; structural check only) | 10 |
| M07 | subset copies whole cache | 4 |
| M08 | `_dir_len` key ignores precision | 6 |
| M09 | R setter keeps RHT operator | 12 |
| M10 | x_hat cache served at any precision | 6 |
| M11 | subset skips mean_dots | 4 |
| M12 | bias key ignores precision | 82 |
| M13 | GPU norms cache ignores precision | SURVIVED 153 (caught at 55 in an earlier run) |
| M14 | GPU drop_cache no-op (answer-equivalent; structural) | 6 |
| M15 | IVF tie order | 10 |
| M16 | IVF drops last row per cell | 4 |
| M17 | guards skipped on warm cache | 50 |

M13 survival is a sensitivity gap: GPU searches on warm state at differing precisions are drawn too rarely (gpu_xhat_warm = 8). A rule that cycles GPU search_adc across precisions would fix it; not done.

## Harness false positive (500-example pass)

`GPUSearcher.search_batch` failed the 2e-4-relative score tolerance on a centred, 8-bit, `scaled` corpus (norms 176-2746, scores up to 513). The error is 0.064 against a 0.055 tolerance. `Quantizer.search_batch` has the identical error against the float64 oracle, so it is float32 cancellation in the centred estimator, shared by CPU and GPU, not a GPU defect. The tolerance in `_oracle.tolerance` should be loosened for centred/large-norm data; not changed here.

## Findings (ordered by severity; IDs match test names)

- **F4 (high; defect confirmed, contract ambiguous).** A default `Quantizer(rotation="rht")` pickled before first use segfaults (rc -11) when encoding in a new process. deepcopy then freeing the original and reusing memory segfaults (rc 139). After first use, deepcopy and pickle raise (ctypes handle). Root cause: `rotation.py:403-405` stores raw `array.ctypes.data` ints, and `rotation.py:392/445-450` cache the CDLL. Fix: `RHTOperator.__reduce__ -> (RHTOperator, (d, seed))`. Tests: `test_f4_*`.
- **F1 (medium, confirmed).** `RemexIndex(bits=4).add([])` raises ZeroDivisionError and leaves `dim=0`, so every later add fails. Same for a MaxSim batch with mixed d (dim stays set). Root cause: `st.py:304-308` (`_check_dim` commits before acceptance), `st.py:310-319` (`[]` becomes (1,0) before the emptiness test), `st.py:321-336`. Tests: `test_f1_*`.
- **F2 (medium, confirmed).** `Quantizer.search` on a warm cache skips `_check_bits`/`_check_mean` when `renorm=False` or in scalar mode. A centred container is then scored without its mean (about 14 vs 31). Root cause: `core.py:1665-1672` (cache hit bypasses `_resolve_centroids`; `_effective_norms` returns early at 1406). Fix: add the two checks as `search_batch` does (`core.py:1868-1871`). Tests: `test_f2_*`.
- **F8 (medium, ambiguous: thread-safety undocumented).** A search straddling an `add` publishes a stale `_decoded` (`st.py:464-466`); later searches raise IndexError. Simulated, not real threads. Test: `test_f8_*` (xfail).
- **F3 (low-medium, confirmed inconsistency).** `RemexIndex.load` uses `np.load` (`st.py:534`), not `open_npz`: suffix-less path not found, damaged files give raw EOFError/BadZipFile/pickle advice, contrary to CHANGELOG 2.1.0. Tests: `test_f3_*`.
- **F5 (low).** `IVFCoarseIndex.__init__` (`ivf.py:94-137`) runs no guards; bit-width mismatch gives an obscure cumsum error or a garbage index. Queries still refuse. Fix: call `quantizer._resolve_centroids(compressed, None)`. Tests: `test_f5_*`.
- **F6 (low, ambiguous).** `q.R = M` on a `rotation="none"` quantizer is ignored by CPU (`core.py:1166`) but read by GPUSearcher (`gpu.py:299`).
- **F7 (low, ambiguous).** `resident_bytes` ignores `_dir_len`; `PackedVectors` says "no caches" yet ADC fills it and it has no `drop_cache`.

Fixes F1, F2, F3, F4, F5 were validated earlier by monkeypatch (21 regression tests passed with them). The prototype plugin lived in the old scratchpad and is gone; not re-validated after the container restart.

## Blind spots

- Oracle shares production's normalisation, Lloyd-Max tables, `_cosine_factor`, dense `R`, `corpus_mean`.
- 2e-4 relative score tolerance hides smaller systematic errors; only codes and repeated answers are bit-exact.
- Not covered: real sentence-transformers, cupy/torch backends, real threads, NaN/inf inputs, in-place mutation of container arrays.
- Small n and d (n <= 40, d <= 24); 4096-row chunking covered only by the plain boundary tests.
- GPU warm-state sampling is thin (M13).
