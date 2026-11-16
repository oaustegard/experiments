# remex area A: search-path agreement. Report

Target: remex 2.2.0 at /home/user/remex (HEAD b8d527a), unmodified. Nothing committed.
Files (all in this directory): `test_search_props.py` (Hypothesis harness), `oracle.py` (float64 oracles),
`test_search_regressions.py`, `mutants.py` + `mutation_calibration.py` + `MUTATION_RESULTS.md`,
`coverage_report.py` + `COVERAGE.md` + `hypothesis_statistics.txt`.
Run: `pytest -q test_search_props.py -p no:cacheprovider` (about 2m40s at the default 400 examples per property).

## Findings, by severity

### F1 (Medium, confirmed). `Quantizer.search` skips the bits/mean guards once the container cache is warm
- Contract: README.md:23 ("decoding against a quantizer that differs ... raises"); `_check_bits` (core.py:1261); `_resolve_centroids` docstring (core.py:2019, "the one place the invariant has to hold"); the comment at core.py:1408 about cached arrays skipping the guards.
- Repro: writer `Quantizer(8, bits=4, renorm=False)` encodes, then `search` once. Then `Quantizer(8, bits=2, renorm=False).search(cv, q)` returns scores. On a cold container it raises "bits mismatch". The same holds for scalar mode (bits) and for an uncentred `renorm=False` reader of centred codes (mean). In that last case the scores silently ignore the mean.
- Regression: `test_F1_search_refuses_a_mismatched_reader_after_the_cache_is_warm[scalar_bits|renorm_off_bits|renorm_off_mean_dropped]`. The cold controls `test_F1_control_...` pass.
- Root cause: `search` (core.py:1665) calls only `_check_rotation`. `_get_x_hat_rot` returns its cache before `_resolve_centroids` (core.py:2005). `_effective_norms` returns early (core.py:1406) before its guards (1408-1412) for scalar mode and `renorm=False`. `search_batch` has explicit checks (1868-1871). `search_adc`, `search_twostage`, IVF and GPU all refuse.
- Fix sketch: add `self._check_bits(compressed); self._check_mean(compressed)` to `search`. Validated on a scratch copy: the regressions pass, and `test_mismatched_reader_is_refused_on_every_path` passes at 1500 examples (no other path leaks).
- Detection note: the property needs a warm cache plus a scalar or `renorm=False` writer, so it is rare. It is set to at least 1500 examples. It failed in 5 of 6 runs at 400 examples and in 3 of 3 runs at 1500.

### F2 (Low, confirmed). `_score_vectors(PackedVectors)` raises AttributeError unless the codes are centred
- Contract: the explicit Packed branch (core.py:1620); `tests/test_v2_fixes.py::test_score_vectors_accepts_packed`, which only uses centred codes.
- Repro: `pq._score_vectors(PackedVectors.from_compressed(pq.encode(X)))` raises `'PackedVectors' object has no attribute 'indices'`.
- Regression: `test_F2_score_vectors_accepts_packed_for_every_mode[uncentred|uncentred_norenorm|scalar]`.
- Root cause: core.py:1617-1618 calls `decode`, which calls `_resolve_indices` (core.py:2043/2045), and that reads `.indices`.
- Fix: make `_resolve_indices` unpack PackedVectors. Validated on the scratch copy.
- Variants, documented as CompressedVectors-only but with an unhelpful AttributeError: `decode(packed)` and `GPUSearcher(pq, packed)`. `search` and `search_batch` raise a proper TypeError.

### F3 (Low, confirmed). IVF at nprobe = 2**n_bits is not byte-identical to the flat paths for centred codes
- Contract: README.md:395-398 ("byte-identical agreement with search_adc / search_twostage") and the "bit-identical to flat" table row. `tests/test_ivf.py` only compares ids, and only for uncentred codes.
- Repro: centred, d=8, bits=2. Scores differ by 1-2 ulp in nearly every query. The top-10 id lists matched in 200 of 200 queries.
- Regression: `test_F3_ivf_flat_scan_is_byte_identical_to_search_adc_for_centred_codes` and `..._twostage_...`. The uncentred control passes.
- Root cause: ivf.py:374-379 and 447-454 add the offset and then the bias. core.py:1732-1733 and 1826-1829 add the bias and then the offset.
- Fix: reorder the additions. Validated: the property passes at 500 examples.

### F4 (Low, confirmed, extreme regime). The constant q.mu is added in float32 before top-k selection and can reorder rows
- Contract: `_query_offset` docstring (core.py:1337-1346): "cannot reorder anything".
- Repro: d=4, bits=8, mean=[1e6,0,0,0], two rows, q ≈ mu. q.mu is about 1e12 and the float32 ulp is 65536. The float64 oracle and the exact unquantized q.x agree that row 1 wins by 3.9e4. The production scores tie, and row 0 is returned.
- Regression: `test_F4_constant_query_offset_cannot_reorder_rows`. Property: `test_query_offset_cannot_reorder`.
- Sites: core.py:1672, 1733, 1829, 1887-1890; ivf.py:374, 451; gpu.py:375, 406, 454, 489.
- Fix sketch: select top-k on the offset-free scores, then add the offset to the returned values. This is only validated on `search_adc`. It needs ||mu||/||residual|| of roughly 1e5 or more, so practical impact is small.

### F5 (Low, confirmed). `IVFCoarseIndex.candidate_indices(nprobe=1)` returns a view of `sorted_idx`
- Repro: `c = ivf.candidate_indices(cell, 1); c[:] = 0` corrupts the index. Two or more cells return a copy (ivf.py:274-283).
- Regression: `test_F5_candidate_indices_does_not_alias_the_index`.
- Fix: `.copy()`. Validated.

### Ambiguous (tests named `test_AMBIGUOUS_*`; they fail by design, and no fix was prototyped)
- A1: `PackedVectors.at_precision(5..7)` on an 8-bit container passes its own range check, then raises "Bit width 5 is not supported" (core.py:605 → packing.py:144). `search(precision=5)` works, and the docstring says 1..bits. Either the docstring or the early check should change.
- A2: for centred codes, IVF hashes the raw query but the corpus hash is the residual direction (ivf.py:232-252 vs 170-187). A stored vector queried as itself lands in its own cell 5-22% of the time, against 100% uncentred. Hashing `q - mu` makes it 100%, but it did not clearly improve recall in a quick test (0.31 → 0.36). Recall quality, not a flat-scan violation.
- A3 (doc gap, no test): `Quantizer.search_twostage` returns min(k, candidates) rows. The IVF docstring says so, the Quantizer one does not.

## Properties that passed
Default budget is 400 Hypothesis examples per property. A separate run at 6000 examples per property (about 78,000 cases, run against the pre-F1 version of the harness, so it excludes the mismatched-reader property) found no failure.
- `cpu_paths_match_oracle`: operation scripts of 1-8 steps on one warming container (`search`, `search_adc`, `search_batch`, `search_twostage` full and partial, packed variants, `_score_vectors`, `decode`, `drop_cache`), checked against the oracle.
- `cold_cache_each_path`
- `gpu_numpy_backend_matches`: against the oracle and against the CPU paths.
- `ivf_flat_limit_matches_oracle_and_flat_paths`
- `ivf_partial_probe_sound`: the Hamming pool is re-derived independently.
- `ivf_structure`
- `batch_equals_single_and_permutation`
- `query_scaling...`
- `derived_containers_search_like_their_rows`
- `invalid_precision_rejected_uniformly`
- `empty_corpus_and_query_dtypes`
- `serialized_containers_search_identically`: npz, arrow, mmap, `from_rows`, `.pq`.
- `adc_scores_independent_of_chunking_and_position`: exact equality.
- `renorm_is_a_reader_property_and_caches_do_not_leak`
- `precision_equal_to_bits_is_full_precision`
- `large_n_chunk_boundaries` (n up to 8193), `large_n_ivf_prefix_chunk_boundary` (n=131089), `encode_block_size_does_not_change_the_container`.

Config sweeps at 250 examples, all passing: `REMEX_NO_NATIVE=1`; `REMEX_PARALLEL_MIN=1 REMEX_NUM_THREADS=3`.

Final default run: 42 passed, 3 failed. The 3 failures are the F2, F3 and F4 properties; the F1 property passed in that run (see the detection note under F1).

Event counts are in COVERAGE.md. Notable counts from that run:
- `tie_boundary` 437
- `tie_boundary_stage1` 386
- `path_vs_path_tie_difference` 14
- `identical_rows_unequal_scores_in_search(BLAS gemv)` 20

That last event is documented float noise: identical rows can differ by about 1e-6 relative in the BLAS gemv path. The oracle is checked within tolerance.

## Mutants
`MUTATION_RESULTS.md` records 15 of 15 caught on the final harness (200 examples per property). The mutants:
- `search_adc` skipping `_effective_norms`
- `_query_offset` dropped in `search`, in IVF `search_coarse`, and in GPU `search_batch`
- `search_batch` dropping the row bias
- `search_twostage` using coarse norms in the rerank
- `_direction_lengths` cache ignoring precision
- an ADC chunk-boundary norm shift
- packed ADC ignoring the precision shift
- `search_twostage` returning candidate-relative ids
- IVF `search_twostage` dropping the bias
- GPU ADC using full-precision norms
- `k == n` boundary
- IVF probe tie-break reversed
- IVF flat ignoring precision

An earlier 15/15 run on a previous version of the harness also caught all of them.

Calibration decisions:
- Float tolerance was tightened to 1e-6·(1+√d/4), about 8x above the measured worst error at small d and about 3x at d=256.
- Rows of kind `jitter` (near-duplicates) were added so near-ties are generated, not only exact ties.

## Blind spots
- The oracle shares the codebook tables, the dense `pq.R` and the stored norms/`mean_dots` with production. So `lloyd_max_codebook`, `nested_codebooks` (spot-checked to within 6e-4σ of the conditional mean), `encode`'s norm and dots computation, and the dense RHT matrix against a spec are common-mode. The operator is checked against the dense matrix.
- Torch and CuPy backends were not run (no GPU).
- `RemexIndex` / MaxSim scoring was only spot-checked (the `low_memory` ranking matched in 576 of 576 cases), with no property.
- IVF recall and hash quality are unchecked beyond the A2 observation.
- NaN and inf inputs: only a by-hand check that the paths do not crash. There is no ordering contract.
- Thread-safety of the container caches, and in-place mutation of containers after caching, are untested.
- Large d is rare: d ≤ 256 only in properties, and n > 4096 only in the fixed tests.
- Query shapes other than (d,): wrong-length queries raise ValueError, but (1,d) queries are accepted by some paths and rejected by others. This is undocumented, so it is not reported.
