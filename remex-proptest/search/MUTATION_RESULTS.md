# Mutation calibration, area A (search-path agreement)

Harness `test_search_props.py`, 200 examples per property, pristine-failing properties deselected (test_ivf_flat_limit_is_byte_identical_to_adc, test_score_vectors_accepts_packed, test_query_offset_cannot_reorder).

| mutant | planted defect | caught | by |
|---|---|---|---|
| `adc_skips_effective_norms` | search_adc multiplies by raw stored norms (skips _effective_norms/_score_norms) | yes | test_batch_equals_single_and_permutation, test_cold_cache_each_path, test_cpu_paths_match_oracle, test_derived_containers_search_like_their_rows, test_gpu_numpy_backend_matches, test_ivf_flat_limit_matches_oracle_and_flat_paths |
| `search_drops_query_offset` | Quantizer.search forgets _query_offset (centred scores lose q.mu) | yes | test_batch_equals_single_and_permutation, test_cold_cache_each_path, test_cpu_paths_match_oracle, test_derived_containers_search_like_their_rows, test_gpu_numpy_backend_matches |

**2/2 mutants caught.**

## First violation per mutant

- `adc_skips_effective_norms` (139s): search_adc: score np.float32(0.5388164) for id 0 vs oracle np.float64(0.6753061491701295) (err 1.365e-01 > tol 3.351e-05)
- `search_drops_query_offset` (96s): search: score np.float32(0.26442108) for id 0 vs oracle np.float64(0.6753061491701295) (err 4.109e-01 > tol 4.172e-05)
