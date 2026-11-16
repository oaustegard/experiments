# Mutation calibration, area A (search-path agreement)

Harness `test_search_props.py`, 200 examples per property, pristine-failing properties deselected (test_mismatched_reader_is_refused_on_every_path, test_ivf_flat_limit_is_byte_identical_to_adc, test_score_vectors_accepts_packed, test_query_offset_cannot_reorder).

| mutant | planted defect | caught | by |
|---|---|---|---|
| `adc_skips_effective_norms` | search_adc multiplies by raw stored norms (skips _effective_norms/_score_norms) | yes | test_batch_equals_single_and_permutation, test_cold_cache_each_path, test_cpu_paths_match_oracle, test_derived_containers_search_like_their_rows, test_gpu_numpy_backend_matches, test_ivf_flat_limit_matches_oracle_and_flat_paths, test_large_n_chunk_boundaries, test_renorm_is_a_reader_property_and_caches_do_not_leak |
| `search_drops_query_offset` | Quantizer.search forgets _query_offset (centred scores lose q.mu) | yes | test_batch_equals_single_and_permutation, test_cold_cache_each_path, test_cpu_paths_match_oracle, test_derived_containers_search_like_their_rows, test_gpu_numpy_backend_matches, test_large_n_chunk_boundaries |
| `ivf_coarse_drops_query_offset` | IVFCoarseIndex.search_coarse forgets _query_offset | yes | test_ivf_flat_limit_matches_oracle_and_flat_paths, test_ivf_partial_probe_sound, test_large_n_chunk_boundaries |
| `gpu_batch_drops_query_offset` | GPUSearcher.search_batch forgets the centred q.mu offset | yes | test_gpu_numpy_backend_matches |
| `batch_drops_row_bias` | Quantizer.search_batch forgets _row_bias (centred rows lose mean_dots term) | yes | test_batch_equals_single_and_permutation |
| `twostage_fine_uses_coarse_norms` | search_twostage reranks with the coarse-precision norms (per-precision confusion) | yes | test_cold_cache_each_path, test_cpu_paths_match_oracle, test_ivf_flat_limit_matches_oracle_and_flat_paths, test_large_n_chunk_boundaries |
| `dir_len_cache_ignores_precision` | _direction_lengths caches under one key for every precision (stale across levels) | yes | test_cold_cache_each_path, test_cpu_paths_match_oracle, test_derived_containers_search_like_their_rows, test_gpu_numpy_backend_matches, test_ivf_flat_limit_matches_oracle_and_flat_paths, test_ivf_partial_probe_sound, test_large_n_chunk_boundaries |
| `adc_chunk_boundary_norm_shift` | _adc_score_chunked takes norms from the previous row at every chunk after the first | yes | test_adc_scores_independent_of_chunking_and_position, test_cold_cache_each_path, test_cpu_paths_match_oracle, test_large_n_chunk_boundaries |
| `packed_adc_ignores_precision_shift` | search_adc on PackedVectors forgets to right-shift for lower precision | yes | test_adc_scores_independent_of_chunking_and_position, test_batch_equals_single_and_permutation, test_cold_cache_each_path, test_cpu_paths_match_oracle, test_derived_containers_search_like_their_rows, test_ivf_flat_limit_matches_oracle_and_flat_paths, test_large_n_chunk_boundaries, test_serialized_containers_search_identically |
| `twostage_returns_candidate_relative_ids` | search_twostage returns positions in the candidate set instead of corpus rows | yes | test_cold_cache_each_path, test_cpu_paths_match_oracle, test_ivf_flat_limit_matches_oracle_and_flat_paths, test_large_n_chunk_boundaries |
| `ivf_twostage_drops_bias` | IVFCoarseIndex.search_twostage rerank omits the centred row bias | yes | test_ivf_flat_limit_matches_oracle_and_flat_paths, test_ivf_partial_probe_sound |
| `gpu_adc_full_precision_norms` | GPUSearcher.search_adc scales lower-precision scores with full-precision norms | yes | test_gpu_numpy_backend_matches, test_large_n_chunk_boundaries, test_renorm_is_a_reader_property_and_caches_do_not_leak |
| `search_k_equals_n_boundary` | Quantizer.search: `k >= n` becomes `k > n`, so k == n reaches argpartition(kth=n) | yes | test_batch_equals_single_and_permutation, test_cold_cache_each_path, test_cpu_paths_match_oracle, test_derived_containers_search_like_their_rows, test_gpu_numpy_backend_matches, test_precision_equal_to_bits_is_full_precision, test_renorm_is_a_reader_property_and_caches_do_not_leak |
| `ivf_probe_tiebreak_reversed` | IVFCoarseIndex.probe_cells breaks Hamming ties by descending cell id | yes | test_ivf_partial_probe_sound |
| `ivf_flat_ignores_precision` | IVFCoarseIndex.search_coarse scores with the full-precision table but shifted codes | yes | test_ivf_flat_limit_matches_oracle_and_flat_paths, test_ivf_partial_probe_sound, test_large_n_chunk_boundaries |

**15/15 mutants caught.**

## First violation per mutant

- `adc_skips_effective_norms` (419s): search_adc: score np.float32(0.5388164) for id 0 vs oracle np.float64(0.6753061491701295) (err 1.365e-01 > tol 2.094e-06)
- `search_drops_query_offset` (231s): search: score np.float32(0.26442108) for id 0 vs oracle np.float64(0.6753061491701295) (err 4.109e-01 > tol 2.608e-06)
- `ivf_coarse_drops_query_offset` (210s): ivf.search_coarse[flat]: score np.float32(0.26442108) for id 0 vs oracle np.float64(0.6753061491701295) (err 4.109e-01 > tol 2.608e-06)
- `gpu_batch_drops_query_offset` (139s): gpu.search_batch[0]: score np.float32(0.26442108) for id 0 vs oracle np.float64(0.6753061491701295) (err 4.109e-01 > tol 2.608e-06)
- `batch_drops_row_bias` (157s): search_batch vs search: top-k sets differ beyond a tie: [0] vs [4]  case={'mode': 'centred', 'd': 2, 'bits': 1, 'rotation': 'rht', 'n': 5, 'k': 1, 'm': 1}
- `twostage_fine_uses_coarse_norms` (118s): twostage_part: fine score err 6.031e-01 > 2.094e-06
- `dir_len_cache_ignores_precision` (242s): packed_twostage_part: returned rows outside the stage-1 candidate set [11]
- `adc_chunk_boundary_norm_shift` (158s): twostage_part: returned rows outside the stage-1 candidate set [2]
- `packed_adc_ignores_precision_shift` (208s): perm:packed_adc: score np.float32(0.63587964) for id 0 vs oracle np.float64(2.5037540403641723) (err 1.868e+00 > tol 4.850e-06)  case={'mode': 'unit', 'd': 2, 'bits': 2, 'rotation': 'rht', 'n': 1, 'k': 1, 'm': 1}
- `twostage_returns_candidate_relative_ids` (104s): twostage_part: fine score err 6.170e-01 > 2.094e-06
- `ivf_twostage_drops_bias` (106s): ivf.search_twostage[flat]: score np.float32(3.0226388) for id 0 vs oracle np.float64(2.2428838857329105) (err 7.798e-01 > tol 1.021e-05)
- `gpu_adc_full_precision_norms` (104s): gpu.search_adc: score np.float32(0.35673338) for id 0 vs oracle np.float64(0.6753061491701295) (err 3.186e-01 > tol 2.094e-06)
- `search_k_equals_n_boundary` (114s): ValueError: kth(=1) out of bounds (1)
- `ivf_probe_tiebreak_reversed` (82s): candidate_indices differ from independent Hamming pool: [0] vs []  case={'mode': 'scalar', 'd': 96, 'bits': 1, 'rotation': 'rht', 'n': 1, 'k': 0, 'm': 1} nprobe=6 qcell=1
- `ivf_flat_ignores_precision` (164s): ivf.search_coarse[flat]: score np.float32(-0.093020275) for id 0 vs oracle np.float64(0.16391942978819962) (err 2.569e-01 > tol 1.638e-06)
