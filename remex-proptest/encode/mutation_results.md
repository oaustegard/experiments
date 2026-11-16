# Mutation calibration of test_encode_props.py

Each mutant is a one-token edit of the real remex function applied by `conftest.py` (`ENCODE_MUTANT=<name>`); remex itself is untouched. `--maxfail=4`, 0.4x example budget, Hypothesis seed 0, database off. 'Caught' = at least one test failed (xfail-strict tests that start passing also count, and are listed).

| mutant | planted defect | caught | first failing tests | wall s |
|---|---|---|---|---|
| `float64_midpoints` | boundaries = midpoints of the unrounded float64 centroids (pre-fix behaviour, codebook.py:84-86) | YES | `test_negation_mirrors_codes`, `test_codebook_invariants`, `test_matryoshka_codes_nest`, `test_scalar_equal_inputs_equal_codes_and_monotone` | 150 |
| `rht_one_round` | rht_rounds returns 1 when B == d (the pre-1.0 rule behind issue #89) | YES | `test_encode_matches_nearest_centroid_oracle`, `test_rht_operator_matches_dense_and_independent_reference`, `test_rht_rounds_floor_and_plan_shape_for_every_even_d`, `test_rht_rounds_for_large_d_is_floating_point_safe` | 43 |
| `kernel_ties_go_up` | C quant kernel uses v >= b instead of v > b (ties go to the upper cell) | YES | `test_native_and_numpy_paths_give_identical_bits`, `test_negation_mirrors_codes`, `test_assign_codes_equals_searchsorted`, `test_scalar_none_codes_are_searchsorted_and_nearest_centroid` | 129 |
| `numpy_assign_side_right` | NumPy fallback of assign_codes uses side='right' | YES | `test_assign_codes_equals_searchsorted`, `test_assign_codes_exhaustive_float16_and_ulp_windows[1]`, `test_assign_codes_exhaustive_float16_and_ulp_windows[2]`, `test_assign_codes_exhaustive_float16_and_ulp_windows[3]` | 196 |
| `nested_unweighted_mean` | nested centroids use a plain mean instead of the probability-weighted mean | YES | `test_nested_codebook_invariants` | 222 |
| `length_floor_1e-3` | encode floors the norm at 1e-3 instead of 1e-8 | YES | `test_anisotropy_check_agrees_with_anisotropy_and_never_changes_codes`, `test_env_switch_remex_no_native_gives_identical_codes` | 155 |
| `thread_chunk_drops_last_row` | threaded assign_codes drops the last element of every chunk | YES | `test_assign_codes_equals_searchsorted`, `test_assign_codes_exhaustive_float16_and_ulp_windows[1]`, `test_assign_codes_exhaustive_float16_and_ulp_windows[2]`, `test_assign_codes_exhaustive_float16_and_ulp_windows[3]` | 132 |
| `scalar_encode_float32` | scalar mode casts input to float32 instead of float64 | YES | `test_scalar_none_codes_are_searchsorted_and_nearest_centroid`, `test_scalar_equal_inputs_equal_codes_and_monotone`, `test_scalar_mode_saturates_huge_finite_values_under_rotation` | 137 |
| `numpy_rht_sign_index` | NumPy RHT path indexes the sign vector with perm (s[perm]) | YES | `test_native_and_numpy_paths_give_identical_bits`, `test_rht_operator_at_embedding_sized_d` | 150 |
| `lloyd_30_iterations` | lloyd_max_codebook default n_iter 300 -> 30 | YES | `test_codebook_matches_independent_lloyd_reference`, `test_shipped_codebook_is_lloyd_optimal_up_to_4_bits[4]`, `test_scalar_none_codes_are_searchsorted_and_nearest_centroid`, `test_env_switch_remex_no_native_gives_identical_codes` | 31 |
| `encode_block_drops_row` | encode block loop drops the last row of every full block | YES | `test_anisotropy_check_agrees_with_anisotropy_and_never_changes_codes` | 132 |
| `haar_no_sign_correction` | Haar rotation without the Mezzadri sign correction | YES | `test_haar_matches_unique_qr_factor` | 129 |
| `rht_plan_draw_order` | rht_plan draws signs before permutations (operator vs dense matrix drift) | YES | `test_encode_matches_nearest_centroid_oracle`, `test_rht_operator_matches_dense_and_independent_reference`, `test_scalar_decode_returns_the_centroids`, `test_dense_matrix_path_agrees_with_operator_except_at_boundaries` | 129 |
| `scalar_boundary_float32_window` | assign_codes compares float64 haystacks in float32 | YES | `test_assign_codes_equals_searchsorted` | 132 |
| `linspace_init_asymmetric` | Lloyd-Max initial centroids linspace(-3s, 3.0001s) | YES | `test_negation_mirrors_codes`, `test_codebook_invariants`, `test_codebook_matches_independent_lloyd_reference`, `test_nested_codebook_invariants` | 55 |
| `anisotropy_uses_max_norm` | anisotropy() divides by max ||x|| instead of mean ||x|| | **NO (survived)** | - | 127 |
| `encode_ratio_drops_zero_guard` | encode's anisotropy judge loses its norm_total > 0 guard | **NO (survived)** | - | 1 |

baseline (no mutant) is the unmodified harness: all green with 8 xfails, see coverage_events.txt.
