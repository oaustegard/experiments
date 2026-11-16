# Coverage of generated cases, area A

Event counts from one pytest run of `test_search_props.py` (`REMEX_PT_EVENTS`).  Counts are per generated case for case-class labels and per executed operation for `cpu_op=` / `gpu_op=` / `roundtrip:` / `adc_chunk_invariance:` labels.

## Quantizer mode

| event | count |
|---|---:|
| `mode=unit_norenorm` | 2101 |
| `mode=unit` | 1961 |
| `mode=centred` | 1736 |
| `mode=scalar` | 1646 |

## Rotation

| event | count |
|---|---:|
| `rotation=haar` | 3573 |
| `rotation=none` | 2063 |
| `rotation=rht` | 1808 |

## Bits

| event | count |
|---|---:|
| `bits=1` | 1759 |
| `bits=4` | 1497 |
| `bits=8` | 1482 |
| `bits=2` | 1360 |
| `bits=3` | 1346 |

## Dimension d

| event | count |
|---|---:|
| `d=9-33` | 3125 |
| `d=odd` | 3065 |
| `d=even` | 1984 |
| `d=3-8` | 1879 |
| `d=even_pow2` | 1862 |
| `d=34-64` | 941 |
| `d=65-256` | 640 |
| `d=1` | 533 |
| `d=odd_pow2` | 533 |
| `d=2` | 326 |

## Corpus size n

| event | count |
|---|---:|
| `n=4-24` | 3062 |
| `n=1` | 1559 |
| `n=25+` | 1319 |
| `n=3` | 765 |
| `n=2` | 739 |

## k relative to n

| event | count |
|---|---:|
| `k<n` | 4504 |
| `k>n` | 1487 |
| `k==n` | 1453 |
| `k=0` | 745 |

## Corpus content classes

| event | count |
|---|---:|
| `duplicate_rows` | 4279 |
| `tiny_or_zero_norm_rows` | 3406 |
| `zero_rows` | 3119 |
| `zero_query` | 518 |
| `empty_corpus` | 400 |
| `threaded_kernel_regime(n*d>=2^15)` | 6 |

## Packing alignment (PackedVectors)

| event | count |
|---|---:|
| `row_aligned_packing` | 3749 |
| `row_unaligned_packing` | 3695 |

## Centred mean kinds

| event | count |
|---|---:|
| `centred_mean=zero` | 583 |
| `centred_mean=const` | 307 |
| `centred_mean=big` | 285 |
| `centred_mean=corpus` | 241 |
| `centred_mean=spike` | 234 |
| `centred_mean=mix` | 171 |

## Script length (ops replayed on one container)

| event | count |
|---|---:|
| `script_len=1` | 149 |
| `script_len=3` | 53 |
| `script_len=8` | 44 |
| `script_len=2` | 42 |
| `script_len=6` | 37 |
| `script_len=5` | 28 |
| `script_len=7` | 28 |
| `script_len=4` | 19 |

## CPU operations

| event | count |
|---|---:|
| `cpu_op=search` | 409 |
| `cpu_op=adc` | 284 |
| `cpu_op=batch` | 261 |
| `cpu_op=score_vectors` | 258 |
| `cpu_op=packed_adc` | 256 |
| `cpu_op=twostage_part` | 250 |
| `cpu_op=twostage_full` | 234 |
| `cpu_op=decode_dot` | 214 |
| `cpu_op=drop_cache` | 185 |
| `cpu_op=packed_twostage_full` | 156 |
| `cpu_op=packed_twostage_part` | 113 |

## CPU operation x precision class

| event | count |
|---|---:|
| `cpu_op=search:precision=None` | 264 |
| `cpu_op=batch:precision=None` | 170 |
| `cpu_op=adc:precision=None` | 163 |
| `cpu_op=twostage_part:precision=None` | 162 |
| `cpu_op=twostage_full:precision=None` | 150 |
| `cpu_op=score_vectors:precision=None` | 126 |
| `cpu_op=packed_adc:precision=None` | 124 |
| `cpu_op=decode_dot:precision=None` | 116 |
| `cpu_op=score_vectors:precision=lower` | 95 |
| `cpu_op=packed_adc:precision=lower` | 93 |
| `cpu_op=packed_twostage_full:precision=None` | 92 |
| `cpu_op=search:precision=lower` | 82 |
| `cpu_op=adc:precision=lower` | 78 |
| `cpu_op=search:precision=bits` | 63 |
| `cpu_op=packed_twostage_part:precision=None` | 59 |
| `cpu_op=decode_dot:precision=bits` | 57 |
| `cpu_op=twostage_full:precision=lower` | 51 |
| `cpu_op=batch:precision=bits` | 47 |
| `cpu_op=twostage_part:precision=bits` | 45 |
| `cpu_op=batch:precision=lower` | 44 |
| `cpu_op=adc:precision=bits` | 43 |
| `cpu_op=twostage_part:precision=lower` | 43 |
| `cpu_op=packed_twostage_part:precision=lower` | 42 |
| `cpu_op=decode_dot:precision=lower` | 41 |
| `cpu_op=packed_adc:precision=bits` | 39 |
| `cpu_op=packed_twostage_full:precision=lower` | 39 |
| `cpu_op=score_vectors:precision=bits` | 37 |
| `cpu_op=twostage_full:precision=bits` | 33 |
| `cpu_op=packed_twostage_full:precision=bits` | 25 |
| `cpu_op=packed_twostage_part:precision=bits` | 12 |

## GPU(numpy) operations

| event | count |
|---|---:|
| `gpu_op=search` | 255 |
| `gpu_op=adc` | 160 |
| `gpu_op=twostage_full` | 136 |
| `gpu_op=drop_cache` | 135 |
| `gpu_op=twostage_part` | 135 |
| `gpu_op=batch` | 100 |

## Cache state

| event | count |
|---|---:|
| `derive:caches_warmed` | 430 |
| `gpu:cpu_warmed_first` | 120 |

## IVF configuration

| event | count |
|---|---:|
| `ivf_self_query_cell_checked` | 1291 |
| `ivf_mode=lsh` | 1040 |
| `ivf_src=compressed` | 1019 |
| `ivf_n_bits=1-4` | 956 |
| `ivf_src=packed` | 436 |
| `ivf_mode=rotated_prefix` | 415 |
| `ivf_n_bits=5-8` | 306 |
| `ivf_pool=empty` | 228 |
| `ivf_n_bits=12-16` | 193 |
| `ivf_pool=partial` | 102 |
| `ivf_pool=all` | 95 |

## Ties and near-ties actually met

| event | count |
|---|---:|
| `tie_boundary` | 437 |
| `tie_boundary_stage1` | 386 |
| `identical_rows_unequal_scores_in_search(BLAS gemv)` | 20 |
| `path_vs_path_tie_difference` | 14 |

## Metamorphic / derived containers

| event | count |
|---|---:|
| `identical_code_rows_compared` | 2008 |
| `adc_chunk_invariance:CompressedVectors` | 1600 |
| `adc_chunk_invariance:PackedVectors` | 1600 |
| `permuted_corpus_checked` | 1200 |
| `query_dtype_checked` | 1200 |
| `renorm_reader=raw` | 447 |
| `renorm_reader=renorm` | 444 |
| `derive:CompressedVectors.subset` | 429 |
| `derive:PackedVectors.subset` | 427 |
| `derive:PackedVectors.to_compressed.subset` | 421 |
| `encode_block_small` | 400 |
| `invalid_precision:decode` | 400 |
| `invalid_precision:gpu.search_adc` | 400 |
| `invalid_precision:gpu.search_twostage` | 400 |
| `invalid_precision:ivf.search_coarse` | 400 |
| `invalid_precision:ivf.search_twostage` | 400 |
| `invalid_precision:search` | 400 |
| `invalid_precision:search_adc` | 400 |
| `invalid_precision:search_adc[packed]` | 400 |
| `invalid_precision:search_batch` | 400 |
| `invalid_precision:search_twostage` | 400 |
| `invalid_precision:search_twostage[packed]` | 400 |
| `precision_bits_eq_none:gpu.search_adc` | 400 |
| `precision_bits_eq_none:ivf.search_coarse` | 400 |
| `precision_bits_eq_none:search` | 400 |
| `precision_bits_eq_none:search_adc` | 400 |
| `precision_bits_eq_none:search_adc[packed]` | 400 |
| `precision_bits_eq_none:search_batch` | 400 |
| `roundtrip:CompressedVectors.load(npz)` | 400 |
| `roundtrip:CompressedVectors.load_arrow` | 400 |
| `roundtrip:PackedVectors.from_rows` | 400 |
| `roundtrip:PackedVectors.load(npz)` | 400 |
| `roundtrip:PackedVectors.load_arrow` | 400 |
| `roundtrip:PackedVectors.load_arrow(mmap)` | 400 |
| `roundtrip:save_pq/load_pq` | 314 |
| `derive:at_precision_codes_checked` | 301 |
| `scaling_checked:linear` | 296 |
| `scaling_checked:centred` | 104 |
| `byte_identical:uncentred` | 99 |
| `offset_reorder:mean_norm=<1e6` | 91 |
| `byte_identical:centred` | 59 |
| `score_vectors_packed:unit` | 24 |
| `score_vectors_packed:scalar` | 7 |
| `score_vectors_packed:centred` | 4 |
| `offset_reorder:mean_norm=>=1e6` | 2 |

## Other

| event | count |
|---|---:|
| `mismatch:not_expressible` | 178 |
| `mismatch:bits:cold` | 97 |
| `decode_dot:skipped_centred` | 46 |
| `mismatch:rotation:cold` | 36 |
| `mismatch:seed:cold` | 21 |
| `mismatch:mode:cold` | 20 |
| `mismatch:mode:warm` | 9 |
| `mismatch:scale:warm` | 8 |
| `mismatch:rotation:warm` | 7 |
| `mismatch:mean_added:cold` | 6 |
| `mismatch:mean_changed:cold` | 6 |
| `mismatch:seed:warm` | 5 |
| `mismatch:bits:warm` | 4 |
| `mismatch:mean_dropped:cold` | 2 |
| `mismatch:scale:cold` | 1 |

## Hypothesis statistics (per property)

Passing / failing / invalid test cases per property, from `--hypothesis-show-statistics`.

- `test_cpu_paths_match_oracle`: 400 passing, 0 failing, and 54 invalid test cases; runtimes ~ 2-99 ms, of which ~ 1-89 ms in data generation
- `test_cold_cache_each_path`: 400 passing, 0 failing, and 64 invalid test cases; runtimes ~ 2-125 ms, of which ~ 1-101 ms in data generation
- `test_score_vectors_accepts_packed`: 0 passing, 7 failing, and 3 invalid test cases; runtimes ~ 1-270 ms, of which ~ 0-266 ms in data generation
- `test_gpu_numpy_backend_matches`: 400 passing, 0 failing, and 56 invalid test cases; runtimes ~ 2-84 ms, of which ~ 1-73 ms in data generation
- `test_ivf_flat_limit_matches_oracle_and_flat_paths`: 400 passing, 0 failing, and 51 invalid test cases; runtimes ~ 3-192 ms, of which ~ 1-99 ms in data generation
- `test_ivf_flat_limit_is_byte_identical_to_adc`: 88 passing, 3 failing, and 13 invalid test cases; runtimes ~ 2-130 ms, of which ~ 2-125 ms in data generation
- `test_ivf_partial_probe_sound`: 400 passing, 0 failing, and 64 invalid test cases; runtimes ~ 3-95 ms, of which ~ 1-76 ms in data generation
- `test_ivf_structure`: 400 passing, 0 failing, and 88 invalid test cases; runtimes ~ 1-99 ms, of which ~ 1-88 ms in data generation
- `test_batch_equals_single_and_permutation`: 400 passing, 0 failing, and 70 invalid test cases; runtimes ~ 2-70 ms, of which ~ 1-58 ms in data generation
- `test_query_scaling_and_mean_shift_invariance`: 400 passing, 0 failing, and 59 invalid test cases; runtimes ~ 1-45 ms, of which ~ 1-40 ms in data generation
- `test_derived_containers_search_like_their_rows`: 400 passing, 0 failing, and 58 invalid test cases; runtimes ~ 2-37 ms, of which ~ 1-24 ms in data generation
- `test_invalid_precision_rejected_uniformly`: 400 passing, 0 failing, and 61 invalid test cases; runtimes ~ 1-66 ms, of which ~ 1-52 ms in data generation
- `test_empty_corpus_and_query_dtypes`: 400 passing, 0 failing, and 75 invalid test cases; runtimes ~ 1-13 ms, of which ~ 0-10 ms in data generation
- `test_query_offset_cannot_reorder`: 7 passing, 3 failing, and 0 invalid test cases; runtimes ~ 3-17 ms, of which ~ 0-1 ms in data generation
- `test_serialized_containers_search_identically`: 400 passing, 0 failing, and 65 invalid test cases; runtimes ~ 1-64 ms, of which ~ 1-34 ms in data generation
- `test_adc_scores_independent_of_chunking_and_position`: 400 passing, 0 failing, and 63 invalid test cases; runtimes ~ 1-40 ms, of which ~ 1-29 ms in data generation
- `test_renorm_is_a_reader_property_and_caches_do_not_leak`: 400 passing, 0 failing, and 66 invalid test cases; runtimes ~ 1-29 ms, of which ~ 1-24 ms in data generation
- `test_precision_equal_to_bits_is_full_precision`: 400 passing, 0 failing, and 64 invalid test cases; runtimes ~ 1-64 ms, of which ~ 1-50 ms in data generation
- `test_encode_block_size_does_not_change_the_container`: 400 passing, 0 failing, and 81 invalid test cases; runtimes ~ 1-46 ms, of which ~ 1-37 ms in data generation
