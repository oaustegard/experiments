| id | machine | defect | caught | examples run | seconds | first failure |
|---|---|---|---|---|---|---|
| baseline | RemexIndexMachine | unmutated remex | NO | 155 | 31.4 |  |
| baseline | CacheMachine | unmutated remex | NO | 156 | 49.7 |  |
| baseline | GpuIvfMachine | unmutated remex | NO | 153 | 35.0 |  |
| M01 | RemexIndexMachine | RemexIndex.add forgets to reset the decoded-token cache (stale _decoded after add) | yes | 4 | 0.6 | IndexError: index 54 out-of-bounds in maximum.reduceat [0, 54) |
| M02 | RemexIndexMachine | RemexIndex.add returns ids counted from 0 instead of from len(index) | yes | 4 | 0.2 | AssertionError: add returned ids [np.int64(0), np.int64(1)], expected 1..2 |
| M03 | RemexIndexMachine | RemexIndex.add re-derives the centring mean from every batch instead of the first | yes | 4 | 0.4 | ValueError: mean mismatch: these codes were encoded against a different corpus mean. Decoding against this one would shi |
| M04 | RemexIndexMachine | RemexIndex.load drops the corpus mean when rebuilding the quantizer | yes | 4 | 0.3 | ValueError: mean mismatch: these codes were encoded with a corpus mean but this Quantizer uses no mean. Rebuild the Quan |
| M05 | RemexIndexMachine | MaxSim block loop forgets to rebase reduceat offsets (breaks once the corpus spans several blocks) | yes | 19 | 5.1 | IndexError: index 32 out-of-bounds in maximum.reduceat [0, 15) |
| M06 | CacheMachine | CompressedVectors.drop_cache does not clear _dir_len (answer-equivalent: only the structural postcondition sees it) | yes | 10 | 3.1 | AssertionError: drop_cache left per-precision state behind |
| M07 | CacheMachine | CompressedVectors.subset copies the parent's whole float32 cache instead of the selected rows | yes | 4 | 1.0 | AssertionError:  |
| M08 | CacheMachine | Quantizer._direction_lengths caches under one key regardless of precision | yes | 6 | 1.6 | AssertionError: decode at precision 4 disagrees with the cache-free reconstruction |
| M09 | CacheMachine | Quantizer.R setter does not drop the RHT operator (assigned matrix ignored) | yes | 12 | 3.6 | AssertionError: search_twostage cp=1 cand=7: scores [1.6786531209945679, 0.7138461470603943, 0.34738782048225403, 0.1119 |
| M10 | CacheMachine | Quantizer._get_x_hat_rot serves the full-precision cache to every precision | yes | 6 | 2.0 | AssertionError: search_batch row 0 p=4: returned scores [12.983535766601562, 7.174972057342529, 3.630688190460205, 3.630 |
| M11 | CacheMachine | CompressedVectors.subset does not slice mean_dots (centred containers) | yes | 4 | 0.7 | ValueError: mean_dots has 5 rows, the codes have 3 |
| M12 | CacheMachine | Quantizer._row_bias caches the centred bias under one key regardless of precision | yes | 82 | 18.5 | AssertionError: search_batch row 0 p=None: returned scores [1.8079841136932373, 1.104323148727417, 1.104323148727417, 1. |
| M13 | GpuIvfMachine | GPUSearcher caches the device scale/bias under one key regardless of precision | NO | 153 | 25.3 |  |
| M14 | GpuIvfMachine | GPUSearcher.drop_cache does not release the device cache (answer-equivalent) | yes | 6 | 0.7 | AssertionError:  |
| M15 | GpuIvfMachine | IVF probe order breaks Hamming ties by descending cell id | yes | 10 | 1.8 | AssertionError: IVF.search_twostage nprobe=13: returned (8,) ids / (8,) scores, expected 0 |
| M16 | GpuIvfMachine | IVF candidate_indices drops the last row of every visited cell | yes | 4 | 0.7 | AssertionError: candidate set for nprobe=15 differs from an independent derivation |
| M17 | CacheMachine | Quantizer._effective_norms skips the per-search guards when the container is warm (renorm shortcut) | yes | 50 | 10.8 | Failed: DID NOT RAISE ValueError |
