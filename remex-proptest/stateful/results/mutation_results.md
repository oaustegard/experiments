| id | machine | defect | caught | examples run | seconds | first failure |
|---|---|---|---|---|---|---|
| baseline | RemexIndexMachine | unmutated remex | NO | 155 | 22.7 |  |
| baseline | CacheMachine | unmutated remex | NO | 156 | 38.5 |  |
| baseline | GpuIvfMachine | unmutated remex | NO | 155 | 36.0 |  |
| M07 | CacheMachine | CompressedVectors.subset copies the parent's whole float32 cache instead of the selected rows | yes | 4 | 0.7 | AssertionError:  |
| M17 | CacheMachine | Quantizer._effective_norms skips the per-search guards when the container is warm (renorm shortcut) | yes | 50 | 12.3 | Failed: DID NOT RAISE ValueError |
