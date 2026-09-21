# Error log — sparseup-tag-probe

What was wrong, how it was caught, and which direction it pushed the conclusion.
Convention as in `remex-vs-higgs-ablation/ERRORS.md`.

## Run 1 (2026-09-21)

### 1 — sparse-tensor `coalesce()` crashed under `inference_mode`
`encode.py` asked `encode_document` for a sparse tensor and called `coalesce()`
inside `torch.inference_mode()`; PyTorch raised an internal assert in
`SparseTensor.cpp`. Caught on the first batch by the background job's exit code.
Fix: encode to a dense batch (64 × 50,370 floats) under `no_grad` and build the
CSR from that. No effect on any number; cost one relaunch.
