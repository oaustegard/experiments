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

### 2 — read-only memmap under joblib broke the TF-IDF arms
Parallelising the per-label fits with joblib processes memmaps the training
matrix read-only; scikit-learn's `np.max(X)` guard then calls scipy's in-place
`sort_indices()` and raises `WRITEBACKIFCOPY base is read-only`. The two SPARSEUP
arms passed because their CSR was already canonical; the freshly built TF-IDF
matrices were not. Caught by the run's exit code. Fix: canonicalise once before
dispatch; per-arm probabilities are now cached so a rerun does not repeat the
finished arms. No effect on any number.
