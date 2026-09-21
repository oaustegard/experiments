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

### 3 — one C across arms with different feature scales (caught before the write-up)
At C = 1.0 the TF-IDF arms scored micro-F1 0.09–0.16 and gte-small 0.09 against
SPARSEUP's 0.49, with P@5 gaps a third that size. Those rows are L2-normalised
(max feature value < 1) while SPARSEUP's `log1p(relu)` weights run to 3.9 and
mean row L2 norm is 32 (gte-small rows: 1.0), so the same C is a far stronger prior on the normalised arms
and their probabilities rarely cross the 0.5 threshold. The first table would
have "confirmed" the learned expansion by a regularisation artefact. Caught
reading the table against the prediction (TF-IDF was expected to tie), before
any prose. Fix: `--C` sweep over {1, 10, 100, 1000} for every arm; the write-up
reports each arm at its best C and shows the grid. Direction: the C = 1 table
overstated SPARSEUP's margin over TF-IDF and the binarization gap (binary rows
have max value 1, so they were more regularised than the weighted rows too).
