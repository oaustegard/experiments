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

### 4 — F1 at a fixed 0.5 threshold ordered the arms by calibration, not separability
Micro/macro-F1 at 0.5 was the pre-registered metric (the issue names it). With
325 one-vs-rest models and positives at 1–14% per label, 0.3–0.6% of the
probability cells cross 0.5 for the sparse and TF-IDF arms, so the F1 ordering
reports where each arm's sigmoid sits relative to 0.5. Threshold-free
ranking metrics at C = 1000 reverse it: micro-AP TF-IDF word+char 0.657,
SPARSEUP 0.610, binary 0.587, gte-small 0.539; macro-AP TF-IDF 0.583, gte-small
0.509, SPARSEUP 0.508. Caught by the scheduled adversarial read before the
write-up (PLAN.md item 3). Fix: C selected by micro-AP; the write-up leads
with AP and with F1 at each arm's best single threshold, and keeps F1@0.5 as
the pre-registered number. Direction: the F1@0.5 table overstated SPARSEUP
against TF-IDF by the whole margin; on ranking, TF-IDF is ahead, which is
what PLAN.md predicted.

### 5 — TF-IDF saw the whole text; the encoders saw 512 tokens
`encode.py` truncates at 512 tokens (994 memories, 28.8%); the TF-IDF arms in
`probe.py` were fit on the full text. Found by the same adversarial read.
Fix: a `tfidf_word+char_trunc512` arm on the encoder's exact token window,
decoded back to text with the SPARSEUP tokenizer. Direction: part of TF-IDF's
AP lead was an information advantage; the truncated arm bounds it.
