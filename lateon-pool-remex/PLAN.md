# lateon-pool-remex — plan and pre-registered predictions

Written 2026-11-12 before any LateOn number was seen.

## Question

Oskar: "investigate a training-unaware hierarchical pooling of a modern ColBERT embedder, with integrated
remex quantization of the vectors."

`token-pooling-remex` pooled first and quantized second, each with its own setting, on two one-epoch
demonstration checkpoints (SciFact 0.658). Two things are open:

1. Does its rule (pool factor 4 + centered 4-bit beats unpooled 1-bit at the same bytes) hold on a strong
   ColBERT that was never trained for pooling? Model: `lightonai/LateOn` (ModernBERT-base, 128-d, Apache-2.0,
   BEIR 57.22, card SciFact 0.766 / NFCorpus 0.378).
2. **Integration.** Let the codec decide how much to pool. Centered remex at b bits moves each token by a
   measurable mean squared error D_b. Merging a Ward cluster moves its members by
   E_pool = 2(1 − ‖mean‖) (exact for unit vectors, measured to the renormalised mean). Cut each document's
   Ward tree where E_pool ≤ c·D_b for the node and every sub-cluster. One knob (bits) then sets both the
   codec and the pooling, and documents with redundant tokens pool more than dense ones.

Prior art (searched 2026-11-12, 4 searches): fixed-factor pooling (Clavié 2024, PyLate, sentence-transformers
6.3), pooling then ColBERTv2 residual compression (Clavié 2024), attention-score pooling (Jha et al. 2026,
arXiv 2603.22434). Did not find a pooling threshold derived from the quantizer's distortion.

## Arms

See `bench.py` docstring. Families: `none`, `st` (library pooling, sequential), `wu` (true Ward uniform
factor, implementation control), `adapt` (integrated, c ∈ {0.5, 1, 2, 4} × b ∈ {4, 3, 2, 1}), `match`
(uniform Ward at the adapt arm's total token count and bits). Side-info diagnostics: exact float32 fields,
fp16 μ·x (counted), no per-vector side info.

Bytes count the fp16 side scalar (2 B/vector), which `token-pooling-remex` left out.

## Predictions

| # | quantity | prediction |
|---|---|---|
| P1 | LateOn unpooled fp32, brute-force MaxSim | SciFact 0.766 ± 0.01, NFCorpus 0.378 ± 0.01 |
| P2 | unpooled 2-bit → 1-bit drop, SciFact | −0.02 to −0.05 |
| P3 | st pf4 4-bit − none 1-bit (≈5 KB/doc) | +0.01 to +0.04, CI clear of zero on SciFact |
| P4 | adapt c=1 at b − none at b | within ±0.005, CI spanning zero, at every b |
| P5 | adapt c=1 tokens kept | 4-bit > 90 %, 2-bit 60–85 %, 1-bit 30–60 % |
| P6 | adapt − match, same bits and tokens | 0 to +0.01; ~55 % that adapt wins on the point estimate |
| P7 | fp16 side vs exact float32 side | identical to 4 decimals of nDCG |
| P8 | no per-vector side info vs fp16 side, 1-bit | loses ≥ 0.01 |

## What would change the plan

If P1 misses by more than 0.02, stop and fix the encode before benching. If adapt c=1 already loses
measurably at every b (P4 fails), the codec's error is not the right yardstick for pooling error and the
writeup says so.
