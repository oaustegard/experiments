# emeans-pooled-residual — plan and pre-registered predictions

Written 2026-11-13 before any number from this directory was seen.

## Question

Oskar, 2026-11-13, after reading LookAtMyQuant 2 (which fits trellis codebooks with e-means):
"What about our document vector pooling we just did earlier today?" and "we should [run it] as
a matter of being exhaustive".

`lateon-pool-remex` codes Ward-pooled LateOn tokens with centered remex: one stored mean, K = 1.
`emeans-residual` showed that coding each vector against the nearest of K k-means centroids
raises 1-bit recall, but at 100k single vectors the centroid table cost more than a second bit.
A token index has millions of rows, so the table amortizes better there, and the earlier
mxbai-edge run found PLAID (K = 16k centroids) ahead of centered remex at 1 bit on SciFact
(+0.029). Two questions:

1. Does residualising pooled tokens against K = 1k / 4k / 16k centroids beat the single mean,
   and does it pay once the table is counted?
2. Does the trainer matter: e-means at 2 or 5 passes against Lloyd at 2 or 25 iterations, from
   the same init? And how much fit time does e-means save at these sizes?

Fixture: `lateon-pool-remex/encode.py` output (LateOn, SciFact + NFCorpus, card settings),
sentence-transformers 6.3 `HierarchicalTokenPooling` pf4 and pf2, remex 2.2.0 codebooks and
rotation (seed 42), faiss-cpu 1.16.0 (`contrib/e_means` now ships in the wheel), 4 vCPU,
centroid seeds 0 and 1.

## Predictions

| # | quantity | prediction |
|---|---|---|
| P1 | K=1 control vs `Quantizer(mean=mu)` | nDCG@10 within 0.002; close to round 3 (SciFact pf4@2 0.751, pf2@2 0.760) within the rotation-seed spread (~0.005) |
| P2 | K=4096 residual − K=1, pf4 1-bit, SciFact | +0.005 to +0.015 |
| P3 | same at 2 bits | 0 to +0.008 |
| P4 | e-means-2 − Lloyd-25 nDCG@10, every arm | within ±0.005 |
| P5 | Lloyd-2 − Lloyd-25 nDCG@10 | within ±0.005 |
| P6 | e-means-2 fit time vs Lloyd-25 at K=16384 | 5–15× sooner |
| P7 | fp16 table at SciFact scale (5.2k docs) | K=16384 costs ~800 B/doc, so it loses at equal bytes to K=1 one bit up; K ≤ 1024 is the only setting that can pay at this corpus size |

## What would change the plan

If P1 misses by more than 0.005, stop and fix the estimator before reading any residual arm.

## Grid (added 2026-11-13, Oskar: "determine some form of heuristic for where it pays off")

`grid.py`: time to reach Lloyd-25's error, and best e-means error / Lloyd-25 error, over
n ∈ {1e4, 1e5, 1e6} × k ∈ {16 … 16,384} × d ∈ {3, 32, 128, 768} on a 1,000-component Gaussian
mixture, plus Starry Night pixels and pooled LateOn tokens. NumpyEmeans and torch `Emeans` (CPU).
Model: speedup ≈ (I_L / P_e) / (1 + ρ), ρ = t_o / (B·k·d·t_f).

| # | quantity | prediction |
|---|---|---|
| G1 | numpy e-means reaches Lloyd-25 error sooner | only where k·d ≳ 1e5; slower than Lloyd-25 where k·d ≲ 1e4 |
| G2 | best e-means error / Lloyd-25 | < 1 for k ≥ 1024 at every d; ≈ 1 (±0.5%) for k ≤ 64 |
| G3 | torch CPU `Emeans` vs numpy | not materially faster on CPU (a smoke cell at n=20k, k=64, d=32 took 0.50 vs 0.48 s), so it is not the compiled-loop stand-in hoped for; a real stand-in would need a GPU or a JS/WGSL port |
