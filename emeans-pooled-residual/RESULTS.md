# emeans-pooled-residual — centroid-residual codes for pooled ColBERT tokens, and where e-means beats Lloyd

**Started:** 2026-11-13 · **Status:** done (SciFact, NFCorpus, clustering grid) · **Runtime:** re-encode 28 min, residual bench 3 h 40 min (one restart at the 2 h job cap), grid 46 min, 4 vCPU

## Answer

A ColBERT index stores a 128-number vector per token. `lateon-pool-remex` shrank a LateOn index by merging
similar tokens and quantizing each vector's offset from one corpus-wide mean to 1–2 bits per number (remex). Does
coding each vector relative to its nearest of 1,024–16,384 k-means centroids do better? And where does e-means,
a new mini-batch k-means, beat Lloyd's algorithm?

At 1 bit per number, centroids gain up to +0.021 nDCG@10 on SciFact <!-- @evidence[[summary.json#/headline/scifact/resid_1b_gain/1]] -->
and +0.015 on NFCorpus, about what a second bit is worth. <!-- @evidence[[summary.json#/headline/nfcorpus/resid_1b_gain/1]] -->
At 2 bits the gain is smaller and mostly not significant. The corpora are small (5,183 and 3,633 documents),
so the centroid table costs as much as the saved bit and the two codings tie at similar size. A corpus large
enough to spread the table thin would save 35.6% of the bytes (arithmetic, not measured). <!-- @evidence[[summary.json#/headline/scifact/row_bytes_saving_pct]] -->

The k-means trainer barely moves retrieval. e-means pays at 32 or more dimensions with at least
~50 points per cluster, reaching Lloyd's error up to 8.89 times sooner in numpy. <!-- @evidence[[summary.json#/headline/grid/np_speedup_d32plus_nk50/1]] -->
With fewer than ~25 points per cluster it ends at a higher error, and at 3 dimensions faiss-based e-means is slower
than Lloyd from about 256 clusters.

## Findings

1. **At 1 bit, centroid residuals recover about one bit's worth of ranking.** e-means-2 centroids against the
   single mean, paired bootstrap over queries: SciFact +0.0144 to +0.0206, <!-- @evidence[[summary.json#/headline/scifact/resid_1b_gain/0]] -->
   4 of 6 CIs clear of zero; <!-- @evidence[[summary.json#/headline/scifact/resid_1b_ci_clear]] -->
   NFCorpus +0.010 to +0.0153, <!-- @evidence[[summary.json#/headline/nfcorpus/resid_1b_gain/1]] -->
   6 of 6 clear. <!-- @evidence[[summary.json#/headline/nfcorpus/resid_1b_ci_clear]] -->
   Going from 2 bits to 1 on the single mean costs 0.0156 on SciFact <!-- @evidence[[summary.json#/headline/scifact/drop_2b_to_1b_pf4]] -->
   and 0.0186 on NFCorpus. <!-- @evidence[[summary.json#/headline/nfcorpus/drop_2b_to_1b_pf4]] -->
   (`summary.json` `residual_vs_mean`)
2. **At 2 bits the gain is smaller and mostly not significant**: SciFact +0.0057 to +0.0144 <!-- @evidence[[summary.json#/headline/scifact/resid_2b_gain/1]] -->
   with 1 of 6 CIs clear of zero, <!-- @evidence[[summary.json#/headline/scifact/resid_2b_ci_clear]] -->
   NFCorpus −0.0001 to +0.003 with 0 of 6. <!-- @evidence[[summary.json#/headline/nfcorpus/resid_2b_ci_clear]] -->
3. **At similar size, with the table counted, 1-bit centroid codes tie 2-bit single-mean codes.** Pool factor 4
   (pf4: a quarter of each document's tokens kept; pf2: half), K = 16,384 centroids:
   SciFact 2,299 vs 2,314 B per document, +0.0047 [−0.0073, +0.0172]; <!-- @evidence[[summary.json#/scifact/matched_bytes/pf4 K16384 e-means2 1b vs pf4 mean 2b (~2.3 KB)/delta_ci/2]] -->
   NFCorpus 2,693 vs 2,389 B (13% larger), −0.0038 [−0.0108, +0.0028]. <!-- @evidence[[summary.json#/nfcorpus/matched_bytes/pf4 K16384 e-means2 1b vs pf4 mean 2b (~2.3 KB)/delta_ci/0]] -->
   The fp16 table is 809 B per document on SciFact <!-- @evidence[[summary.json#/headline/scifact/table_bytes_K16384]] -->
   and 1,155 B on NFCorpus <!-- @evidence[[summary.json#/headline/nfcorpus/table_bytes_K16384]] -->
   because these corpora hold only 5,183 and 3,633 documents. Per-token bytes alone (codes, two side scalars and
   the cell id, no table): 1,490 against 2,314 on SciFact, 1,538 against 2,389 on NFCorpus. <!-- @evidence[[summary.json#/headline/scifact/row_bytes_pf4_K16384_emeans2_1b]] -->
   The byte frontier from the pre-chosen e-means-2 arms is in `summary.json` (`frontier_table_included`,
   `frontier_table_excluded`) and `fig_frontier.png`.
4. **The trainer barely moves retrieval.** An arm is one pool factor × K × bit width (12 per corpus). Against
   Lloyd at 25 iterations (Lloyd-25), e-means at 2 passes over the data (e-means-2) is within ±0.005 nDCG@10
   in 9 of 12 SciFact arms <!-- @evidence[[summary.json#/headline/scifact/trainer_within_0005/emeans2]] -->
   and 11 of 12 NFCorpus arms; <!-- @evidence[[summary.json#/headline/nfcorpus/trainer_within_0005/emeans2]] -->
   Lloyd-2 in 10 of 12 <!-- @evidence[[summary.json#/headline/scifact/trainer_within_0005/lloyd2]] -->
   and 12 of 12. <!-- @evidence[[summary.json#/headline/nfcorpus/trainer_within_0005/lloyd2]] -->
   Of the 36 trainer-vs-Lloyd-25 comparisons per corpus, 5 have CIs clear of zero <!-- @evidence[[summary.json#/headline/scifact/trainer_ci_clear]] -->
   and most favour Lloyd-25 (3 of 5 on SciFact, all 5 on NFCorpus, by 0.004–0.006 there). <!-- @evidence[[summary.json#/headline/nfcorpus/trainer_ci_clear_favour_lloyd25]] -->
5. **e-means-2 fits in a twelfth to an eighth of Lloyd-25's time and ends within 1.4% of its error.** This is
   total fit time, Lloyd-25's over e-means-2's, on the pooled tokens (d = 128); the grid's matched-error
   speedup on the same tokens is lower (finding 6). SciFact 7.91 to 12.33 times, <!-- @evidence[[summary.json#/headline/scifact/emeans2_fit_speedup/0]] -->
   NFCorpus 7.53 to 11.88 times, <!-- @evidence[[summary.json#/headline/nfcorpus/emeans2_fit_speedup/0]] -->
   at 0.9866 to 1.0086 of Lloyd-25's error. <!-- @evidence[[summary.json#/headline/nfcorpus/emeans2_mse_ratio/0]] -->
   Lloyd-2 costs about the same time at 1.0631 to 1.0843 times the error <!-- @evidence[[summary.json#/headline/nfcorpus/lloyd2_mse_ratio/1]] -->
   and ranks as well (finding 4). At K = 16,384 e-means-2 leaves 29.5–117.5 tiny cells (under a tenth of the mean
   cell size, mean over seeds) against Lloyd-25's 244–531. <!-- @evidence[[summary.json#/headline/nfcorpus/tiny_cells_K16384_lloyd25/1]] -->
6. **Where e-means pays depends on the dimension d and the points per cluster n/k, not on k × d.** Speedup here is
   Lloyd's time to reach e-means-2's error over e-means-2's time, same random init (`fig_grid.png`,
   `grid_matched_emeans2`):
   - d ≥ 32 and n/k ≥ 50: 1.42 to 8.89 times with `NumpyEmeans`; <!-- @evidence[[summary.json#/headline/grid/np_speedup_d32plus_nk50/0]] -->
     2.22 to 52.47 times with torch `Emeans` on CPU at d ≥ 128; <!-- @evidence[[summary.json#/headline/grid/torch_speedup_d128plus_nk50/0]] -->
     0.48 to 8.24 times with torch at d = 32. <!-- @evidence[[summary.json#/headline/grid/torch_speedup_d32_nk50/0]] -->
   - LateOn tokens (real, d = 128): 2.58 to 5.82 times numpy, <!-- @evidence[[summary.json#/headline/grid/lateon_np_speedup/1]] -->
     11.54 to 23.83 times torch, <!-- @evidence[[summary.json#/headline/grid/lateon_torch_speedup/1]] -->
     ending at 0.9954 to 1.009 of Lloyd-25's error. <!-- @evidence[[summary.json#/headline/grid/lateon_err/1]] -->
   - On the clustered synthetic mixture at k = 256–1,024, e-means-2 ends as low as 0.8088 of Lloyd-25's error. <!-- @evidence[[summary.json#/headline/grid/gmm_err_k256_1024_d32plus_nk50/0]] -->
7. **Below ~25 points per cluster at d ≥ 128, e-means-2 ends 1.0387 to 1.3812 times Lloyd-25's error** (3 cells). <!-- @evidence[[summary.json#/headline/grid/small_nk_d128plus_err/1]] -->
   The two d ≥ 128 cells at 25–50 points per cluster (n/k = 39) both end below Lloyd-25's error.
8. **At d = 3, faiss's `NumpyEmeans` is faster than Lloyd only for small k**: 1.28 to 3.32 times at k ≤ 64, <!-- @evidence[[summary.json#/headline/grid/d3_k64_or_less_speedup/1]] -->
   0.68 to 1.13 at k = 256, <!-- @evidence[[summary.json#/headline/grid/d3_k256_speedup/1]] -->
   0.1 to 0.47 from k = 1,024 (n ≥ 100k, synthetic data and the pixels of Van Gogh's Starry Night). <!-- @evidence[[summary.json#/headline/grid/d3_k1024plus_speedup/1]] -->
   faiss's `knn` on a batch of rows costs more per distance than one call over all rows:
   up to 51.7 times at d = 3, k = 4,096, <!-- @evidence[[results_knn.json#d3_k4096.batch_over_full.32768]] -->
   1.2 times at d = 32, k = 4,096 for batches of 4,096 (6.9 times for batches of 512), <!-- @evidence[[results_knn.json#d32_k4096.batch_over_full.4096]] -->
   and no more at d = 128. e-means sizes its mini-batches from a power law in n, k and passes that its authors fitted to their runs (its batch law); at d ≤ 16 it uses the smaller of two such laws, which picks 512–8,192 rows here. A hand-written loop
   (JS, WGSL) should not pay this; untested.
9. **torch `Emeans` on CPU runs e-means-2 2.06 to 9.75 times faster than `NumpyEmeans` at d ≥ 128, n ≥ 100k.** <!-- @evidence[[summary.json#/headline/grid/torch_vs_np_emeans2_time_d128plus_n1e5plus/1]] -->
   A 20k-row smoke test had suggested otherwise (`ERRORS.md` #2).

## Method

Tokens: `lateon-pool-remex/encode.py` (LateOn at its model card's settings, SciFact 300 and NFCorpus 323 test queries),
pooled with sentence-transformers 6.3 `HierarchicalTokenPooling` at factor 4 and 2. Codes: remex 2.2.0
Lloyd-Max levels and rotation (seed 42). Each token x in cell j is stored as remex codes of the direction of
r = x − c_j plus fp32 ‖r‖ and c_j·r (8 B, as `RemexIndex` stores), and scored with remex 2's centred estimator
applied per cell, `q·c + c·r + ‖r‖ (q−c)·û / κ` (û the decoded unit direction, κ remex's expected cosine between û and the true direction), then brute-force MaxSim and nDCG@10. K = 1 (the pooled corpus
mean) is the control: it scores identically to `remex.Quantizer(mean=)`. Centroids: K = 1,024 / 4,096 / 16,384
from stepped `faiss.Kmeans` (2 and 25 iterations) and `faiss.contrib.e_means.NumpyEmeans` (2 and 5 passes),
same random init per seed, seeds 0 and 1, faiss-cpu 1.16.0. Bytes per document count codes, side scalars, the
cell id and the fp16 centroid table spread over the corpus. CIs: paired bootstrap over queries, 10,000 resamples,
per-query nDCG averaged over seeds first. Headline comparisons use the e-means-2 arms, chosen before the run.

Grid (`grid.py`): 1,000-component Gaussian mixture, n ∈ {1e4, 1e5, 1e6}, k 16–16,384, d ∈ {3, 32, 128, 768},
plus Starry Night pixels (n = 625k, d = 3) and LateOn pf4 tokens (n = 300k, d = 128); cells with n·k·d > 2.5e11
or k > n/8 skipped and listed. Per cell: Lloyd stepped to 25 iterations (time and error per step), `NumpyEmeans`
and torch `Emeans` at 1, 2, 3, 5 passes; speedup = Lloyd's time to reach e-means-2's error over e-means-2's time
(a lower bound where Lloyd-25 never gets there). `knn_batch_probe.py` measures faiss's batch penalty.
`analyze.py` writes `summary.json` (with a `headline` block for the numbers above) and both figures.

```bash
python3 ../lateon-pool-remex/encode.py                  # ~30 min, writes ../lateon-pool-remex/.work
python3 bench.py scifact && python3 bench.py nfcorpus   # resumable; ~2 h and ~1.5 h on 4 vCPU
python3 grid.py && python3 knn_batch_probe.py           # ~46 min, ~1 min
python3 analyze.py
```

## Log

### Round 1 — residual codes on pooled LateOn tokens (2026-11-13)

Asked (Oskar, after reading LookAtMyQuant 2, which fits trellis codebooks with e-means): "What about our
document vector pooling we just did earlier today?", then "we should [run it] as a matter of being exhaustive".
Re-encoded (the `.work` cache did not survive the session), then `bench.py` on both corpora: 2 pool factors ×
(1 control + 3 K × 2 seeds × 4 trainers) × 2 bit widths per corpus. SciFact hit the 2 h job cap at 94 of 100
rows and resumed (`ERRORS.md` #1).

| # | prediction (PLAN.md) | outcome |
|---|---|---|
| P1 | K=1 control matches `Quantizer(mean=)`, near `lateon-pool-remex` round 3 | held: identical nDCG (0.7304 / 0.746 SciFact pf4), median score diff 1.2e-7; pf4 2-bit 0.746 vs round-3 four-seed 0.751 |
| P2 | K=4096 − K=1, pf4 1 bit, SciFact: +0.005 to +0.015 | slightly above: +0.016 |
| P3 | same at 2 bits: 0 to +0.008 | above on SciFact (+0.014), held on NFCorpus (+0.000) |
| P4 | e-means-2 − Lloyd-25 within ±0.005 | 20 of 24 |
| P5 | Lloyd-2 − Lloyd-25 within ±0.005 | 22 of 24 |
| P6 | e-means-2 5–15× sooner at K=16384 | held: 11.5–12.3× |
| P7 | K=16384 loses at equal bytes to K=1 one bit up | wrong: ties at similar size on both corpora (NFCorpus arm 13% larger), CIs span zero |

Reading at the time (finding 2 later qualified the 2-bit half): the residual gain is real at 1 bit and gone at 2; the byte comparison is decided by the
centroid table, so corpus size decides whether to residualize. The first frontier took the best of four trainers
per point and was replaced by the pre-chosen e-means-2 arms (`ERRORS.md` #3).

### Round 2 — where e-means pays: clustering grid (2026-11-13)

Asked: "Can we determine some form of heuristic for where it pays off on a speed/size/efficiency Pareto
frontier?" Before the grid, `colour_bench.py` checked passes-to-error on two images at K = 16 and 56: on Starry
Night one e-means pass beat 20 Lloyd passes (K = 56 error 258–259 vs 260–263); on a photo one pass matched 4–5
Lloyd passes and e-means levelled off about 1.5% above Lloyd-20 (`results_colour.json`). Then `grid.py`
(66 cells run, 17 skipped as too large or k > n/8).

| # | prediction (PLAN.md) | outcome |
|---|---|---|
| G1 | numpy e-means sooner only where k·d ≳ 1e5 | wrong: at d ≥ 32 it is sooner from k = 16 (k·d = 512); at d = 3 it is slower from k ≈ 256 whatever k·d is |
| G2 | best error below Lloyd-25 for k ≥ 1,024 | mostly; fails below 25 points per cluster (3 cells, up to 1.38×) |
| G3 | torch CPU not faster than numpy | wrong at scale (finding 9) |

Profiling one d = 3 cell put 88% of e-means time in `knn_L2sqr` on 392 batches of 512 rows, which led to
`knn_batch_probe.py`. Two runs of the probe gave 7–44× and 5–52× at d = 3 (the stored run is the second).

Not done: one rotation seed and two centroid seeds; no corpus large enough to show the table amortizing; no
search-speed measurement; no JS/WGSL e-means port; synthetic data starts from random init, and with k-means++
Lloyd would start closer, likely shrinking e-means' error advantage there.
