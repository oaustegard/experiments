# e-means centroids for remex: residual codes and IVF cells

[e-means](https://e-means.github.io/) (Bessa et al., Meta FAIR, [arXiv 2611.09501](https://arxiv.org/abs/2611.09501), 6 November 2026) is a mini-batch replacement for Lloyd's k-means. The paper reports that it passes Lloyd's 25-iteration error in 1.4 to 2.6 passes over the data. [remex](https://github.com/oaustegard/remex) compresses embedding vectors without training and has no k-means anywhere: its inverted-file index, `IVFCoarseIndex`, assigns vectors to cells by hashing. [remex#81](https://github.com/oaustegard/remex/issues/81) asks for a mode that quantizes `x - centroid` instead of `x`, and that mode needs centroids. This experiment asks two questions on 100,000 Quora vectors from EmbeddingGemma 2 (768 dimensions) on 4 CPUs. Does the e-means claim hold at this size? And do k-means centroids improve remex enough to add them? Below, k and K both mean the number of centroids, and a width of 1, 2 or 4 bits is per dimension (96, 192 or 384 bytes per vector).

## Answer

The claim holds on this corpus. e-means reaches Lloyd's 25-iteration held-out error in 1 to 3 passes at every k tested. At k = 4,096 two passes take 10.9 s against Lloyd's 110.6 s, and ten passes end with 1.80% lower error. <!-- @evidence[[results/summary.json#cluster.4096.emeans_best_gain_pct]] -->

For remex the answer is yes for index cells and conditional for residual codes. Used as index cells, 4,096 k-means centroids put 90% of true neighbours inside 0.56% of the corpus. <!-- @evidence[[results/summary.json#ivf.lloyd25|K4096.scanned_at_90_pct]] --> The best of remex's four hash configurations holds 70% after scanning 34%.

Residual codes raise recall@10 at 1 bit from 0.840 with one corpus mean to 0.891 with 4,096 centroids, for a 1.5-byte cell id per row. <!-- @evidence[[results/summary.json#residual.emeans5|K4096|b1.recall10]] --> The centroid table costs more: spread over 100,000 vectors it is 126 bytes each, while a second bit costs 96 and reaches 0.913. The extrapolated break-even is about 194,000 vectors (finding 6); a larger corpus would test it.

The choice of trainer matters little for recall. e-means and Lloyd at 25 iterations differ by at most 0.005, and two Lloyd iterations come within about 0.005 of 25. e-means saves training time: 5 to 10 times less to reach Lloyd's error at k of 1,024 and up.

## Findings

1. **e-means matches Lloyd at 25 iterations in 1 to 3 passes.** Held-out MSE relative to Lloyd at 25, mean of 3 seeds: 1.0021 after 2 passes and 0.9986 after 3 at k = 1,024; 0.9939 after 2 at k = 4,096. <!-- @evidence[[results/summary.json#cluster.4096.emeans2_test_ratio]] --> At k = 16 one pass is enough (`results/summary.json` `cluster`, `fig_mse.png`).
2. **e-means reaches Lloyd's error 1.6 to 10.2 times sooner, and the largest gains are at the largest k.** The factors are 3.4, 1.6, 3.1, 5.1 and 10.2 for k from 16 to 4,096. <!-- @evidence[[results/summary.json#cluster.4096.wallclock_speedup]] --> One e-means pass costs 7.3 faiss Lloyd iterations at k = 16, because 88 small batches each pay fixed overhead, and 1.2 at k = 4,096. <!-- @evidence[[results/summary.json#cluster.4096.epoch_cost_ratio]] --> The paper's 8 to 15 times on CPU is for larger problems (`results/summary.json` `cluster`).
3. **The error gain at convergence is small at this size.** Ten e-means passes end 0.18% below Lloyd at k = 16, 0.67% at k = 1,024 and 1.80% at k = 4,096 (`results/summary.json` `cluster.*.emeans_best_gain_pct`).
4. **Lloyd leaves more near-empty clusters.** At k = 4,096 Lloyd ends with 108 clusters under a tenth of the mean size, e-means at 5 passes with 4. Each such cluster is a centroid, and its 3 KB of table, serving almost no rows (`results/summary.json` `cluster.4096.lloyd25_small`, `emeans5_small`).
5. **Residual codes raise recall at every width, most at 1 bit.** With 4,096 e-means centroids: +0.051 at 1 bit (95% interval +0.046 to +0.057), +0.028 at 2 bits, +0.011 at 4 bits, against remex centred on one mean. <!-- @evidence[[results/summary.json#residual.emeans5|K4096|b1.vs_mean.delta]] --> Each fourfold increase in K adds 0.003 to 0.013 at 1 bit (`fig_recall_bytes.png`).
6. **At 100,000 vectors the table costs more than the bits.** 1-bit codes with 4,096 centroids total 231 bytes per vector for 0.891; 2-bit codes with one mean total 200 bytes for 0.913. <!-- @evidence[[results/summary.json#residual.mean|K1|b2.recall10]] --> K = 1,024 at 1 bit (137 bytes, 0.879) is ahead of the 0.865 that a straight line between the 1-bit and 2-bit single-mean codes gives at 137 bytes. remex has no code at that size, so the line is only a yardstick. At the rate a second bit buys recall (0.073 for 96 bytes), the 0.051 gain at K = 4,096 is worth about 65 bytes, and the table falls to 65 bytes per vector at about 194,000 vectors, assuming the gain holds as the corpus grows. Storing the table as fp16 halves it with no measurable change in recall (`results/summary.json` `residual.emeans5-fp16|*`).
7. **Any trained centroids give nearly the same recall, and untrained ones give much less.** At K = 4,096 and 1 bit: random points 0.844, Lloyd at 2 iterations 0.885, Lloyd at 25 0.886, e-means at 2 passes 0.889, e-means at 5 0.891. <!-- @evidence[[results/summary.json#residual.lloyd25|K4096|b1.recall10]] --> Of the 15 comparisons between e-means at 5 passes and Lloyd at 25 (5 values of K at 3 widths), 12 have intervals that include zero. The largest gap is 0.0044, at K = 4,096 and 1 bit, where the interval excludes zero. At K = 16, random points score 0.785, below the single mean's 0.840, and they stay below it at 1 bit until K = 4,096 (`results/summary.json` `residual`).
8. **k-means cells need a small fraction of the scan that hash cells need.** Share of the corpus scanned to hold 90% of the true top 10: 3.8% with 256 Lloyd cells, 0.56% with 4,096. remex `IVFCoarseIndex` was run in four configurations (`lsh` or `rotated_prefix` mode, uncentred or centred quantizer). Uncentred `rotated_prefix` needs 67% at 256 cells, and the other three do not reach 90% within 128 probes. At 4,096 cells and 128 probes uncentred `rotated_prefix` holds 70% of the neighbours in 34% of the corpus. The figure plots the two centred configurations, whose cells are closer in size to the k-means cells; they hold at most 36% in 3.3%. <!-- @evidence[[results/summary.json#ivf.remex-rotated_prefix|K256.scanned_at_90_pct]] --> (`fig_ivf.png`)
9. **Lower k-means error did not give better cells.** At 4,096 cells e-means at 2 passes has to scan 0.65% of the corpus to hold 90% of the true neighbours and Lloyd at 25 scans 0.56%, although e-means has the lower MSE, and the seed ranges do not overlap (0.62 to 0.69 against 0.52 to 0.58). e-means at 5 passes scans 0.58% (`results/summary.json` `ivf.*|K4096`).

## Method

**Data.** The first 100,000 BEIR Quora documents and 1,000 test queries embedded with `google/embeddinggemma-2`, unit norm, d = 768, from the R2 embedding store (`embstore.py get quora/embeddinggemma-2/fp32-768-first100k`, in the claude-workspace repo). The corpus mean has length 0.79, so the vectors sit in a narrow cone and subtracting one mean already removes most of each vector; that single-mean code is the baseline. Ground truth is the exact inner-product top 10.

**Clustering (`cluster_bench.py`).** Trained on rows 0 to 89,999, with the last 10,000 held out. k is 16, 64, 256, 1,024 or 4,096, with three seeds, and both methods start from the same k random training rows. Lloyd is `faiss.Kmeans` (1.15.1) stepped one iteration at a time with subsampling off, which gives the centroids after every iteration; stepped and uninterrupted runs give identical centroids, and the wall-clock figures use the uninterrupted run. e-means is `NumpyEmeans` from faiss main at `f34c0b88` (not yet in the PyPI wheel), default settings, one fresh run per budget of 1, 2, 3, 5 or 10 passes because its batch size and step schedule depend on the budget. MSE is the mean squared distance to the nearest final centroid under exact assignment. Both methods ran in the same process with 4 threads, inside a sandbox with no network (the repo's rule for third-party source); the sandbox does not change timings.

**Residual codes (`residual_bench.py`).** Every corpus row is assigned to its nearest centroid `c`. The row stores remex codes (1, 2 or 4 bits per dimension, with one rotation shared by all cells) of the direction of `r = x - c`, plus `||r||` and `c . r` in fp32 and a cell id. Scoring is remex 2's centred estimator applied per cell: `q.c + c.r + ||r|| (q - c).u_hat / kappa`, where `u_hat` is the unit direction decoded from the codes and `kappa` is the expected cosine between a direction and its decoded version (0.798 at 1 bit). Control: with K = 1 and `c` the corpus mean, the scores equal `remex.Quantizer(mean=...)` 2.1.0 to a median of 1.2e-7 and recall is identical at all three widths. Bytes per vector count the codes, 8 bytes of floats, the cell id and the fp32 centroid table divided by 100,000. Intervals are paired bootstraps over the 1,000 queries after averaging the three k-means seeds.

**Index cells (also `residual_bench.py`).** Cells are ranked by L2 distance from the query to the centroid. For nprobe from 1 to 128 the script records the share of true top-10 rows inside the probed cells and the share of the corpus those cells hold. The baseline is `remex.IVFCoarseIndex` with the same number of cells, both modes, with an uncentred quantizer (its default) and with a centred one whose query hash uses `q - mean`. `lsh` hashes with random hyperplanes and `rotated_prefix` with the signs of the first rotated coordinates; both probe cells in order of Hamming distance from the query's cell.

**Reproduce.** `python3 cluster_bench.py --data data/docs.npy --out out --emeans-src <faiss>/contrib`, copy `out/runs.jsonl` to `results/cluster_runs.jsonl` and `out/cent/` to `cent/`, then `python3 residual_bench.py` and `python3 analyze.py`.

**Not tested.** One corpus, one model, one size. A GPU. k-means++ initialisation. Search speed: the residual scores here come from a dense matrix product, and a real implementation needs a scan kernel that adds a per-cell query offset.

## Log

### Round 1, 2026-11-10

Asked: whether the e-means post changes anything for remex. Ran `cluster_bench.py` (470 runs, about 28 minutes), then `residual_bench.py` (about 15 minutes) and `analyze.py`.

Clustering, mean of 3 seeds. Error columns are held-out MSE relative to Lloyd at 25 iterations.

| k | Lloyd 25, s | e-means 1 pass | 2 passes | 3 passes | 5 passes | 10 passes | e-means 2 passes, s | pass cost in Lloyd iterations |
|---|---|---|---|---|---|---|---|---|
| 16 | 2.2 | 0.9989 | 0.9989 | 0.9987 | 0.9987 | 0.9982 | 1.3 | 7.3 |
| 64 | 3.5 | 1.0025 | 1.0003 | 0.9993 | 0.9991 | 0.9992 | 1.5 | 5.4 |
| 256 | 8.5 | 1.0080 | 1.0004 | 0.9989 | 0.9974 | 0.9968 | 2.0 | 2.8 |
| 1,024 | 28.9 | 1.0194 | 1.0021 | 0.9986 | 0.9955 | 0.9933 | 3.6 | 1.6 |
| 4,096 | 110.6 | 1.0212 | 0.9939 | 0.9895 | 0.9858 | 0.9820 | 10.9 | 1.2 |

Lloyd itself is within 1.2% of its 25-iteration error after 5 iterations at every k.

Recall@10 of residual codes, e-means centroids at 5 passes, with the gain over one corpus mean:

| K | 1 bit | 2 bits | 4 bits | table, bytes per vector |
|---|---|---|---|---|
| 1 (remex centred) | 0.840 | 0.913 | 0.972 | 0.03 |
| 16 | 0.852 (+0.012) | 0.917 (+0.004) | 0.975 (+0.003) | 0.5 |
| 64 | 0.855 (+0.016) | 0.922 (+0.009) | 0.977 (+0.005) | 2.0 |
| 256 | 0.866 (+0.027) | 0.931 (+0.018) | 0.979 (+0.007) | 7.9 |
| 1,024 | 0.879 (+0.039) | 0.935 (+0.021) | 0.981 (+0.009) | 31.5 |
| 4,096 | 0.891 (+0.051) | 0.942 (+0.028) | 0.983 (+0.011) | 125.8 |

Share of the corpus scanned to hold 90% of the true top 10, with the range over seeds:

| cells | random points | Lloyd 2 | Lloyd 25 | e-means 2 | e-means 5 |
|---|---|---|---|---|---|
| 256 | 9.7% | 4.6% | 3.8% (3.7 to 3.9) | 3.9% (3.8 to 4.1) | 3.6% (3.5 to 3.6) |
| 1,024 | 4.5% | 1.8% | 1.5% (1.5 to 1.6) | 1.7% (1.7 to 1.8) | 1.7% (1.6 to 1.8) |
| 4,096 | 1.6% | 0.63% | 0.56% (0.52 to 0.58) | 0.65% (0.62 to 0.69) | 0.58% (0.55 to 0.61) |

Predictions against measurements. I expected e-means to match Lloyd in about 2 passes (it took 1 to 3) and expected the centroid source to matter little for recall (it did not matter beyond 0.004). I did not expect random points to fall below the single mean, or the table to outweigh a second bit at this corpus size.

The break-even corpus size for K = 4,096 at 1 bit comes out at about 194,000 vectors. It assumes the recall gain at fixed K holds as the corpus grows and prices a byte by the slope between the 1-bit and 2-bit single-mean codes; neither was measured.

Reading for remex#81: the residual mode can accept centroids computed elsewhere (`encode(X, base=...)`) and needs no trainer of its own, since two Lloyd iterations are enough for recall. The same centroids used as `IVFCoarseIndex` cells cut the candidate set by far more than the residual codes raise recall, but search speed and end-to-end recall through such an index were not measured here.
