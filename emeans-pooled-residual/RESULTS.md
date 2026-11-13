# emeans-pooled-residual — centroid-residual codes for pooled ColBERT tokens, and where e-means beats Lloyd

**Started:** 2026-11-13 · **Status:** done (SciFact, NFCorpus, clustering grid) · **Runtime:** re-encode 28 min, residual bench 3 h 40 min (one restart at the 2 h job cap), grid 46 min, 4 vCPU

## Answer

`lateon-pool-remex` codes each Ward-pooled [LateOn](https://huggingface.co/lightonai/LateOn) token as a remex
offset from one corpus mean. Here each token is coded as an offset from the nearest of K = 1,024 to 16,384
k-means centroids, with the cell id and a shared fp16 centroid table added to the bytes.

At 1 bit, residual codes beat the single mean on both corpora: +0.014 to +0.021 nDCG@10 on SciFact and
+0.010 to +0.015 on NFCorpus (e-means-2 centroids; 10 of 12 CIs clear of zero). That is roughly the size of the
drop from 2 bits to 1 bit (0.016 on SciFact, 0.019 on NFCorpus, pool factor 4). At 2 bits the gain is +0.006 to
+0.014 on SciFact and 0 on NFCorpus.

At matched bytes it is a tie. Pool factor 4 at 1 bit against 16,384 centroids, 2.3 KB per document with the
table, against the single mean at 2 bits, 2.3 KB: +0.005 [−0.007, +0.017] on SciFact, −0.004 [−0.011, +0.003]
on NFCorpus. The centroid table is 0.8–0.9 KB of each document's 2.3 KB here, because these corpora have
3,600–5,200 documents. Without the table, as in a corpus large enough to amortize it, the residual 1-bit codes
take 1.5 KB per document against 2.3 KB, about 35% fewer, at the same nDCG. That last step is arithmetic on the
measured row bytes, not a measurement on a large corpus.

The trainer does not matter for retrieval. e-means at 2 passes and Lloyd at 2 iterations land within ±0.005
nDCG@10 of Lloyd at 25 iterations in 20 of 24 and 22 of 24 comparisons. It matters for fit time: e-means-2
reaches Lloyd-25's k-means error (0.987 to 1.009 of it) 7.5 to 12.3 times sooner, and leaves 30–120 tiny cells
at K = 16,384 where Lloyd leaves 245–530. Lloyd-2 is as fast as e-means-2 with 6–8% higher error, and retrieves
just as well.

The grid shows where e-means pays as a k-means trainer:

- **d ≥ 32 and at least ~50 points per cluster:** e-means-2 reaches Lloyd's error 1.4–9 times sooner with
  `NumpyEmeans`, and 2–52 times sooner with torch `Emeans` on CPU at d ≥ 128 (mixed at d = 32: 0.5–8). On
  clustered synthetic data at k = 256–1,024 it ends 3–19% lower than Lloyd-25 from random init. On LateOn tokens it ends equal (0.995–1.009) and is 2.6–5.8 times
  faster (numpy) or 12–24 times (torch).
- **Fewer than ~25 points per cluster:** e-means-2 ends 3–38% *above* Lloyd-25 at d ≥ 128 (3 cells).
- **d = 3 (colour), n ≥ 100k:** faster only for k ≤ 64 (1.3–3.3 times). Around k = 256 it breaks even
  (0.7–1.1); from k = 1,024 it runs at 0.1–0.5 times Lloyd's speed, on synthetic data and on Starry Night
  pixels alike. (Cells at n = 10k finish in under half a second and are noisy.) The cause is faiss's assignment kernel, not the
  algorithm: at d = 3 a search over a batch of rows costs 7–44 times more per distance than one search over all
  rows (growing with k), and the e-means batch law picks small batches at low d.

## Findings

1. **Residual codes recover most of a bit at 1 bit and little at 2.** Paired bootstrap over queries, e-means-2
   centroids against the single mean at the same pool factor and bits (`summary.json` `residual_vs_mean`):

   | | SciFact 1 bit | SciFact 2 bits | NFCorpus 1 bit | NFCorpus 2 bits |
   |---|---|---|---|---|
   | pf4 K=1,024 | +0.016 [+0.001, +0.032] | +0.007 [−0.007, +0.021] | +0.010 [+0.001, +0.019] | −0.000 |
   | pf4 K=4,096 | +0.016 [+0.000, +0.032] | +0.014 [+0.002, +0.028] | +0.011 [+0.002, +0.019] | +0.000 |
   | pf4 K=16,384 | +0.020 [+0.006, +0.035] | +0.011 [−0.002, +0.025] | +0.015 [+0.006, +0.024] | +0.002 |
   | pf2 K=16,384 | +0.021 [+0.006, +0.036] | +0.008 [−0.002, +0.019] | +0.013 [+0.005, +0.022] | +0.002 |

   Single-mean baselines: SciFact pf4 0.730 / 0.746 (1 / 2 bits), pf2 0.744 / 0.756; NFCorpus pf4 0.356 / 0.375,
   pf2 0.358 / 0.377.
2. **At matched bytes, residual 1-bit ties single-mean 2-bit** (`matched_bytes`). The table is K × 128 × 2 bytes
   shared by the corpus: 50 B per document at K = 1,024 and 800–920 B at K = 16,384 on these corpora. Cell ids
   add 1.25–1.75 B per token. The byte frontier built from the pre-chosen e-means-2 arms (not the best of four
   trainers) is in `summary.json` `frontier_table_included` / `frontier_table_excluded` and `fig_frontier.png`.
3. **Trainer choice moves nDCG by under 0.005 in most arms** (`trainer_vs_lloyd25`). CIs clear of zero in 10
   of 72 comparisons, 8 of them favouring Lloyd-25; the largest is SciFact pf2 K = 4,096 at 1 bit, where all three
   alternatives score 0.011–0.013 below Lloyd-25 (one Lloyd-25 arm unusually high, 0.769).
4. **e-means-2 matches Lloyd-25's k-means error in a tenth of the time** (`fit`): 7.5–12.3 times sooner on
   300k–600k pooled tokens, d = 128, at the same error within 1.4%. At K = 16,384 on SciFact pf2: 53 s against
   631 s.
5. **The grid's speedup depends on d and on points per cluster, not on k × d.** `fig_grid.png`,
   `summary.json` `grid_matched_emeans2`: Lloyd's time to reach e-means-2's error divided by e-means-2's time,
   both from the same random init. Triangles are lower bounds (Lloyd-25 never reached e-means-2's error).
6. **faiss small-batch assignment is the low-d bottleneck.** `faiss.knn(x, C, 1)` per distance, batch against
   one full pass (1M rows): d = 3, k = 4,096: 41–44× for batches of 512–32,768; d = 32: 5× at 512, ~1× at
   ≥ 4,096; d = 128: 0.9–2.0×. At d = 3, k = 4,096, n = 100k, 88% of e-means time is in `knn_L2sqr` on 392
   batches of 512. A hand-written loop (JS, WGSL) has no such penalty; that port is untested.
7. **torch `Emeans` on CPU is 2–10× faster than `NumpyEmeans` at d ≥ 128 and n ≥ 100k.** The 20k-row smoke
   test that suggested otherwise (G3) was too small to show it.

## Predictions (PLAN.md)

| # | prediction | outcome |
|---|---|---|
| P1 | K=1 control matches `Quantizer(mean=)`, near round 3 | **held**: identical nDCG (0.7304 / 0.746), median score diff 1.2e-7; pf4 2-bit 0.746 vs round-3 four-seed 0.751 |
| P2 | K=4096 − K=1, pf4 1 bit, SciFact: +0.005 to +0.015 | slightly above: +0.016 |
| P3 | same at 2 bits: 0 to +0.008 | above on SciFact (+0.014), held on NFCorpus (+0.000) |
| P4 | e-means-2 − Lloyd-25 within ±0.005 | 20 of 24 |
| P5 | Lloyd-2 − Lloyd-25 within ±0.005 | 22 of 24 |
| P6 | e-means-2 5–15× sooner at K=16384 | held: 11.5–12.3× |
| P7 | K=16384 loses at equal bytes to K=1 one bit up | **wrong**: ties (+0.005, −0.004, both CIs span zero) |
| G1 | numpy e-means sooner only where k·d ≳ 1e5 | **wrong**: at d ≥ 32 it is sooner from k = 16 (k·d = 512); at d = 3 it is slower from k = 256 regardless of k·d |
| G2 | best error < Lloyd-25 for k ≥ 1,024 | mostly; fails where n/k < 25 (3 cells, up to +38%) |
| G3 | torch CPU not faster | **wrong** at scale (finding 7) |

## Method

`bench.py`: re-encoded with `lateon-pool-remex/encode.py` (the `.work` cache does not survive a session;
SciFact fp32 reproduces round 1). sentence-transformers 6.3 `HierarchicalTokenPooling` pf4 / pf2; remex 2.2.0
codebooks and rotation (seed 42); remex 2's centred estimator applied per cell,
`q·c + c·r + ||r|| (q−c)·û / κ`, with fp32 `||r||` and `c·r` (8 B per token, as `RemexIndex` stores);
brute-force MaxSim; centroid seeds 0 and 1, shared random init per seed; faiss-cpu 1.16.0 (`contrib/e_means`
ships in the wheel). `grid.py`: 1,000-component Gaussian mixture, n ∈ {1e4, 1e5, 1e6}, k 16–16,384,
d ∈ {3, 32, 128, 768}, plus Starry Night pixels and LateOn pf4 tokens; cells with n·k·d > 2.5e11 or k > n/8
skipped (listed in `results_grid.jsonl`). `colour_bench.py`: passes-to-error check on two images, before the grid.
`analyze.py` writes `summary.json` and the figures.

## Reproduce

```bash
python3 ../lateon-pool-remex/encode.py            # ~30 min, writes ../lateon-pool-remex/.work
python3 bench.py scifact && python3 bench.py nfcorpus   # resumable; ~2 h SciFact, ~1.5 h NFCorpus on 4 vCPU
python3 grid.py                                    # ~46 min
python3 analyze.py
```

## Not done

One rotation seed; two centroid seeds. No corpus large enough to show the table amortizing (the 35% figure is
row bytes only). No search-speed measurement. The JS/WGSL e-means port for the browser tracer is untested; the
grid measured faiss's kernels only. Synthetic data uses random init; with k-means++ init Lloyd would start
closer and e-means' error advantage on the mixture would likely shrink.
