# RESULTS: compressing MICE-P's stored document vectors to a 2-bit budget

MICE-P ([Vast, Morand et al., arXiv 2610.31932](https://arxiv.org/abs/2610.31932),
code at [xpmir/mice](https://github.com/xpmir/mice)) is a reranker. It precomputes each
document's contextualised token vectors and keeps them, 128-d, in FastPLAID used only as a
compressed vector store; SPLADE-v3 retrieves the top 100, and MICE-P's four interaction
layers let query tokens cross-attend to those candidates' stored vectors. The paper sets
FastPLAID to 4-bit residuals for MICE-P because 2-bit loses 0.024 nDCG@10 on average (its
Figure 1), while LateOn, the ColBERT it is compared with, loses nothing at 2 bits. Tom
Aarsen's [thread on the release](https://bsky.app/profile/tomaarsen.com/post/3myu7ijgk6k2m)
puts the trade plainly: MICE-P scores above LateOn + PLAID on BEIR13 but stores more. The
question here is whether a different codec brings MICE-P's per-token storage down to
LateOn's 2-bit budget without the loss. One round, 2026-10-27; errors in
[`ERRORS.md`](ERRORS.md).

## Answer

Yes, on four of the five datasets the paper used to pick the bit width (ArguAna was not
scored). FastPLAID's codec fits MICE-P's vectors badly in two ways. The vectors have norm
0.43 to 0.47 against unit-length centroids, and FastPLAID does not normalise its input,
though MICE-P reads a vector only through `LayerNorm(W_up x)`, where the norm carries
nothing. And its per-dimension residual buckets are coarse at 2 bits.

Keeping PLAID's centroid and coding the residual with remex (random rotation, Lloyd-Max
codes of the residual direction, an fp16 length) at 2 bits, arm `plaid-rx-2`, costs
38 B/token against 36 for LateOn's 2-bit PLAID. Over 401 queries it changes nDCG@10 by
-0.001 [-0.006, +0.005] against uncompressed vectors. <!-- @evidence[[results/summary.json#pooled.plaid-rx-2.delta]] -->
PLAID 2-bit loses -0.031 on the same queries, <!-- @evidence[[results/summary.json#pooled.plaid-2.delta]] -->
and PLAID 4-bit, the shipped setting at 68 B/token, loses -0.008. <!-- @evidence[[results/summary.json#pooled.plaid-4.delta]] -->
Per dataset, its ranking agrees with the uncompressed one at Spearman 0.989 to 0.991
(PLAID 4-bit 0.9625 to 0.9769). Normalising alone recovers under half the 2-bit loss.

Not settled: the BEIR13 average, total index size (MICE-P also keeps a SPLADE index) and
decode latency. The other nine BEIR13 datasets through the authors' pipeline would settle the first.

## Findings

Pooled deltas weight each query once; "mean over datasets" averages the four per-dataset deltas, as the paper's Figure 1 does.

1. **PLAID centroid + remex 2-bit residual (`plaid-rx-2`) removes the 2-bit loss** at 38 B/token (Round 1).
   Over PLAID 2-bit, pooled: +0.031 [+0.016, +0.045]. <!-- @evidence[[results/paired.json#plaid-rx-2__plaid-2.delta]] -->
   Over PLAID 4-bit, pooled: +0.008 [-0.001, +0.016]. <!-- @evidence[[results/paired.json#plaid-rx-2__plaid-4.delta]] -->
   Per dataset against fp32: SciFact -0.003, NFCorpus +0.000, FiQA -0.009, SciDocs +0.002,
   every CI spanning zero; Spearman 0.991 / 0.989 / 0.990 / 0.991.
2. **Normalising before FastPLAID (`plaid-n-2`) recovers under half the loss; coding the
   residual with remex recovers the rest** (Round 1, `results/vector_stats.json`).
   Stored vectors have mean norm 0.463 (sd 0.054) on SciFact. <!-- @evidence[[results/vector_stats.json#scifact.norm_mean]] -->
   FastPLAID's centroids are unit-norm, so each residual starts with about 0.54 along its
   centroid. Normalising first cuts the proxy error (squared direction error at the input
   of the interaction layers' LayerNorm) from 0.082 to 0.022 at 2 bits. It cuts the pooled
   scored loss from -0.031 to -0.018, a paired gain of +0.014 [+0.002, +0.026]. <!-- @evidence[[results/paired.json#plaid-n-2__plaid-2.delta]] -->
   Replacing PLAID's per-dimension buckets with remex on the residual adds +0.017 [+0.007, +0.026]. <!-- @evidence[[results/paired.json#plaid-rx-2__plaid-n-2.delta]] -->
3. **The PLAID re-implementation reproduces the paper's direction and roughly its size**
   (Round 1). PLAID 2-bit, ours then the paper's: NFCorpus -0.025 / -0.019, SciDocs -0.025 /
   -0.014, FiQA -0.084 / -0.044, SciFact +0.001 / -0.018 (ours inside a +/-0.03 CI).
   PLAID 4-bit, mean over datasets: -0.007 (paper -0.012). <!-- @evidence[[results/summary.json#pooled.plaid-4.mean_delta_over_datasets]] -->
4. **Centred remex with no centroid table (`remex-c-2`) ties PLAID 4-bit at half the bytes**
   (34 B/token, Round 1). Pooled against fp32: -0.007 [-0.015, +0.002]. <!-- @evidence[[results/summary.json#pooled.remex-c-2.delta]] -->
   Against PLAID 4-bit: +0.002 [-0.008, +0.012]. <!-- @evidence[[results/paired.json#remex-c-2__plaid-4.delta]] -->
   Its ranking agreement is lower (Spearman 0.936 to 0.961), so it trails `plaid-rx-2`.
5. **At 1 bit (`plaid-rx-1`, 22 B/token, under LateOn's budget) the same codec is level with
   PLAID 4-bit** (Round 1; FiQA not scored at 1 bit).
   Paired over 340 queries: +0.009 [-0.002, +0.019]. <!-- @evidence[[results/paired.json#plaid-rx-1__plaid-4.delta]] -->
6. **The harness reproduces the paper's uncompressed numbers** (Round 1). nDCG@10
   NFCorpus 0.2512 (paper 0.249), FiQA 0.5249 (0.541), <!-- @evidence[[results/summary.json#nfcorpus.fp32.ndcg10]] -->
   SciFact 0.8544 (0.856). <!-- @evidence[[results/summary.json#scifact.fp32.ndcg10]] -->
   Scoring from stored 128-d vectors matches the CrossEncoder path to 4e-6 (`worker.py check`).
7. **A basis fitted to the model's own weights was no better than plain centring**
   (Round 1, `results/arms_scifact.json`). Quantising in the SVD basis of the up-projection
   gave proxy error 0.041 against centring's 0.040 at 2 bits on SciFact; the basis that also
   folds in each layer's K/V weights gave 0.059. Neither was scored. <!-- @evidence[[results/arms_scifact.json#remex-wkv-2.err_up]] -->

## Method

**Model and inputs.** `xpmir/Mice-P-130m` (Ettin-150m, 12 contextualisation + 4
interaction layers, 128-d document bottleneck). Documents are `[CLS] title text [SEP]`
truncated to 512 tokens and queries `[CLS] q [SEP]` to 32, the same truncation the HF
CrossEncoder applies. The stored vector is `doc_compress(normalize(center(h)))`, which is
what MICE-P hands FastPLAID. The model's remote code runs only inside the hub's
`scripts/jail.sh` (no network, no secrets); `worker.py` encodes and scores there.

**Retrieval.** SPLADE-v3-DistilBERT top-100 (documents to 256 tokens, queries to 64), the
paper's first stage, re-ranked by MICE-P with each codec's decoded vectors. Data are the
LightOn decontaminated BEIR sets the paper uses. Only queries with at least one positive
judgement count (SciDocs 130 of 201), so SciDocs nDCG is higher than the paper's, which
averages over all 201. The paper's text names five datasets for choosing the bit width and
its Figure 1 plots four; ArguAna (1,373 queries) was left out for CPU time.

**Arms.** Every codec is fitted per dataset on the candidate documents' tokens, as an index
build would be. `<bits>` is the residual or code width.
- `plaid-<bits>`: FastPLAID's residual codec re-implemented. faiss k-means with 4
  iterations (FastPLAID's default), K = 2^floor(log2(16 sqrt(N))) (8,192 to 16,384), unit
  centroids; code = argmax x.c; residual buckets from global quantiles; decode
  normalise(c + w). Inputs are not normalised, as in FastPLAID.
- `plaid-n-<bits>`: the same on L2-normalised inputs.
- `plaid-rx-<bits>`: normalised input and the same centroid, with the residual coded by
  remex 2.1.0 (`Quantizer(128, bits=b)`) as a unit direction plus an fp16 length. Coding a
  residual against a k-means centroid follows RSLM ([arXiv 2608.30384](https://arxiv.org/abs/2608.30384)).
- `remex-c-<bits>`: remex on x minus the corpus mean, with an fp16 length and no centroid.

**Bytes per token.** Code bits × 128 / 8, plus 4 B for a centroid id and 2 B for a stored
length. FastPLAID writes centroid ids as int64, so on its disk format every centroid arm
(and LateOn) is 4 B larger: PLAID 2-bit 40, `plaid-rx-2` 42.

**Metrics.** nDCG@10 with linear gains, and a paired bootstrap (10,000 resamples) for every
delta. Ranking agreement with fp32 is computed per query: Spearman over the 100 candidates
and top-10 overlap. It needs no judgements, so it separates arms that the 41-query SciFact
delta cannot. Proxy error is the mean squared distance between unit-normalised `P W_up x`
and `P W_up x_hat`, where P removes the per-token mean, so it measures the input the
LayerNorm sees.

**Cost.** 4 vCPU. Encoding 11,000 documents took about 1.6 h in the jail; scoring ran 5 to
16 min per arm per dataset. SPLADE for SciDocs, FiQA and ArguAna ran on a Colab T4.

## Log

### Round 1 (2026-10-27)

Asked: "Let's test if we can compress those vectors for them, regaining the lead from
LateOn." Planned arms were PLAID 2/4/8 as a reproduction, then plain, centred and
metric-aware remex. Reading FastPLAID's `create.rs` and `search.rs` turned up the
normalisation mismatch before anything ran; the `plaid-n` and `plaid-rx` arms came from that
and from RSLM's residual-against-centroid result.

Proxy errors on SciFact, sorted: `plaid-8` 0.001, `plaid-n-4` 0.005, `remex-4` 0.006,
`plaid-rx-2` 0.013, `plaid-4` 0.020, `plaid-n-2` 0.022, `remex-c-2` 0.040, `plaid-rx-1`
0.042, `remex-2` 0.075, `plaid-2` 0.082. NFCorpus and FiQA gave the same order to within
0.004. After scoring, ranking agreement kept the ends of that order (`plaid-rx-2` first
among the 2-bit arms, `plaid-2` last) and swapped two close pairs: `plaid-rx-1` agrees with
fp32 better than `remex-c-2` on every dataset it was scored on, and `plaid-4` against
`plaid-n-2` splits two datasets each. Pooled nDCG reversed one pair the proxy and Spearman
agree on: `plaid-n-2` loses -0.018 and `remex-c-2` -0.007.

| dataset (queries) | arm | B/tok | nDCG@10 | delta vs fp32 [95% CI] | Spearman | top-10 |
|---|---|---|---|---|---|---|
| SciFact (41) | fp32 | 512 | 0.8544 | | | |
|  | plaid-2 | 36 | 0.8553 | +0.0009 [-0.0291, +0.0328] | 0.9152 | 0.763 |
|  | plaid-4 | 68 | 0.8523 | -0.0021 [-0.0216, +0.0229] | 0.9716 | 0.888 |
|  | plaid-n-2 | 36 | 0.8431 | -0.0113 [-0.0402, +0.0172] | 0.9745 | 0.871 |
|  | plaid-rx-1 | 22 | 0.8483 | -0.0061 [-0.0445, +0.0319] | 0.9726 | 0.871 |
|  | plaid-rx-2 | 38 | 0.8510 | -0.0034 [-0.0289, +0.0243] | 0.9911 | 0.937 |
|  | remex-c-2 | 34 | 0.8547 | +0.0003 [-0.0380, +0.0424] | 0.9536 | 0.807 |
| NFCorpus (169) | fp32 | 512 | 0.2512 | | | |
|  | plaid-2 | 36 | 0.2265 | -0.0247 [-0.0411, -0.0100] | 0.8816 | 0.718 |
|  | plaid-4 | 68 | 0.2452 | -0.0060 [-0.0144, +0.0013] | 0.9625 | 0.859 |
|  | plaid-n-2 | 36 | 0.2416 | -0.0096 [-0.0174, -0.0026] | 0.9634 | 0.844 |
|  | plaid-n-4 | 68 | 0.2460 | -0.0052 [-0.0098, -0.0012] | 0.9900 | 0.921 |
|  | plaid-rx-1 | 22 | 0.2503 | -0.0009 [-0.0073, +0.0057] | 0.9661 | 0.851 |
|  | plaid-rx-2 | 38 | 0.2516 | +0.0004 [-0.0044, +0.0050] | 0.9893 | 0.920 |
|  | remex-c-2 | 34 | 0.2478 | -0.0034 [-0.0124, +0.0063] | 0.9362 | 0.798 |
| FiQA (61) | fp32 | 512 | 0.5249 | | | |
|  | plaid-2 | 36 | 0.4406 | -0.0843 [-0.1395, -0.0317] | 0.8934 | 0.741 |
|  | plaid-4 | 68 | 0.5186 | -0.0062 [-0.0292, +0.0167] | 0.9769 | 0.897 |
|  | plaid-n-2 | 36 | 0.4853 | -0.0395 [-0.0887, +0.0009] | 0.9708 | 0.862 |
|  | plaid-rx-2 | 38 | 0.5161 | -0.0088 [-0.0258, +0.0044] | 0.9897 | 0.928 |
|  | remex-c-2 | 34 | 0.5232 | -0.0017 [-0.0305, +0.0252] | 0.9550 | 0.843 |
| SciDocs (130) | fp32 | 512 | 0.1695 | | | |
|  | plaid-2 | 36 | 0.1441 | -0.0254 [-0.0500, -0.0034] | 0.9081 | 0.733 |
|  | plaid-4 | 68 | 0.1550 | -0.0145 [-0.0286, -0.0018] | 0.9748 | 0.860 |
|  | plaid-n-2 | 36 | 0.1502 | -0.0193 [-0.0343, -0.0055] | 0.9665 | 0.837 |
|  | plaid-rx-1 | 22 | 0.1717 | +0.0022 [-0.0148, +0.0192] | 0.9688 | 0.846 |
|  | plaid-rx-2 | 38 | 0.1715 | +0.0020 [-0.0088, +0.0137] | 0.9909 | 0.920 |
|  | remex-c-2 | 34 | 0.1546 | -0.0149 [-0.0307, +0.0009] | 0.9612 | 0.832 |

Not scored: `plaid-8` and `remex-2` on any dataset (SciFact's job hit its 2 h limit after
six arms); `plaid-n-4` except on NFCorpus; `plaid-rx-1` on FiQA; ArguAna entirely, whose
SPLADE run exists in `cache/runs/`.

Reading at the time: SciFact's 41 queries could not separate any arm (every CI about
+/-0.03), so ranking agreement with fp32 became the second measure. NFCorpus and SciDocs
carried the significance. The FiQA PLAID 2-bit loss (-0.084) is larger than the paper's
(-0.044); that dataset has 61 queries and its CI runs from -0.140 to -0.032.
