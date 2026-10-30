# Negative-margin direction drop on EmbeddingGemma 2 / Quora

Relevance-Guided Residual Calibration, or RRC ([arXiv 2610.37707](https://arxiv.org/abs/2610.37707)), compresses dense-retrieval embeddings by projecting them onto their principal directions. It uses relevance labels to drop some of those directions and reweight the rest. A direction's *margin* is how much it adds, on average, to a relevant document's score minus a non-relevant one's. RRC drops the directions whose margin is not positive. The question here is whether that drop, or the full method, improves the EmbeddingGemma 2 (EG2) embeddings we already hold.

Terms: *k* is the number of dimensions kept. nDCG@10 is the standard ranking-quality score (×100 here, 100 = perfect). The *Matryoshka prefix* (MRL) keeps the first k coordinates, which EG2 was trained to support. BRIGHT is a benchmark of reasoning-heavy retrieval queries, where the paper reports its largest gains.

## Answer

No, at least not for EG2 on Quora. The drop removes no direction among the 512 that carry any signal, so scores do not change. The full reweighting adds 0.13 nDCG@10 at k=512 with the PCA prior, <!-- @evidence[[ndcg.json#PCA_rrc.k512.delta]] --> and no more than 0.14 with any prior at k ≤ 512. At 128 dimensions the model's own Matryoshka prefix stays the better cut: 91.49, against 90.03 for PCA. <!-- @evidence[[ndcg.json#MRL_prefix.k128.ndcg]] -->

The test did turn up something worth adopting. EG2's text tower has hidden size 512 and projects linearly to 768, so every text embedding, query or document, lies in one fixed 512-dimensional subspace. Rotating into that subspace and storing 512 floats is exact for dot-product search: it keeps 1.0 of the 768-dim top-10 neighbours, <!-- @evidence[[span512.json#recall10_span_512]] --> where the Matryoshka prefix at 512 keeps 0.926. <!-- @evidence[[span512.json#recall10_mrl_512]] -->

A reasoning-retrieval benchmark such as BRIGHT, where the paper reports its gains, could change the RRC answer. We hold no embeddings for one.

## Findings

1. EG2 text embeddings span exactly 512 dimensions. The 512th singular value is 3.4e-4 times the largest and the 513th is 3.4e-8 times it, and query vectors have no component outside the document span beyond float noise (max residual 1.2e-7). The cause is architectural: `text_config.hidden_size` is 512 and `embedding_dim` is 768. (`span512.py`)
2. Rotating into the 512 span is lossless for nearest-neighbour search, while the Matryoshka prefix at 512 is not: 1.0 against 0.926 top-10 agreement with 768-dim exact search. Truncating inside the span is also better at 384 (0.955 vs 0.890) and 256 (0.868 vs 0.851), but worse at 128 (0.682 vs 0.715). (`span512.py`)
3. The negative-margin drop is a no-op. At k ≤ 512 it kept every direction in every fold. At k=768 it keeps 651 of 768 directions. <!-- @evidence[[ndcg.json#kept_768]] --> The 256 directions beyond rank 512 carry no variance (the *null space*), so their margins are float noise, and about half come out non-positive. Dropping those 117 changes no score. (`rrc_quora.py`)
4. Full RRC gains at most 0.14 nDCG@10 (k=512; +0.13 with the PCA prior, the only significant one) and nothing at k ≤ 256. The k=768 deltas (−3.13 for PCA, +0.83 for whitening) come from fitting the power law across null-space eigenvalues, which are floored at 1e-8 of the largest and so dominate the fit. That is my reading of why; no run isolates it. Treat k=768 as uninformative. (`rrc_quora.py`)
5. On nDCG, the Matryoshka prefix beats every spectral method at k=128 (91.49 vs PCA 90.03, whitening 90.97) and is within 0.08 of full dimension at k=512. (`rrc_quora.py`)

## Method

Fixture: `quora/embeddinggemma-2/fp32-768-first100k` from the embedding store (100k BEIR Quora docs, EG2 text-only fp32 768d). 885 of the 1,000 test queries have a relevant doc inside the 100k prefix, giving 1,747 positives. Each query's own corpus id is excluded from ranking.

RRC arms follow the paper. The PCA basis is fitted on the corpus without labels, with five-fold cross-fitting over queries and seeds 1999, 5 and 2026. Each positive is paired with one negative: 80% drawn from the query's full-dimension top-200 pool, 20% uniformly from the corpus. Margin means and variances are query-balanced, and the estimated weight ê is `(m)+ / v` (positive part of the mean margin over its variance). "direct ê" uses ê itself as the weights. "drop" zeroes directions with non-positive weight and keeps the prior. "rrc" also multiplies the prior by the exponentiated residual of a power-law fit in log eigenvalue. The priors tested are PCA (γ=0), γ=0.5 and whitening (γ=1), plus direct ê and the MRL prefix as references. Scoring is cosine. The metric is nDCG@10 ×100, with a paired t-test over queries against the prior; `*` marks p < .05. Eigenvalues are floored at 1e-8·λ₁ for the power-law fit.

Span check: SVD of the 100k document matrix, basis = top 512 right singular vectors. The score is the raw dot product in the rotated coordinates with no renormalization, and the metric is top-10 overlap with exact 768-dim dot-product search, over all 1,000 queries.

Cost: about 10 minutes of 4-vCPU CPU for `rrc_quora.py` and about 1 minute for `span512.py`. No labels were used for the span check.

## Log

### 2026-10-30: does the drop help our embeddings?

Oskar asked to try the paper's negative-margin drop on our embeddings. The only labelled set in the store is Quora. The first run crashed at k=768 because float32 eigenvalues in the null space went negative; computing the covariance in float64 and flooring the eigenvalues fixed it.

| nDCG@10 ×100 | k=768 | k=512 | k=256 | k=128 |
|---|---|---|---|---|
| full dim | 92.42 | | | |
| MRL prefix | 92.42 | 92.34 | 91.82 | 91.49 |
| PCA | 92.30 | 92.30 | 92.05 | 90.03 |
| PCA+drop | +0.00 | +0.00 | +0.00 | +0.00 |
| PCA+rrc | −3.13* | +0.13* | −0.02 | −0.06 |
| γ0.5 | 92.23 | 92.23 | 92.42 | 90.75 |
| γ0.5+rrc | −0.51* | −0.05 | −0.07 | −0.06 |
| whiten | 91.56 | 91.56 | 92.29 | 90.97 |
| whiten+rrc | +0.83* | +0.14 | −0.07 | +0.05 |
| direct ê | 91.96 | 91.95 | 92.19 | 91.03 |

Full dim has one value because it is uncompressed. PCA at 512 equals PCA at 768, which led to the eigenvalue check and then the span check. PCA's 0.12 loss against full dimension comes from centring the queries and renormalizing, not from truncation: the uncentred rotation in the span check is exact.

### 2026-10-30: the 512-dim span

Oskar asked whether there was anything worth adopting. The config shows `text_config.hidden_size` 512 and `embedding_dim` 768. `span512.py` confirms that queries lie in the document span (largest query residual outside it: 1.2e-7) and that dot-product search in the rotated coordinates reproduces the 768-dim top 10 exactly.

| top-10 agreement with 768-dim | k=512 | k=384 | k=256 | k=128 |
|---|---|---|---|---|
| rotation into the span, truncated | 1.000 | 0.955 | 0.868 | 0.682 |
| Matryoshka prefix | 0.926 | 0.890 | 0.851 | 0.715 |

The RRC table scores cosine on centred vectors; this table scores the raw dot product, uncentred, which is what makes it exact. The prior METHODS.md entry, "Matryoshka truncation can be free for nDCG and still cost nearest-neighbour recall", measured 0.926 for the prefix at 512; the rotation removes that cost.
