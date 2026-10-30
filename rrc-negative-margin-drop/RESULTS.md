# Negative-margin direction drop on EmbeddingGemma 2 / Quora

RRC ([arXiv 2610.37707](https://arxiv.org/abs/2610.37707)) compresses dense-retrieval embeddings by projecting them onto their principal directions. It uses relevance labels to drop the directions whose average contribution separates relevant from non-relevant documents in the wrong direction, and to reweight the rest. The question here is whether that drop, or the full method, improves the embeddings we already hold.

## Answer

No, at least not for EmbeddingGemma 2 on Quora. No direction among the meaningful 512 has a negative margin, so the drop removes nothing, and the full reweighting adds at most 0.13 nDCG@10 at 512 dimensions. <!-- @evidence[[ndcg.json#PCA_rrc.k512.delta]] --> At 128 dimensions the model's own Matryoshka prefix stays the better cut: 91.49, against 90.03 for PCA. <!-- @evidence[[ndcg.json#MRL_prefix.k128.ndcg]] -->

The test did turn up something worth adopting. EG2's text tower has hidden size 512 and projects linearly to 768, so every text embedding, query or document, lies in one fixed 512-dimensional subspace. Rotating into that subspace and storing 512 floats is exact for dot-product search: it keeps 1.0 of the 768-dim top-10 neighbours, <!-- @evidence[[span512.json#recall10_span_512]] --> where the Matryoshka prefix at 512 keeps 0.926. <!-- @evidence[[span512.json#recall10_mrl_512]] -->

A reasoning-retrieval benchmark such as BRIGHT, where the paper reports its gains, could change the RRC answer. We hold no embeddings for one.

## Findings

1. EG2 text embeddings span exactly 512 dimensions. The singular values fall from 3.4e-4 of the largest at rank 512 to 3.4e-8 at rank 513, and query vectors have no component outside the document span beyond float noise (max residual 1.2e-7). The cause is architectural: `text_config.hidden_size` is 512 and `embedding_dim` is 768. (`span512.py`)
2. Rotating into the 512 span is lossless for nearest-neighbour search, while the Matryoshka prefix at 512 is not: 1.0 against 0.926 top-10 agreement with 768-dim exact search. Truncating inside the span is also better at 384 (0.955 vs 0.890) and 256 (0.868 vs 0.851), but worse at 128 (0.682 vs 0.715). (`span512.py`)
3. The negative-margin drop is a no-op. Every direction among the leading 512 has a positive margin; at k=768 it keeps 651 of 768 directions, <!-- @evidence[[ndcg.json#kept_768]] --> and the 117 it drops are all in the null space, so no score changes. (`rrc_quora.py`)
4. Full RRC gains at most 0.13 nDCG@10 (k=512, PCA prior) and nothing at k ≤ 256. The k=768 deltas (−3.13 for PCA, +0.83 for whitening) are artifacts of fitting a power law across null-space eigenvalues. (`rrc_quora.py`)
5. On nDCG, the Matryoshka prefix beats every spectral method at k=128 (91.49 vs PCA 90.03, whitening 90.97) and is within 0.08 of full dimension at k=512. (`rrc_quora.py`)

## Method

Fixture: `quora/embeddinggemma-2/fp32-768-first100k` from the embedding store (100k BEIR Quora docs, EG2 text-only fp32 768d). 885 of the 1,000 test queries have a relevant doc inside the 100k prefix, giving 1,747 positives. Each query's own corpus id is excluded from ranking.

RRC arms follow the paper. The PCA basis is fitted on the corpus without labels, with five-fold cross-fitting over queries and seeds 1999, 5 and 2026. Each positive is paired with one negative: 80% drawn from the query's full-dimension top-200 pool, 20% uniformly from the corpus. Margin means and variances are query-balanced, and the estimated weight is `(m)+ / v`. "drop" zeroes directions with non-positive weight and keeps the prior. "rrc" also multiplies the prior by the exponentiated residual of a power-law fit in log eigenvalue. The priors tested are PCA (γ=0), γ=0.5 and whitening (γ=1), plus direct ê and the MRL prefix as references. Scoring is cosine. The metric is nDCG@10 ×100, with a paired t-test over queries against the prior. Eigenvalues are floored at 1e-8·λ₁ for the power-law fit.

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

PCA at 512 equals PCA at 768, which led to the eigenvalue check and then the span check. PCA's 0.12 loss against full dimension comes from centring the queries and renormalizing, not from truncation: the uncentred rotation in the span check is exact.

### 2026-10-30: the 512-dim span

Oskar asked whether there was anything worth adopting. The config shows `text_config.hidden_size` 512 and `embedding_dim` 768. `span512.py` confirms that queries lie in the document span and that dot-product search in the rotated 512 coordinates reproduces the 768-dim top 10 exactly. The prior METHODS.md entry, "Matryoshka truncation can be free for nDCG and still cost nearest-neighbour recall", measured 0.926 for the prefix at 512; the rotation removes that cost.
