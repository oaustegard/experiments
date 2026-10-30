# Negative-margin direction drop on EmbeddingGemma 2 / Quora

Tests the support-selection step of RRC ([arXiv 2610.37707](https://arxiv.org/abs/2610.37707)) on embeddings already in the store: `quora/embeddinggemma-2/fp32-768-first100k` (100k BEIR Quora docs; 885 of the 1,000 test queries have a relevant doc inside that prefix, 1,747 positives). Run 2026-10-30, `python3 -I rrc_quora.py <set dir>`, about 10 minutes on 4 vCPU.

Protocol follows the paper: corpus PCA basis without labels; five-fold cross-fitting over queries with three seeds (1999, 5, 2026); one negative per positive, 80% from the query's full-dim top-200 pool and 20% uniform from the corpus; query-balanced margin mean and variance per direction; the weight is `(m)+ / v`. "drop" zeroes directions with non-positive weight and keeps the prior; "rrc" also adds the log-residual from a power-law fit. Priors: PCA (γ=0), γ=0.5 and whitening (γ=1). Cosine scoring with the query's own corpus id excluded. Metric is nDCG@10 ×100; `*` marks p<.05 on a paired t-test over queries against the prior.

| method | k=768 | k=512 | k=256 | k=128 |
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

## Findings

The drop is a no-op here. Within the leading 512 directions every one has a positive estimated margin, so nothing is dropped at k ≤ 512. At k=768 it removes 117 directions, but they carry no corpus variance (next point), so scores do not change.

EmbeddingGemma 2 embeddings on this corpus span about 511 dimensions. Centred eigenvalues fall from 2.1e-4 at rank 384 to 6.6e-8 at rank 512 and 1e-17 beyond it; 100.000% of the variance sits in the first 512. PCA to 512 therefore loses nothing against PCA at 768 (92.30 both), and the residual-weighting results at k=768 come from fitting a power law across the numerical null space (floored at 1e-8·λ₁). They are artifacts, not effects.

Residual weighting gives at most +0.13–0.14 at k=512 and nothing at k ≤ 256. MRL prefix truncation beats every spectral method at k=128 and loses only 0.08 at 512, so for this model the native Matryoshka cut stays the right compressor.

## Limits

Quora is a symmetric duplicate-question task near ceiling (92.4). The paper's gains come on BRIGHT reasoning retrieval with 4B–8B encoders, and it reports small changes on BEIR too, which this run agrees with. We hold no BRIGHT embeddings in the store, and the bsky set has no qrels, so this does not test the case where the paper claims RRC helps.
