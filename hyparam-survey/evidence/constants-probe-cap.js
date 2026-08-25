// src/constants.js:33-50
// hypvector@0.2.2, sha256 5ac1eb31e4e81b19...
// why cited: THE CLAIM THIS SURVEY TESTS: 'Residual misses are a rerankFactor limit, not a probe limit.' Also the defaultClusterProbeCap = 48 rationale.

// Upper bound on clusters probed under the *default* fraction. Clusters grow
// as ~sqrt(N)/2, so 0.25 x nlist keeps rising with N, but the clusters needed
// to reach the recall ceiling stay roughly flat (~25-45) regardless of N. A
// WildChat 1024-dim sweep found 48, 72, and 96 lists give statistically
// indistinguishable recall@10 at 1M and 3.2M (within ~1pp over 20 exact-scan
// queries, no consistent direction). Their top-10 sets are not bit-identical:
// over 200 queries, cap 48 matches cap 96 on ~93% (1M) to ~97% (3.2M), the
// rest reshuffling near-ties at the list boundary, not losing true neighbors.
// Capping at 48 reads ~42% fewer bytes than 96 at scale with no measurable
// recall loss; structurally, shrinking the cap can only lose recall, never
// gain it, since probed clusters are a subset. Residual misses are a
// rerankFactor limit, not a probe limit. Only applies when `probe` is left
// default; an explicit `probe` is honored literally.
export const defaultClusterProbeCap = 48

// When `binary` is not specified at write time, the column is added once
// the corpus is at least this large. Below the threshold, exact full scan
// is fast enough that the rerank path's overhead isn't worth the column.
