// src/constants.js:53-65
// hypvector@0.2.2, sha256 5ac1eb31e4e81b19...
// why cited: Degeneracy guard rationale: below dimension * 1/16 expected Hamming, phase-1 ranking degenerates. README: 'A degeneracy guard on the sign codes'.

// Auto-binary also requires the sign-bit codes to discriminate: when the
// expected Hamming distance between two random vectors falls below
// dimension * defaultBinaryMinSpread, phase-1 binary ranking is dominated
// by ties (embeddings with all-non-negative components share one code) and
// approximate search degenerates to a near-random candidate pick that no
// rerankFactor can repair. Exact scan is correct and no slower in that
// regime. 1/16 sits far below healthy embeddings (mixed-sign embeddings
// measure ~0.3-0.5 of dimension) and far above degenerate ones (~0).
export const defaultBinaryMinSpread = 1 / 16

// Rows sampled (evenly strided) when estimating the sign-bit spread. Per-bit
// one-frequencies converge fast, so a few thousand rows suffice at any N.
export const binarySpreadSample = 4096
