### Scorecard

| prediction | SmolLM2-135M | Qwen2.5-0.5B | Qwen2.5-1.5B |
|---|---|---|---|
| P1 | **WRONG** | **WRONG** | **WRONG** |
| P2 | **RIGHT** | **RIGHT** | **RIGHT** |
| P3 | **RIGHT** | **RIGHT** | **RIGHT** |
| P4 | **WRONG** | **WRONG** | **WRONG** |
| P5 | **WRONG** | **WRONG** | **WRONG** |
| P6 | **PARTIAL** | **WRONG** | **PARTIAL** |

### SmolLM2-135M (30 layers, d=576, 3780 prompts, 49.3 s)

Median |K| / |h(AB) − h0| over 20 pairs × 3 templates (real) and 20 random-noun pairs × 3 templates; layers = 1/2, 2/3, last−2.

| condition | L15 | L20 | L28 |
|---|---|---|---|
| COMPOUND / real | 0.369 | 0.347 | 0.483 |
| COMPOUND / random | 0.293 | 0.257 | 0.350 |
| CONJUNCTION / real | 0.304 | 0.291 | 0.337 |
| CONJUNCTION / random | 0.258 | 0.245 | 0.289 |
| LIST / real | 0.458 | 0.414 | 0.524 |
| LIST / random | 0.360 | 0.322 | 0.404 |
| **P1 ratio COMPOUND/CONJUNCTION, real (pair-level)** | 1.19 | 1.20 | 1.49 |
| ratio, random nouns | 1.13 | 1.06 | 1.21 |
| ratio of ratios (real / random), p | 1.06 (p=0.44) | 1.12 (p=0.21) | 1.23 (p=0.23) |
| pairs with COMPOUND > CONJUNCTION (of 20), perm p | 16 (p=0.006) | 16 (p=0.017) | 19 (p=0.001) |
| P1 ratio by template (real) | 1.22/1.12/1.28 | 1.17/1.17/1.30 | 1.49/1.25/1.72 |

Commutator direction (P2 and the positional prong): PC1 explained-variance of the 60 COMPOUND commutators, mean |cos| between them, and how much of K_COMPOUND is the same-pair CONJUNCTION / LIST commutator.

| quantity | L15 | L20 | L28 |
|---|---|---|---|
| PC1 ratio, COMPOUND K (raw / unit-normed) | 0.17 / 0.17 | 0.21 / 0.19 | 0.17 / 0.16 |
| PC1 ratio per template, n=20 each | 0.21/0.23/0.24 | 0.24/0.27/0.31 | 0.24/0.19/0.21 |
| PC1 ratio, CONJUNCTION K | 0.13 | 0.15 | 0.13 |
| mean abs cos between COMPOUND commutators | 0.18 | 0.18 | 0.16 |
| median cos(K_COMPOUND, K_CONJUNCTION) same pair | 0.35 | 0.25 | 0.26 |
| median cos(K_COMPOUND, K_LIST) same pair, real / random | 0.46 / 0.49 | 0.39 / 0.43 | 0.52 / 0.36 |
| K_COMPOUND outside span{K_CONJ, K_LIST}: own pair / other pair | 0.83 / 0.99 | 0.89 / 0.98 | 0.82 / 0.99 |

P3 (K outside span{h0, hA, hB}) and P5 (additive fit R²), medians; `frame` adds a free frame-mean term (post-hoc, see text).

| quantity | L15 | L20 | L28 |
|---|---|---|---|
| P3 outside, own spokes / other pair's spokes | 0.914 / 0.981 | 0.923 / 0.977 | 0.928 / 0.983 |
| P3 own − other, perm p | -0.065 (p=0.003) | -0.052 (p=0.003) | -0.061 (p=0.001) |
| P5 R² COMPOUND / CONJUNCTION | 0.888 / 0.754 | 0.909 / 0.745 | 0.866 / 0.761 |
| P5 pairs with CONJUNCTION higher (of 20), perm p | 0 (p=0.002) | 0 (p=0.002) | 4 (p=0.023) |
| P5 R² random nouns COMPOUND / CONJUNCTION | 0.908 / 0.738 | 0.915 / 0.708 | 0.896 / 0.749 |
| P5 R² with frame term, COMPOUND / CONJUNCTION, p | 0.932 / 0.882 (p=0.005) | 0.937 / 0.902 (p=0.003) | 0.880 / 0.803 (p=0.065) |

P4 (three-word compositions): median fraction of h(ABC) − h0 outside span{3 spokes, 6 ordered pairs}, 12 triples × 6 orders × 3 templates.

| condition | L15 | L20 | L28 |
|---|---|---|---|
| COMPOUND / real | 0.130 | 0.128 | 0.152 |
| COMPOUND / random | 0.113 | 0.115 | 0.124 |
| CONJUNCTION / real | 0.269 | 0.250 | 0.245 |
| CONJUNCTION / random | 0.253 | 0.250 | 0.263 |
| LIST / real | 0.300 | 0.304 | 0.252 |
| LIST / random | 0.252 | 0.268 | 0.233 |
| COMPOUND − CONJUNCTION real, perm p | -0.138 (p=0.018) | -0.117 (p=0.018) | -0.096 (p=0.016) |
| same with frame term | -0.111 (p=0.018) | -0.090 (p=0.015) | -0.092 (p=0.016) |

P6 (behaviour): median KL(next-token | AB ∥ BA) at the `:` token.

| condition | real pairs | random nouns |
|---|---|---|
| COMPOUND | 0.0935 | 0.0591 |
| CONJUNCTION | 0.0401 | 0.0266 |
| LIST | 0.1554 | 0.0905 |
| COMPOUND / CONJUNCTION | 2.17 (perm p=0.001) | 2.74 |
| ratio of ratios | 0.79 (p=0.38) | |

Layer sweep of the P1 ratio (real / random nouns): L1: 1.28/1.70, L2: 0.88/0.80, L3: 0.68/0.59, L4: 0.84/0.65, L5: 0.81/0.78, L6: 0.76/0.71, L7: 0.66/0.68, L8: 0.71/0.63, L9: 0.65/0.60, L10: 0.67/0.68, L11: 0.70/0.67, L12: 0.87/0.89, L13: 0.91/0.92, L14: 1.21/1.18, L15: 1.21/1.14, L16: 1.14/1.12, L17: 1.09/1.04, L18: 1.14/1.02, L19: 1.28/1.13, L20: 1.19/1.05, L21: 1.16/1.08, L22: 1.30/1.12, L23: 1.30/1.22, L24: 1.24/1.18, L25: 1.26/1.17, L26: 1.31/1.21, L27: 1.33/1.26, L28: 1.44/1.21, L29: 1.45/1.22, L30: 1.14/0.99

Single-token pairs: 20/20. Gap only at the last layer: False.

### Qwen2.5-0.5B (24 layers, d=896, 3780 prompts, 135.8 s)

Median |K| / |h(AB) − h0| over 20 pairs × 3 templates (real) and 20 random-noun pairs × 3 templates; layers = 1/2, 2/3, last−2.

| condition | L12 | L16 | L22 |
|---|---|---|---|
| COMPOUND / real | 0.474 | 0.478 | 0.552 |
| COMPOUND / random | 0.310 | 0.319 | 0.374 |
| CONJUNCTION / real | 0.344 | 0.368 | 0.384 |
| CONJUNCTION / random | 0.289 | 0.317 | 0.357 |
| LIST / real | 0.419 | 0.462 | 0.508 |
| LIST / random | 0.345 | 0.339 | 0.390 |
| **P1 ratio COMPOUND/CONJUNCTION, real (pair-level)** | 1.40 | 1.28 | 1.31 |
| ratio, random nouns | 1.10 | 0.98 | 1.01 |
| ratio of ratios (real / random), p | 1.28 (p=0.01) | 1.30 (p=0.04) | 1.30 (p=0.08) |
| pairs with COMPOUND > CONJUNCTION (of 20), perm p | 19 (p=0.001) | 16 (p=0.001) | 17 (p=0.007) |
| P1 ratio by template (real) | 1.11/1.77/1.24 | 0.90/1.87/1.21 | 1.00/2.00/1.25 |

Commutator direction (P2 and the positional prong): PC1 explained-variance of the 60 COMPOUND commutators, mean |cos| between them, and how much of K_COMPOUND is the same-pair CONJUNCTION / LIST commutator.

| quantity | L12 | L16 | L22 |
|---|---|---|---|
| PC1 ratio, COMPOUND K (raw / unit-normed) | 0.19 / 0.15 | 0.15 / 0.15 | 0.14 / 0.12 |
| PC1 ratio per template, n=20 each | 0.19/0.30/0.29 | 0.20/0.25/0.24 | 0.20/0.18/0.18 |
| PC1 ratio, CONJUNCTION K | 0.12 | 0.15 | 0.15 |
| mean abs cos between COMPOUND commutators | 0.15 | 0.14 | 0.13 |
| median cos(K_COMPOUND, K_CONJUNCTION) same pair | 0.27 | 0.22 | 0.17 |
| median cos(K_COMPOUND, K_LIST) same pair, real / random | 0.29 / 0.27 | 0.28 / 0.24 | 0.33 / 0.23 |
| K_COMPOUND outside span{K_CONJ, K_LIST}: own pair / other pair | 0.90 / 0.98 | 0.93 / 0.99 | 0.91 / 0.99 |

P3 (K outside span{h0, hA, hB}) and P5 (additive fit R²), medians; `frame` adds a free frame-mean term (post-hoc, see text).

| quantity | L12 | L16 | L22 |
|---|---|---|---|
| P3 outside, own spokes / other pair's spokes | 0.926 / 0.981 | 0.926 / 0.979 | 0.904 / 0.977 |
| P3 own − other, perm p | -0.063 (p=0.001) | -0.051 (p=0.001) | -0.071 (p=0.001) |
| P5 R² COMPOUND / CONJUNCTION | 0.842 / 0.753 | 0.869 / 0.742 | 0.847 / 0.800 |
| P5 pairs with CONJUNCTION higher (of 20), perm p | 1 (p=0.001) | 0 (p=0.003) | 2 (p=0.019) |
| P5 R² random nouns COMPOUND / CONJUNCTION | 0.892 / 0.768 | 0.889 / 0.722 | 0.886 / 0.755 |
| P5 R² with frame term, COMPOUND / CONJUNCTION, p | 0.890 / 0.886 (p=0.863) | 0.894 / 0.872 (p=0.065) | 0.857 / 0.823 (p=0.014) |

P4 (three-word compositions): median fraction of h(ABC) − h0 outside span{3 spokes, 6 ordered pairs}, 12 triples × 6 orders × 3 templates.

| condition | L12 | L16 | L22 |
|---|---|---|---|
| COMPOUND / real | 0.206 | 0.202 | 0.203 |
| COMPOUND / random | 0.145 | 0.158 | 0.159 |
| CONJUNCTION / real | 0.310 | 0.358 | 0.324 |
| CONJUNCTION / random | 0.299 | 0.380 | 0.353 |
| LIST / real | 0.338 | 0.370 | 0.343 |
| LIST / random | 0.329 | 0.348 | 0.343 |
| COMPOUND − CONJUNCTION real, perm p | -0.107 (p=0.012) | -0.157 (p=0.018) | -0.120 (p=0.018) |
| same with frame term | -0.061 (p=0.014) | -0.096 (p=0.012) | -0.095 (p=0.018) |

P6 (behaviour): median KL(next-token | AB ∥ BA) at the `:` token.

| condition | real pairs | random nouns |
|---|---|---|
| COMPOUND | 0.2955 | 0.1114 |
| CONJUNCTION | 0.1166 | 0.1110 |
| LIST | 0.3034 | 0.1807 |
| COMPOUND / CONJUNCTION | 1.69 (perm p=0.002) | 0.90 |
| ratio of ratios | 1.88 (p=0.11) | |

Layer sweep of the P1 ratio (real / random nouns): L1: 0.81/0.82, L2: 0.72/0.74, L3: 0.88/0.83, L4: 0.88/0.81, L5: 1.25/1.05, L6: 1.47/1.19, L7: 1.44/1.07, L8: 1.48/1.07, L9: 1.28/0.98, L10: 1.35/1.01, L11: 1.26/1.02, L12: 1.38/1.07, L13: 1.35/1.07, L14: 1.32/1.03, L15: 1.37/1.07, L16: 1.30/1.01, L17: 1.23/0.95, L18: 1.26/0.96, L19: 1.27/0.97, L20: 1.30/1.00, L21: 1.38/1.00, L22: 1.44/1.05, L23: 1.43/1.01, L24: 1.34/1.03

Single-token pairs: 20/20. Gap only at the last layer: False.

### Qwen2.5-1.5B (28 layers, d=1536, 3780 prompts, 301.7 s)

Median |K| / |h(AB) − h0| over 20 pairs × 3 templates (real) and 20 random-noun pairs × 3 templates; layers = 1/2, 2/3, last−2.

| condition | L14 | L19 | L26 |
|---|---|---|---|
| COMPOUND / real | 0.540 | 0.557 | 0.657 |
| COMPOUND / random | 0.395 | 0.387 | 0.494 |
| CONJUNCTION / real | 0.416 | 0.423 | 0.447 |
| CONJUNCTION / random | 0.384 | 0.375 | 0.396 |
| LIST / real | 0.492 | 0.494 | 0.522 |
| LIST / random | 0.350 | 0.298 | 0.291 |
| **P1 ratio COMPOUND/CONJUNCTION, real (pair-level)** | 1.23 | 1.24 | 1.46 |
| ratio, random nouns | 1.03 | 1.02 | 1.27 |
| ratio of ratios (real / random), p | 1.20 (p=0.03) | 1.21 (p=0.02) | 1.14 (p=0.05) |
| pairs with COMPOUND > CONJUNCTION (of 20), perm p | 17 (p=0.001) | 16 (p=0.014) | 16 (p=0.003) |
| P1 ratio by template (real) | 1.12/1.41/1.27 | 0.99/1.59/1.44 | 0.99/1.87/1.68 |

Commutator direction (P2 and the positional prong): PC1 explained-variance of the 60 COMPOUND commutators, mean |cos| between them, and how much of K_COMPOUND is the same-pair CONJUNCTION / LIST commutator.

| quantity | L14 | L19 | L26 |
|---|---|---|---|
| PC1 ratio, COMPOUND K (raw / unit-normed) | 0.16 / 0.14 | 0.13 / 0.13 | 0.14 / 0.13 |
| PC1 ratio per template, n=20 each | 0.17/0.21/0.31 | 0.18/0.20/0.29 | 0.18/0.15/0.20 |
| PC1 ratio, CONJUNCTION K | 0.11 | 0.16 | 0.24 |
| mean abs cos between COMPOUND commutators | 0.12 | 0.12 | 0.12 |
| median cos(K_COMPOUND, K_CONJUNCTION) same pair | 0.23 | 0.21 | 0.19 |
| median cos(K_COMPOUND, K_LIST) same pair, real / random | 0.29 / 0.29 | 0.19 / 0.13 | 0.21 / 0.09 |
| K_COMPOUND outside span{K_CONJ, K_LIST}: own pair / other pair | 0.93 / 0.99 | 0.95 / 0.99 | 0.95 / 0.99 |

P3 (K outside span{h0, hA, hB}) and P5 (additive fit R²), medians; `frame` adds a free frame-mean term (post-hoc, see text).

| quantity | L14 | L19 | L26 |
|---|---|---|---|
| P3 outside, own spokes / other pair's spokes | 0.934 / 0.977 | 0.936 / 0.970 | 0.916 / 0.967 |
| P3 own − other, perm p | -0.041 (p=0.001) | -0.035 (p=0.001) | -0.059 (p=0.002) |
| P5 R² COMPOUND / CONJUNCTION | 0.808 / 0.719 | 0.819 / 0.699 | 0.764 / 0.670 |
| P5 pairs with CONJUNCTION higher (of 20), perm p | 3 (p=0.002) | 2 (p=0.002) | 6 (p=0.010) |
| P5 R² random nouns COMPOUND / CONJUNCTION | 0.845 / 0.727 | 0.857 / 0.686 | 0.826 / 0.630 |
| P5 R² with frame term, COMPOUND / CONJUNCTION, p | 0.863 / 0.834 (p=0.429) | 0.858 / 0.833 (p=0.321) | 0.792 / 0.765 (p=0.528) |

P4 (three-word compositions): median fraction of h(ABC) − h0 outside span{3 spokes, 6 ordered pairs}, 12 triples × 6 orders × 3 templates.

| condition | L14 | L19 | L26 |
|---|---|---|---|
| COMPOUND / real | 0.256 | 0.265 | 0.274 |
| COMPOUND / random | 0.193 | 0.194 | 0.196 |
| CONJUNCTION / real | 0.346 | 0.386 | 0.389 |
| CONJUNCTION / random | 0.348 | 0.399 | 0.437 |
| LIST / real | 0.332 | 0.360 | 0.322 |
| LIST / random | 0.323 | 0.360 | 0.318 |
| COMPOUND − CONJUNCTION real, perm p | -0.089 (p=0.018) | -0.124 (p=0.014) | -0.119 (p=0.016) |
| same with frame term | -0.049 (p=0.012) | -0.066 (p=0.012) | -0.081 (p=0.018) |

P6 (behaviour): median KL(next-token | AB ∥ BA) at the `:` token.

| condition | real pairs | random nouns |
|---|---|---|
| COMPOUND | 0.4538 | 0.1788 |
| CONJUNCTION | 0.1577 | 0.1433 |
| LIST | 0.3318 | 0.1350 |
| COMPOUND / CONJUNCTION | 2.13 (perm p=0.002) | 1.35 |
| ratio of ratios | 1.58 (p=0.16) | |

Layer sweep of the P1 ratio (real / random nouns): L1: 0.97/0.90, L2: 0.84/0.84, L3: 0.97/0.89, L4: 1.32/1.01, L5: 1.30/0.96, L6: 1.37/1.00, L7: 1.48/1.05, L8: 1.43/1.08, L9: 1.52/1.22, L10: 1.51/1.22, L11: 1.45/1.21, L12: 1.38/1.15, L13: 1.37/1.10, L14: 1.30/1.03, L15: 1.31/1.04, L16: 1.27/1.02, L17: 1.28/0.99, L18: 1.33/1.03, L19: 1.32/1.03, L20: 1.33/1.03, L21: 1.35/1.07, L22: 1.39/1.14, L23: 1.44/1.16, L24: 1.42/1.17, L25: 1.40/1.18, L26: 1.47/1.25, L27: 1.48/1.26, L28: 1.41/1.22

Single-token pairs: 20/20. Gap only at the last layer: False.

