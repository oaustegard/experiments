# rabitq-vs-remex: faiss RaBitQ against remex, brute-force scan

**Started:** 2026-10-14 · **Status:** done (SPECTER2 broad and narrow, MiniLM) · **Runtime:** bench about 30 min on 4 vCPU (most of it faiss's qb=0 slow path) · **Scope:** mechanism only, flat scan. Vector-database engines (Qdrant, pgvector, FAISS RaBitQ at scale) are the job of [`../embeddinggemma2-vectordb/`](../embeddinggemma2-vectordb/); nothing here duplicates it. Algorithm notes: `PAPER_NOTES.md`.

## Answer

faiss RaBitQ's lead over remex is its estimator, not its quantizer: the query-centred residual product plus an exact `<mu,r>`. It ports to remex for 2 bytes per vector, after which the two tie on SPECTER2 at 2 to 14 fewer bytes than faiss. RaBitQ scores `q.mu + <mu,r> + ||r|| * <q-mu, u_hat> / c` for `r = x - mu`, so quantization error is multiplied by `||q-mu||` (8.4 on SPECTER2) instead of `||q||` (21.7), and `<mu,r>` is stored rather than read off the codes. Arm E gives remex's centered codes that formula (fp32 `||r||`, fp16 `<mu,r>`, one constant per width). On SPECTER2 broad, R@10 goes from 0.689 / 0.853 / 0.917 / 0.953 (remex centered) to 0.812 / 0.903 / 0.943 / 0.967 at 1 / 2 / 3 / 4 bits, against faiss 0.809 / 0.903 / 0.944 / 0.971. E minus faiss is +0.003 / 0.000 / 0.000 / -0.004 and every CI spans zero; narrow is the same. On MiniLM E leads at 1 bit (+0.059) and ties above. The stored `<o_bar,o>` division alone (arm C) is not the cause: it is unbiased yet 0.04 to 0.17 worse than remex centered.

What would change it: a real MiniLM-scale corpus, more rotation seeds, a corpus where centering does not help, or IVF residuals (excluded by scope).

## Findings

1. **E ties faiss on SPECTER2 and is ahead or tied on MiniLM, with fewer bytes.** E minus D4 (faiss qb=4) on R@10, broad: +0.003 [-0.008, +0.013], 0.000 [-0.008, +0.008], 0.000 [-0.006, +0.006], -0.004 [-0.009, +0.001] at 1 to 4 bits. Narrow: 0.000, -0.003, -0.003, +0.001 (CIs span zero). MiniLM: +0.059 [+0.045, +0.072], then +0.004, +0.007, +0.006 (CIs span zero). Bytes: E 102 / 198 / 294 / 390 against faiss 104 / 212 / 308 / 404, so faiss costs 2 B more at 1 bit and 14 B more at 2 to 4 bits (2 B more than E at every width in the paper's own 104 / 200 / 296 / 392 format). The nearest-bytes pair is always the same width, so matched bits and matched bytes are one comparison. (`results.json`, Round 2)
2. **The mechanism is the scoring formula.** A numpy 1-bit RaBitQ reproduces faiss (broad R@10 .814, estimator RMS error 1.960) only when scored as `q.mu + <mu,r> + ||r|| <(q-mu)R, o_bar> / c`; using the full `q` in the residual term gives R@10 .50 and RMS 5.08. `||q||` is 21.7 and `||q-mu||` 8.4, which is the 2.6x cut in the error that multiplies the quantization noise, and the exact `<mu,r>` removes a second source (remex B's `<mu, r_hat>` error has RMS 4.0). These probes were run by the parent session in numpy and are not in `bench.py`; arm E is the in-bench port of the result. (parent-session probes `rq3` to `rq6`, Round 2)
3. **E costs 2 B over remex centered, and uses neither of remex's stored columns.** remex's centered `norms` column is neither `||x||` nor `||x-mu||`: it is the `m` solving `||mu + m*u_hat|| = ||x||` (`Quantizer._centred_lengths`). `||r||` and `<mu,r>` cannot both be recovered from it and the codes (one equation, two unknowns; `<mu,r>` computed from the codes collapses to R@10 .59 in the parent's probe), so E stores `||r||` as fp32 and `<mu,r>` as fp16, 6 B against B's 4 B. fp16 versus fp32 `<mu,r>` changes R@10 by at most 0.001. E also keeps one constant per width, `E[c]` = corpus mean of `cos(u_hat, u)`: 0.799 / 0.940 / 0.983 / 0.995. (`results.json` arm E, Round 2)
4. **The `1/E[c]` correction only matters at 1 bit.** E minus E-without-correction (En), broad: +0.038 [+0.030, +0.045] at 1 bit, +0.006 [+0.002, +0.011] at 2, +0.001 and 0.000 at 3 and 4. Narrow is the same shape (+0.043 at 1 bit); MiniLM +0.016 at 1 bit. (`results.json`, Round 2)
5. **Arm C is the negative result that ruled out the `<o_bar,o>` division as the cause.** Cc (centered codes, stored per-vector `c`) is unbiased (all-pairs mean error at 2-bit: +0.118, B +0.094, A1 -24.2; c identical across frames to 1e-7) yet worse than B: Cc minus B on broad R@10 is -0.167 / -0.107 / -0.072 / -0.041, all CIs below zero, same sign on narrow and MiniLM. Cu minus A1 is -0.047 [-0.057, -0.036] at 1 bit, -0.011 at 2 and zero at 3 and 4. Finding 2 explains it: Cc multiplies the noise by `||q||` and reads `<mu,r>` off the codes. Within each query's true top 300 at 2-bit, error std is 2.22 for Cc, 1.42 for B and 0.82 for E (correlation 0.898, 0.949, 0.984). (`results.json` estimator_check, Round 1 and 2)
6. **The quantized direction is not the cause.** remex's Lloyd-Max cosine to the true direction is 0.9393 at 2-bit against 0.9386 for a uniform grid with per-vector rescale search (my numpy re-implementation of the paper's codebook; 0.9829 vs 0.9815 at 3-bit, 0.9954 vs 0.9945 at 4-bit, identical at 1 bit). faiss's `sa_decode` only decodes the 1-bit part at every width, so its own multi-bit cosine cannot be read, but with finding 2 the formula already accounts for the gap. (`results.json` cosine, Round 1)
7. **Before E, faiss won on SPECTER2 and the gap narrowed with bits.** D4 minus B, broad R@10: +0.120 / +0.050 / +0.026 / +0.017 (narrow +0.117 / +0.045 / +0.031 / +0.010), all CIs above zero; on MiniLM -0.038 at 1 bit and ties above. E minus B is +0.124 / +0.050 / +0.026 / +0.014 on broad. (`results.json`, Round 1)
8. **Rotation seed noise is below the SPECTER2 conclusions.** Across 3 seeds on broad, E moves at most 0.008 R@10 (1 bit) and faiss behind a random rotation at most 0.008; B at most 0.007. E minus D4 is inside that band everywhere; the Round-1 gaps (>= 0.010) sat above it. (`results.json` seed_spread)
9. **faiss's flat index is already centered on the corpus mean, a rotation does not help it, and qb is inert above 1 bit.** Default 2-bit RMS error 0.9618, identical (0.9618) when indexing X-mu and querying Q-mu, 2.587 when querying raw Q. `RandomRotationMatrix` changes R@10 by -0.007 to +0.006. Scores are identical across qb 0 / 4 / 8 at 2 to 4 bits; qb=0 means a float query but takes a path about 100x slower, so qb=4 is the headline. (`results.json` faiss_probes, Round 1)
10. **Costs.** Scan including top-100, us/query, broad: E 134 to 156 at every width (a float matmul over a cached decoded matrix, 29 MB, not a compressed-domain scan); faiss D4 136 / 186 at 1 / 2 bits and about 950 at 3 / 4 bits. Encode us/vector: B about 31, faiss 4 / 23 / 43 / 44 for 1 to 4 bits, E 79 to 91 (it recomputes the rotation for `E[c]`, so this overstates a fused encoder). (`results.json`, Round 2)
11. **Shipped in remex 2.0.0.** oaustegard/remex 2.0.0 (commit b3a0ed9) ships arm E as its centred search, with `<mu,r>` stored as fp32 instead of fp16 (104 / 200 / 296 / 392 B per vector) and `E[c]` computed from the codebook rather than the corpus. Through the library, SPECTER2 broad R@10 is 0.812 / 0.903 / 0.943 / 0.967 at 1 to 4 bits. The numbers in this directory were measured against remex 1.1.0, so rerunning `bench.py` on 2.0 changes arms B and C.

### Tables

All arms, SPECTER2 broad (9,500 docs, 500 queries, d=768). Bytes: remex = packed codes + 4 B length (+2 B for C and E); faiss = `sa_code_size`.

| bits | arm | B/vec | R@10 | R@100 | encode us/vec | scan+top100 us/query |
|---|---|---|---|---|---|---|
| 1 | A0 remex renorm=False | 100 | 0.624 | 0.689 | 10 | 150 |
| 1 | A1 remex renorm=True | 100 | 0.624 | 0.689 | 11 | 111 |
| 1 | B remex centered | 100 | 0.689 | 0.750 | 35 | 138 |
| 1 | Cu remex + c | 102 | 0.577 | 0.661 | 54 | 107 |
| 1 | Cc remex centered + c | 102 | 0.522 | 0.651 | 73 | 158 |
| 1 | E remex centered, RaBitQ scoring | 102 | 0.812 | 0.867 | 91 | 153 |
| 1 | En E without 1/Ec | 102 | 0.775 | 0.833 | 79 | 116 |
| 1 | Df faiss qb=0 | 104 | 0.815 | 0.865 | 3 | 11777 |
| 1 | D4 faiss qb=4 | 104 | 0.809 | 0.862 | 3 | 154 |
| 1 | Dr0 faiss+rot qb=0 | 104 | 0.818 | 0.867 | 6 | 11647 |
| 1 | Dr4 faiss+rot qb=4 | 104 | 0.815 | 0.863 | 6 | 149 |
| 2 | A0 remex renorm=False | 196 | 0.494 | 0.613 | 11 | 156 |
| 2 | A1 remex renorm=True | 196 | 0.765 | 0.818 | 11 | 129 |
| 2 | B remex centered | 196 | 0.853 | 0.882 | 32 | 150 |
| 2 | Cu remex + c | 198 | 0.754 | 0.812 | 48 | 116 |
| 2 | Cc remex centered + c | 198 | 0.746 | 0.822 | 84 | 151 |
| 2 | E remex centered, RaBitQ scoring | 198 | 0.903 | 0.933 | 85 | 134 |
| 2 | En E without 1/Ec | 198 | 0.896 | 0.928 | 91 | 159 |
| 2 | Df faiss qb=0 | 212 | 0.903 | 0.932 | 24 | 17512 |
| 2 | D4 faiss qb=4 | 212 | 0.903 | 0.932 | 24 | 173 |
| 2 | Dr0 faiss+rot qb=0 | 212 | 0.901 | 0.934 | 27 | 17259 |
| 2 | Dr4 faiss+rot qb=4 | 212 | 0.901 | 0.934 | 27 | 192 |
| 3 | A0 remex renorm=False | 292 | 0.602 | 0.711 | 12 | 146 |
| 3 | A1 remex renorm=True | 292 | 0.864 | 0.898 | 13 | 126 |
| 3 | B remex centered | 292 | 0.917 | 0.937 | 42 | 168 |
| 3 | Cu remex + c | 294 | 0.862 | 0.896 | 121 | 115 |
| 3 | Cc remex centered + c | 294 | 0.846 | 0.904 | 78 | 125 |
| 3 | E remex centered, RaBitQ scoring | 294 | 0.943 | 0.965 | 79 | 151 |
| 3 | En E without 1/Ec | 294 | 0.943 | 0.964 | 99 | 127 |
| 3 | Df faiss qb=0 | 308 | 0.944 | 0.962 | 44 | 13689 |
| 3 | D4 faiss qb=4 | 308 | 0.944 | 0.962 | 44 | 963 |
| 3 | Dr0 faiss+rot qb=0 | 308 | 0.941 | 0.963 | 44 | 13942 |
| 3 | Dr4 faiss+rot qb=4 | 308 | 0.941 | 0.963 | 44 | 969 |
| 4 | A0 remex renorm=False | 388 | 0.735 | 0.820 | 14 | 133 |
| 4 | A1 remex renorm=True | 388 | 0.923 | 0.945 | 11 | 115 |
| 4 | B remex centered | 388 | 0.953 | 0.968 | 35 | 133 |
| 4 | Cu remex + c | 390 | 0.924 | 0.944 | 126 | 119 |
| 4 | Cc remex centered + c | 390 | 0.913 | 0.950 | 77 | 145 |
| 4 | E remex centered, RaBitQ scoring | 390 | 0.967 | 0.981 | 88 | 156 |
| 4 | En E without 1/Ec | 390 | 0.966 | 0.980 | 103 | 147 |
| 4 | Df faiss qb=0 | 404 | 0.971 | 0.978 | 45 | 13578 |
| 4 | D4 faiss qb=4 | 404 | 0.971 | 0.978 | 45 | 1055 |
| 4 | Dr0 faiss+rot qb=0 | 404 | 0.964 | 0.979 | 49 | 13499 |
| 4 | Dr4 faiss+rot qb=4 | 404 | 0.964 | 0.979 | 49 | 1009 |

SPECTER2 narrow, R@10 / R@100 (bytes/vec):

| bits | A1 (B/vec) | B (B/vec) | Cc (B/vec) | E (B/vec) | En (B/vec) | D4 (B/vec) |
|---|---|---|---|---|---|---|
| 1 | 0.668 / 0.755 (100) | 0.711 / 0.792 (100) | 0.521 / 0.686 (102) | 0.828 / 0.888 (102) | 0.786 / 0.866 (102) | 0.828 / 0.888 (104) |
| 2 | 0.789 / 0.852 (196) | 0.862 / 0.905 (196) | 0.723 / 0.840 (198) | 0.904 / 0.945 (198) | 0.900 / 0.942 (198) | 0.907 / 0.943 (212) |
| 3 | 0.868 / 0.916 (292) | 0.919 / 0.950 (292) | 0.834 / 0.913 (294) | 0.947 / 0.970 (294) | 0.947 / 0.970 (294) | 0.950 / 0.968 (308) |
| 4 | 0.926 / 0.954 (388) | 0.960 / 0.974 (388) | 0.913 / 0.955 (390) | 0.970 / 0.984 (390) | 0.970 / 0.985 (390) | 0.969 / 0.981 (404) |

MiniLM 10k (d=384), R@10 / R@100 (bytes/vec):

| bits | A1 (B/vec) | B (B/vec) | Cc (B/vec) | E (B/vec) | En (B/vec) | D4 (B/vec) |
|---|---|---|---|---|---|---|
| 1 | 0.615 / 0.878 (52) | 0.625 / 0.881 (52) | 0.468 / 0.838 (54) | 0.645 / 0.889 (54) | 0.629 / 0.885 (54) | 0.586 / 0.882 (56) |
| 2 | 0.757 / 0.926 (100) | 0.778 / 0.935 (100) | 0.646 / 0.904 (102) | 0.783 / 0.936 (102) | 0.780 / 0.936 (102) | 0.779 / 0.937 (116) |
| 3 | 0.853 / 0.958 (148) | 0.867 / 0.963 (148) | 0.779 / 0.945 (150) | 0.870 / 0.963 (150) | 0.871 / 0.963 (150) | 0.864 / 0.962 (164) |
| 4 | 0.918 / 0.978 (196) | 0.927 / 0.980 (196) | 0.876 / 0.970 (198) | 0.928 / 0.981 (198) | 0.928 / 0.981 (198) | 0.921 / 0.979 (212) |

Paired bootstrap, R@10, 95% CI, 1000 resamples over queries. D4 is faiss qb=4; the byte gap in the E column is faiss minus E, and the nearest-bytes pairing is always the same width, so it doubles as the matched-bytes comparison. `results.json` also holds R@100 and the faiss-behind-rotation arm.

SPECTER2 broad:

| bits | Cu - A1 | Cc - B | D4 - B | E - D4 (faiss minus E bytes) | E - B | E - En |
|---|---|---|---|---|---|---|
| 1 | -0.047 [-0.057, -0.036] | -0.167 [-0.177, -0.157] | +0.120 [+0.109, +0.132] | +0.003 [-0.008, +0.013] (+2 B) | +0.124 [+0.113, +0.134] | +0.038 [+0.030, +0.045] |
| 2 | -0.011 [-0.018, -0.003] | -0.107 [-0.115, -0.099] | +0.050 [+0.040, +0.058] | -0.000 [-0.008, +0.008] (+14 B) | +0.050 [+0.042, +0.057] | +0.006 [+0.002, +0.011] |
| 3 | -0.002 [-0.007, +0.003] | -0.072 [-0.078, -0.065] | +0.026 [+0.019, +0.033] | -0.000 [-0.006, +0.006] (+14 B) | +0.026 [+0.020, +0.033] | +0.001 [-0.001, +0.003] |
| 4 | +0.001 [-0.002, +0.005] | -0.041 [-0.045, -0.036] | +0.017 [+0.011, +0.023] | -0.004 [-0.009, +0.001] (+14 B) | +0.014 [+0.009, +0.018] | +0.000 [-0.001, +0.002] |

SPECTER2 narrow:

| bits | Cu - A1 | Cc - B | D4 - B | E - D4 (faiss minus E bytes) | E - B | E - En |
|---|---|---|---|---|---|---|
| 1 | -0.050 [-0.062, -0.040] | -0.190 [-0.201, -0.181] | +0.117 [+0.105, +0.129] | +0.000 [-0.010, +0.010] (+2 B) | +0.118 [+0.107, +0.128] | +0.043 [+0.036, +0.050] |
| 2 | -0.009 [-0.017, -0.002] | -0.139 [-0.149, -0.130] | +0.045 [+0.036, +0.054] | -0.003 [-0.011, +0.005] (+14 B) | +0.042 [+0.035, +0.049] | +0.004 [-0.000, +0.008] |
| 3 | -0.001 [-0.006, +0.004] | -0.086 [-0.093, -0.079] | +0.031 [+0.024, +0.038] | -0.003 [-0.009, +0.003] (+14 B) | +0.028 [+0.022, +0.034] | +0.001 [-0.001, +0.003] |
| 4 | -0.000 [-0.003, +0.003] | -0.046 [-0.052, -0.041] | +0.010 [+0.004, +0.015] | +0.001 [-0.004, +0.007] (+14 B) | +0.011 [+0.007, +0.015] | -0.000 [-0.001, +0.001] |

MiniLM:

| bits | Cu - A1 | Cc - B | D4 - B | E - D4 (faiss minus E bytes) | E - B | E - En |
|---|---|---|---|---|---|---|
| 1 | -0.043 [-0.053, -0.032] | -0.157 [-0.170, -0.144] | -0.038 [-0.052, -0.024] | +0.059 [+0.045, +0.072] (+2 B) | +0.021 [+0.014, +0.029] | +0.016 [+0.010, +0.023] |
| 2 | -0.005 [-0.012, +0.002] | -0.132 [-0.142, -0.121] | +0.001 [-0.011, +0.012] | +0.004 [-0.007, +0.016] (+14 B) | +0.005 [-0.000, +0.010] | +0.003 [-0.000, +0.007] |
| 3 | +0.004 [-0.002, +0.009] | -0.088 [-0.097, -0.079] | -0.003 [-0.012, +0.006] | +0.007 [-0.002, +0.015] (+14 B) | +0.003 [+0.000, +0.006] | -0.001 [-0.003, +0.001] |
| 4 | +0.002 [-0.001, +0.005] | -0.052 [-0.058, -0.045] | -0.006 [-0.013, +0.001] | +0.006 [-0.001, +0.013] (+14 B) | +0.000 [-0.002, +0.003] | -0.000 [-0.002, +0.001] |

Rotation-seed spread, R@10, seeds 42 / 7 / 123 (unrotated faiss is deterministic and has no seed). Broad, then narrow, then MiniLM:

| bits | A1 R@10 (3 seeds) | B | E | Dr4 faiss+rotation |
|---|---|---|---|---|
| 1 | 0.624, 0.637, 0.632 | 0.689, 0.683, 0.690 | 0.812, 0.817, 0.809 | 0.816, 0.813, 0.808 |
| 2 | 0.765, 0.771, 0.773 | 0.853, 0.848, 0.852 | 0.903, 0.897, 0.901 | 0.901, 0.903, 0.900 |
| 3 | 0.864, 0.865, 0.862 | 0.917, 0.918, 0.919 | 0.943, 0.945, 0.945 | 0.942, 0.944, 0.944 |
| 4 | 0.923, 0.919, 0.920 | 0.953, 0.954, 0.951 | 0.967, 0.970, 0.970 | 0.968, 0.968, 0.966 |

| bits | A1 R@10 (3 seeds) | B | E | Dr4 faiss+rotation |
|---|---|---|---|---|
| 1 | 0.668, 0.678, 0.676 | 0.711, 0.715, 0.710 | 0.828, 0.825, 0.829 | 0.824, 0.828, 0.825 |
| 2 | 0.789, 0.789, 0.788 | 0.862, 0.861, 0.859 | 0.904, 0.906, 0.910 | 0.906, 0.909, 0.907 |
| 3 | 0.868, 0.876, 0.866 | 0.919, 0.926, 0.922 | 0.947, 0.951, 0.948 | 0.949, 0.947, 0.945 |
| 4 | 0.926, 0.923, 0.922 | 0.960, 0.961, 0.955 | 0.970, 0.975, 0.968 | 0.972, 0.969, 0.970 |

| bits | A1 R@10 (3 seeds) | B | E | Dr4 faiss+rotation |
|---|---|---|---|---|
| 1 | 0.615, 0.620, 0.613 | 0.625, 0.620, 0.613 | 0.645, 0.645, 0.641 | 0.591, 0.604, 0.602 |
| 2 | 0.757, 0.761, 0.754 | 0.778, 0.781, 0.778 | 0.783, 0.785, 0.780 | 0.774, 0.779, 0.787 |
| 3 | 0.853, 0.861, 0.859 | 0.867, 0.868, 0.870 | 0.870, 0.869, 0.872 | 0.862, 0.865, 0.866 |
| 4 | 0.918, 0.919, 0.917 | 0.927, 0.929, 0.931 | 0.928, 0.928, 0.930 | 0.916, 0.923, 0.921 |

Estimator check, SPECTER2 broad, 200 queries, 2-bit then 4-bit (error = estimate minus exact `<q,x>`):

| estimator | mean signed err (all pairs) | RMS err (all) | RMS err (top-100) | within-query err std (top-300) | within-query corr |
|---|---|---|---|---|---|
| A1 | -24.192 | 24.400 | 26.618 | 2.168 | 0.891 |
| B | +0.094 | 2.443 | 2.450 | 1.419 | 0.949 |
| Cu_fp16 | +0.003 | 3.204 | 2.377 | 2.351 | 0.886 |
| Cc_fp16 | +0.118 | 2.394 | 2.253 | 2.221 | 0.898 |
| E | +0.002 | 0.934 | 0.804 | 0.818 | 0.984 |

| estimator | mean signed err (all pairs) | RMS err (all) | RMS err (top-100) | within-query err std (top-300) | within-query corr |
|---|---|---|---|---|---|
| A1 | -1.746 | 1.952 | 2.065 | 0.659 | 0.989 |
| B | -0.004 | 0.669 | 0.392 | 0.392 | 0.996 |
| Cu_fp16 | +0.091 | 0.880 | 0.635 | 0.668 | 0.989 |
| Cc_fp16 | -0.003 | 0.653 | 0.614 | 0.615 | 0.991 |
| E | -0.000 | 0.250 | 0.214 | 0.219 | 0.999 |

Direction cosine, SPECTER2 broad, first 2,000 docs:

| bits | remex Lloyd-Max cos (uncentered) | same, centered | uniform grid + rescale search (uncentered) | same, centered | faiss sa_decode (1-bit part only) |
|---|---|---|---|---|---|
| 1 | 0.7967 +- 0.0055 | 0.7989 +- 0.0077 | 0.7967 | 0.7989 | 0.7976 |
| 2 | 0.9393 +- 0.0026 | 0.9400 +- 0.0034 | 0.9386 | 0.9393 | 0.7976 |
| 3 | 0.9829 +- 0.0010 | 0.9828 +- 0.0013 | 0.9815 | 0.9815 | 0.7976 |
| 4 | 0.9954 +- 0.0003 | 0.9953 +- 0.0005 | 0.9945 | 0.9944 | 0.7976 |

### What would change the result

- A real MiniLM-scale corpus: the MiniLM rows come from a templated, saturated fixture, and E's 1-bit lead there is the one result that depends on it.
- A growing corpus: E's `mu`, `E[c]` and per-vector `<mu,r>` are computed from a static corpus, so online adds would need the constants to drift or be refit.
- More rotation seeds and a second query split: three seeds bound the noise only roughly.
- faiss's own multi-bit cosine and estimator internals: finding 6 uses a reimplementation, and finding 2 rests on the parent session's 1-bit numpy probes.
- IVF residuals, where RaBitQ is deployed; excluded by scope.

## Method

- **Data.** SPECTER2 broad and narrow caches (10,000 x 768, raw, not normalised), split exactly as `remex/bench/norm_correction_eval.py`: `default_rng(99).permutation`, first 500 rows queries, remaining 9,500 the corpus. MiniLM: `real_embedding_eval.load_real_embeddings(10000, 500)`, 10,000 corpus and 500 query sentences from templates. Ground truth is exact float inner-product top-k.
- **Metric.** R@k = fraction of the exact top-k found in the arm's top-k, per query, averaged. remex A and B recall is scored on `decode()` output (the template's definition); a fast rotated-space scorer is used for timing, C and E, and agrees with decode on top-10.
- **Arms.** A0 `Quantizer(renorm=False)`; A1 `renorm=True`; B `Quantizer(mean=corpus_mean, renorm=True)`; Cu / Cc keep the remex codes and store `c = <u_hat,u>` (fp16) with a fp32 length, score `length * <q,u_hat> / c` (+ `q.mu` centered); **E** keeps B's codes and scores `q.mu + m + (q-mu).u_hat_normalised * ||r|| / E[c]` with `r = x - mu`, `||r||` fp32, `m = <mu,r>` fp16, `E[c]` one constant per width; En is E without `1/E[c]`; Df / D4 `faiss.IndexRaBitQ(d, METRIC_INNER_PRODUCT, nb_bits)` with `RaBitQSearchParameters.qb` 0 / 4, `centered=False`; Dr0 / Dr4 the same behind `IndexPreTransform(RandomRotationMatrix)`. Rotation seed 42 for remex.
- **Statistics.** Paired bootstrap over the 500 queries, 1000 resamples, seed 12345, percentile CI. E minus D4 at matched bits; matched bytes takes the E width whose bytes are nearest the faiss size (always the same width). C minus its baseline, D4 minus B, E minus B, E minus En are also in `results.json`.
- **Timing.** Best of 2 (encode) or 3 (scan) on 4 vCPU with faiss using 4 OpenMP threads and numpy BLAS multithreaded. Scan numbers include top-100 for both libraries. Not a controlled microbenchmark.
- **Gate (redefined).** The harness reproduces the unmodified `norm_correction_eval.py` on this checkout exactly. Published values do not all reproduce: SPECTER2 broad 2-bit renorm=False .517 vs .494 here (rotation-seed band .479 to .506, numpy 2.5.2), renorm=True .773 vs .765, 4-bit .736 / .917 vs .735 / .923, MiniLM 4-bit .709 / .919 vs .723 / .918 (the .723 is 0.004 outside the +-0.01 tolerance). renorm=True and 4-bit reproduce within 0.008; the strict gate is recorded as failed in `results.json` and accepted as a harness-equivalence check, not chased.
- **Reproduce.** `python3 bench.py` (needs faiss-cpu 1.15.1, remex 1.1.0, the remex repo with `bench/.specter2_cache/`, sentence-transformers for MiniLM). Writes `results.json` and `results_perquery.npz`.

## Log

### Round 1, 2026-10-14

Asked: does RaBitQ's stored-correction estimator explain a gap to remex, in a flat scan on the same vectors. First pass failed the strict gate (2-bit renorm=False .494 vs .517); the unmodified template gave the same numbers, and six rotation seeds gave .479 to .506, so the gate was redefined as a harness-equivalence check and the run continued. Arm C looked buggy (Cc far below B); the estimator check found it unbiased, frame-consistent and worse-ranked, so no code change. Predicted before running: C would recover most of the gap to faiss; measured: it does not. The reading at the time (direction cosine does not explain the lead, cause unresolved) was incomplete and is superseded by Round 2. faiss's `sa_decode` returning the 1-bit part at every width was found while building the cosine diagnostic.

### Round 2, 2026-10-14

The parent session reproduced faiss's 1-bit estimator in numpy and found the lead is the scoring formula (finding 2). Added arm E and E-without-`1/E[c]` to `bench.py` on all three datasets, plus E in the seed spread, the estimator check and the bootstrap. Prediction from the probes (E about equals faiss, correction matters at 1 bit): measured broad E .812 / .903 / .943 / .967 and En .775 at 1 bit, matching the probe numbers to the third decimal. Reading: the 2 B extra over remex centered buys the whole faiss lead on SPECTER2; the `<o_bar,o>` division in the paper was not the part that mattered for this comparison.
