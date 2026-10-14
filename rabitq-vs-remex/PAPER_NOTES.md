# Extended RaBitQ (arXiv 2409.09913) and faiss 1.15.1 RaBitQ: notes for the remex comparison

Sources read in full or in the parts cited:
- Extended RaBitQ, Gao, Gou, Xu, Yang, Long, Wong, arXiv:2409.09913v1 (16 Sep 2024), HTML version, all main sections and Appendix C. Section and equation numbers below are from that version. The Appendix A and B proofs were not read: UNVERIFIED beyond the statements of Lemma 3.1 and Theorem 3.2.
- Original RaBitQ, Gao and Long, arXiv:2405.12497, read only for Theorem 3.2 (the 1-bit estimator), Theorem 3.3 and the query-quantization section.
- faiss v1.15.1 source from raw.githubusercontent.com, tag `v1.15.1`: `faiss/impl/RaBitQuantizer.{h,cpp}`, `RaBitQuantizerMultiBit.cpp`, `RaBitQUtils.{h,cpp}`, `faiss/IndexRaBitQ.{h,cpp}`. The installed Python package reports `faiss.__version__ == 1.15.1`.
- Not read: `IndexIVFRaBitQ`, the FastScan variants and the HNSW variant. Anything said about them is UNVERIFIED.

Notation: `o_r` is a raw data vector, `c` the centre, `o = (o_r-c)/||o_r-c||`, `q` the normalised query (the paper normalises the query the same way), `P` the random orthogonal matrix, `o' = P^-1 o`, `q' = P^-1 q`, `D` the dimension and `B` bits per dimension.

---

## 1. Encoding

**Distance reduction (ext. §2.2, Eq. 1-2).**
`||o_r - q_r||^2 = ||o_r-c||^2 + ||q_r-c||^2 - 2 ||o_r-c|| ||q_r-c|| <q,o>`.
Only the unit-vector inner product `<q,o>` is quantised. `||o_r-c||` is a stored fp32 scalar.

**Centering (§2.2, §5.1).**
- The centre `c` is "e.g. the centroid of a set of data vectors".
- Distance experiments (§5.2.1) use the global dataset centroid. IVF experiments (§5.2.2) use the local centroid of each IVF cluster.
- Both data and query are centred, then normalised.
- Faiss `IndexRaBitQ::train` computes `center` as the mean of the training data. `IndexRaBitQ.cpp` lines 32-50.

**Rotation (§2.2 Eq. 3, §3.2.1 Eq. 8).**
- The paper uses a random orthogonal matrix `P`, described as a Johnson-Lindenstrauss transformation.
- The codebook is never materialised. Only `P` is stored, and `P^-1` is applied to data and query.
- Appendix C says nothing about a fast structured transform; `P_d` is "the random orthogonal matrix which projects to d-dim". The paper only says "random orthogonal matrix", which I read as dense.
- Whether the authors' released code uses a Hadamard or other fast rotation is UNVERIFIED. The paper text does not say.
- **Faiss does not rotate inside `IndexRaBitQ`.** `RaBitQuantizer.h` says "It is assumed that the Random Matrix Rotation is performed externally." You wrap it with `IndexPreTransform(faiss.RandomRotationMatrix(d,d), IndexRaBitQ(...))`, which is what `index_factory("RR768,RaBitQ")` builds. `RandomRotationMatrix` is a dense orthogonal matrix.
- My probe (§2) confirms this. On anisotropic synthetic data, adding the rotation lowered the mean relative error at every `nb_bits`, e.g. 0.0263 to 0.0218 at 1 bit. The data were synthetic, so this only shows that the rotation matters.

**Codebook at B bits (§3.2.1, Eq. 7-8, Fig. 1).**
- Grid: `G = { -(2^B-1)/2 + u | u = 0..2^B-1 }^D`. That is the half-integer grid ±0.5, ±1.5, ..., ±(2^B-1)/2 in every coordinate.
- Codebook: `G_r = { P y/||y|| | y in G }`. The grid points are normalised to the unit sphere, then rotated, so the codebook is a set of randomly rotated unit vectors.
- `B=1` gives `y` in {±0.5}^D, which is exactly the original RaBitQ hypercube vertices.
- The code stored is the unsigned integer vector `y_u = y + (2^B-1)/2 · 1` (Algorithm 1 line 9).
- Because the grid is symmetric, the most significant bit of each `y_u` coordinate is the sign / orthant bit. The MSBs concatenated equal the 1-bit RaBitQ code (§4.2, Fig. 2, and `y_u = 2^(B-1) y_0 + y_last`).

**Per-vector search for the best codeword (§3.2.2, Eq. 9-10, Lemma 3.1, Algorithm 1).**
- Goal: `y_bar = argmax_{y in G} <y/||y||, o'>`, a maximum-cosine problem. `2^(B·D)` candidates is too many to enumerate.
- Lemma 3.1: the winner is also the nearest grid point to `t·o'` for some rescale `t > 0`. So it is enough to round `t·o'` over the critical values of `t`.
- The critical value that moves coordinate `i` from rounded value `x` to `x+1` is `(x+0.5)/o'[i]`.
- Only about `D · 2^(B-1)` critical values exist, because `y_bar` lies in the same orthant as `o'`. They are enumerated in ascending order with a min-heap.
- `<y_cur,o'>` and `||y_cur||` are updated in O(1) per step. The best `t` maximises `<y_cur,o'>/||y_cur||`.
- **What is maximised:** cosine between `y` and `o'`, which is the same as minimising `||o_bar - o||` over the unit codebook. A per-vector scale `t` is searched and then thrown away. Only the integer codes are stored.
- **Complexity:** `O(2^B · D log D)` per vector (§3.2.2). Reported indexing time on OpenAI-3072 at about 10^6 vectors, 96 threads (Table 3): B=1 43.8 s, 4 58.1 s, 5 64.6 s, 8 143.7 s, 10 418.1 s. B=9 on 113M MSMARCO vectors took about 2 hours (§5.2.4).

**Faiss implementation (`RaBitQuantizerMultiBit.cpp`).**
- Layout: 1 sign bit per dimension (`r_j > 0`, where `r = x - center`), plus `ex_bits = nb_bits - 1` extra bits per dimension.
- Per-dimension value is `±(code_ex + 0.5)`. For negative coordinates the ex code is complemented (`~code & max_code`). This is the same half-integer grid as paper Eq. 7.
- The `t` search is the paper's critical-value / priority-queue search, with deviations. These come from the RaBitQ-Library that the file says it adapts.
  - `t` is searched only in `[t_start, t_end]`. `t_end = (max_code + 10)/max|o_i|` (`kNEnum = 10`) and `t_start = t_end · kTightStart[ex_bits]`. The table is `{0, .15, .20, .52, .59, .71, .75, .77, .81}`. The paper's Algorithm 1 starts at `t = 0`.
  - Rounding is `int(t·|o_i| + 1e-5)`. This is floor with an epsilon, on the half-integer-shifted grid.
  - The maximised quantity is `numerator · 1/sqrt(denominator)`, with `numerator = sum (cur+0.5)|o_i|` and `denominator = D/4 + sum (cur² + cur)`. That is `<y,o'>/||y||` for the shifted grid.
  - Faiss search is therefore a tightened heuristic window. Whether it always matches the exact optimum is UNVERIFIED.
- `nb_bits` is accepted from 1 to 9.

---

## 2. Stored per-vector factors and bytes

**Paper.**
- §5.1 "Performance Metrics" states exactly two fp32 numbers are stored per vector: `||o_r - c||` and `1/(||y_u|| · <o, o_bar>)`.
- The text says "‖y_u‖" at this point. In §3.3 the relevant quantity is `||y_bar||`, with `y_bar = y_u - (2^B-1)/2 · 1`. I take this as a notational slip. UNVERIFIED which one the authors intended, but §3.3 and the faiss code both use the centred `y_bar`.
- Bytes per vector in the paper: `ceil(B·D/8) + 2·4` (assuming fp32). Table 2 is consistent with this.

| d=768 | B=1 | B=2 | B=3 | B=4 |
|---|---|---|---|---|
| paper bytes (B·96 + 8) | 104 | 200 | 296 | 392 |

**Faiss (probe, d=768, `IndexRaBitQ(768, METRIC_L2, nb_bits)`, same for `METRIC_INNER_PRODUCT`).**

| nb_bits | `index.code_size` | `index.sa_code_size()` | `index.rabitq.code_size` | bytes/dim | paper | extra vs paper |
|---|---|---|---|---|---|---|
| 1 | 104 | 104 | 104 | 1.083 | 104 | 0 |
| 2 | 212 | 212 | 212 | 2.208 | 200 | +12 |
| 3 | 308 | 308 | 308 | 3.208 | 296 | +12 |
| 4 | 404 | 404 | 404 | 4.208 | 392 | +12 |

Formula from `RaBitQuantizer::compute_code_size`:
- **1 bit:** `ceil(d/8) + 8`, laid out as `[sign bits][SignBitFactors: or_minus_c_l2sqr f32, dp_multiplier f32]`.
- **nb_bits > 1:** `ceil(d/8) + 12 + ceil(d·ex_bits/8) + 8`, laid out as `[sign bits][SignBitFactorsWithError: or_minus_c_l2sqr, dp_multiplier, f_error (3×f32 = 12 B)][ex-bit codes][ExtraBitsFactors: f_add_ex, f_rescale_ex (2×f32 = 8 B)]`.
- At d=768 the multi-bit total is `96 + 12 + 96·ex_bits + 8 = 116 + 96·(nb_bits-1)`.

All factors are fp32. I parsed real codes at nb_bits 1 and 3 and checked the fields:
- `or_minus_c_l2sqr == ||x - center||²`
- `dp_multiplier == ||r|| / <o_bar,o>`, where `<o_bar,o> = sum|r_i|/(||r|| sqrt(d))`. This is about 0.79 to 0.80 as the paper says (§2.2: "concentrated around 0.8").
- `f_add_ex == ||r||²`
- `f_rescale_ex == -2||r|| / ipnorm`, with `ipnorm = sum (code_ex+0.5)|o_i|`, i.e. `<y_bar, o'>`.
- `f_error` was about 2.85 to 2.99 in the samples.

The extra 12 bytes at B≥2 are the first-stage (1-bit) factors plus `f_error`, kept alongside the ex-bit factors. The paper's single pair of floats is replaced by two pairs plus an error bound. This is for the two-stage prune in §4.2, which the paper describes without that storage detail. Paper-vs-faiss byte overhead above is my derivation from the faiss layout; the paper does not discuss it.

---

## 3. Estimator, bias and error bound

**Estimator (ext. §3.3, Eq. 11-12; same form as RaBitQ Theorem 3.2).**
Estimate `<o,q>` by

`<o_bar, q> / <o_bar, o>`, where `o_bar = P y_bar / ||y_bar||`.

Expanding with the stored code (Eq. 11-12):

`<o_bar, q> = (1/||y_bar||) · ( <y_u, q'> - ((2^B-1)/2) · sum_i q'[i] )`

Hence, with the two stored scalars,

`<o,q>_est = [ <y_u,q'> - ((2^B-1)/2) · sum q'[i] ] / ( ||y_bar|| · <o_bar,o> )`.

Then `||o_r - q_r||² ≈ ||o_r-c||² + ||q_r-c||² - 2 ||o_r-c|| ||q_r-c|| · <o,q>_est` (Eq. 2).
- `1/(||y_bar|| <o_bar,o>)` and `||o_r-c||` are per-vector, precomputed at index time.
- `sum q'[i]` and `||q_r-c||` are per-query.
- The correction divides by `<o_bar, o>`, the projection of `o` onto the codeword, not by `||o_bar||`. `||o_bar|| = 1` by construction, since the codebook is on the unit sphere.
- Note `||y_bar||·<o_bar,o> = <y_bar, o'>` (the "ipnorm" in faiss).

**Unbiasedness.**
- Lemma 2.1 (restating RaBitQ Theorem 3.2, Eq. 4) says the estimator is unbiased: `E[<o_bar,q>/<o_bar,o>] = <o,q>`.
- The expectation is over the random rotation `P`.
- The paper's footnote 8 says the proof needs two things: the codebook is a set of randomly rotated unit vectors, and `o_bar` is the nearest codeword to `o`.
- The extended-RaBitQ paper does not re-prove this for B>1. It argues the original proof carries over and verifies it empirically (§5.2.5, Fig. 6): linear regression of about 10^7 (estimate, truth) pairs for B=2,3 on OpenAI-3072 has slope 1 and intercept 0.

**Error bound (Eq. 4, probability at least `1 - exp(-c0 ε0²)`).**

`| <o_bar,q>/<o_bar,o> - <o,q> | ≤ sqrt( (1 - <o_bar,o>²) / <o_bar,o>² ) · ε0 / sqrt(D-1)`

- At B=1 this is `O(1/sqrt(D))`, and the paper says asymptotically optimal.
- Theorem 3.2 (ext.): for `(1/ε²) log(1/δ) > D`, `B = Θ(log( (1/D)(1/ε²) log(1/δ) ))` suffices. Equivalently the total bits `Θ(D log(...))` match the Alon-Klartag lower bound.
- The naive "pad to B·D bits" variant only gets `O(1/sqrt(B·D))`, i.e. error linear in bits instead of exponential.
- Empirical formula (Remark, §3.4, measured in §5.2.6, Fig. 7): with probability above 99.9%, `ε < c_ε · 2^-B / sqrt(D)` with `c_ε = 5.75`. Measured on unit Gaussian vectors, 5×10^6 pairs per (B, D).

**Faiss estimator.** Matches the paper:
- 1-bit: `pre_dist = or_minus_c_l2sqr + qr_to_c_L2sqr - 2·dp_multiplier·final_dot`, with `dp_multiplier = ||r|| / <o_bar,o>` (the RaBitQ ratio estimator).
- Multi-bit: `dist = ||q-c||² + f_add_ex + f_rescale_ex · ex_ip`, with `f_rescale_ex = -2||r||/ipnorm = -2||r||/(||y_bar|| <o_bar,o>)`. This is Eq. 12 into Eq. 2 term by term (`f_add_ex = ||r||²`).
- Faiss adds a lower-bound prune not in this paper's text, `lower = est - f_error·g_error` with `g_error = ||q-c||` and `f_error = 2·||r||·1.9·sqrt((ratio²-1)/(D-1))` for L2 (1× for IP), where `ratio² = ||r||²·(D/4) / (0.5·sum|r_i|)²` (constant `kConstEpsilon = 1.9`, commented "from RaBitQ paper"). Its derivation from the paper's ε0 is UNVERIFIED.

Empirical check in faiss (my probe, d=768, 4000-vector synthetic anisotropic Gaussians, 50 queries, rotated): the mean signed distance error divided by the mean true distance was |bias| ≤ 3e-5 at every nb_bits. That is consistent with unbiasedness. It is a single-rotation, single-dataset check, not a proof.

---

## 4. Query side

**Paper, extended RaBitQ (§3.3, §4.2).** The inner product `<y_u, q'>` is an integer vector times a float vector. For B=4 or 8 the paper says existing SQ/LVQ-style kernels (faiss, LVQ) apply directly. The paper does not state a query-quantisation bit width for the extended method. UNVERIFIED whether the released code quantises `q'` at B>1.

**Paper, original RaBitQ (arXiv 2405.12497 §3.3.1, Theorem 3.3).** The rotated query `q'` is quantised by randomized uniform scalar quantization to `B_q` bits, so that `<x_b, q'>` (binary data code times quantised query) can use bitwise ops or FastScan. `B_q = Θ(log log D)` suffices to keep that error at the estimator's `O(1/sqrt(D))`. The default `B_q = 4` is the value I remember; I did not find the number in the text I grepped. UNVERIFIED in the paper. It matches the faiss default.

**Faiss.**
- `IndexRaBitQ.qb` (default 4) and `centered` (default `False`) are index-level; `RaBitQSearchParameters` overrides both per search.
- `qb = 0` disables query quantisation. The distance computer is `RaBitQDistanceComputerNotQ`, which uses the raw fp32 rotated query for the 1-bit stage.
- `qb > 0`, up to 8 (`FAISS_THROW_IF_NOT(qb <= 8)`): `compute_query_factors` quantises `q - center` to `qb` bits with `min/max` scalar quantization (`delta = (v_max - v_min)/(2^qb-1)`). Distance uses AND-popcount dot products.
- `centered = True`: the range is symmetric, `[-z·sqrt(||q-c||²/d), +...]` with `z = Z_MAX_BY_QB[qb-1]` (0.797, 1.494, 2.051, 2.509, 2.913, ...), using odd integers instead of unsigned ones. It uses an XOR dot product and `int_dot_scale`.
- **For `nb_bits ≥ 2` the multi-bit full distance always uses the fp32 query.** The faiss code comment says "Even with query quantization (qb > 0), ex-bits distance computation uses the float query". So `qb` only affects the 1-bit pre-filter stage.
- **My probe confirms it.** Index-level search distances were identical for `qb` = 0, 4, 8 at nb_bits = 2, 3, 4 (mean relative error 0.0107 / 0.0057 / 0.0032 for all three, rotated). At nb_bits = 1, `qb` does matter slightly (mean relative error 0.0218 at qb=0, 0.0222 at qb=4, 0.0218 at qb=8).
- **Meaning of qb=0 vs qb=8 at 1 bit:** qb=0 uses the exact fp32 query; qb=8 is the closest to it, with an 8-bit quantised query. qb=4 is the default and has a small accuracy cost.

---

## 5. Datasets, baselines, headline results (ext. §5)

**Datasets (Table 1).** MSong (992,272 × 420), Youtube (999,000 × 1024), OpenAI-1536 and OpenAI-3072 (999,000 each, text-embedding-3-large on DBpedia), Word2Vec (10^6 × 300), GIST (10^6 × 960), MSMARCO (113.5M × 1024, Cohere embed-english-v3). Queries are provided by the dataset or 1,000 held-out vectors. Dimensions are zero-padded to a multiple of 64 in the ANN runs.

**Baselines.** RaBitQ (pad), SQ (global min/max), LVQ (per-vector min/max), PQ and OPQ (k=8, M set by bits). Distance runs centre with the global centroid. ANN runs compare with LVQ only, both in IVF (4,096 clusters; 262,144 for MSMARCO).

**Metrics.** Average and maximum relative error of the squared distance (§5.2.1), recall and average distance ratio vs QPS for K=100 (§5.2.2). Space is counted in bits per dimension including the two fp32 factors.

**Headline results (all without re-ranking against raw vectors).**
- §5.2.1: RaBitQ-ext has lower average and maximum error than every baseline at the same bit count on all datasets. LVQ error is 1.3x to 3.1x larger for B>6, with a bigger gap below B=6. At B=1 or 2, SQ and LVQ are worse by orders of magnitude. PQ/OPQ do not beat SQ/LVQ once bits per dimension are 4 or more. RaBitQ-pad error decays only as `1/sqrt(B·D)`; the ext. method, SQ and LVQ decay exponentially in B.
- §5.2.2: 4, 5 and 7 bits per dimension give over 90%, 95% and 99% recall respectively on all tested datasets. On OpenAI datasets 3 bits already exceeds 95% recall. 5-bit gives a near-perfect average distance ratio at about 97% recall. Same-bit comparison with LVQ (4 and 8 bits) shows better QPS at equal recall, largely from 1-bit-first pruning.
- §5.2.4: MSMARCO (113.5M, B=4) gives over 95% recall in 56.94 GB vs 433.47 GB raw.
- Table 2 (GB, e.g. GIST): raw 3.58, RaBitQ-3 0.37, -4 0.48, -5 0.60, -7 0.82, -8 0.93, -9 1.04, LVQ-4 0.48, LVQ-8 0.92.
- I did not read the numeric values behind Fig. 3 and Fig. 4. Only the statements above are verified from the text.

---

## 6. Mapping to remex

remex, as described in the task: store fp32 `||x||` and a Lloyd-Max-quantised rotated unit direction `u_hat`, with a fixed Gaussian codebook, no per-vector scale search, and a renorm decode that divides by `||u_hat||` computed from the codes.

| aspect | RaBitQ (ext.) / faiss | remex |
|---|---|---|
| rotation | random orthogonal `P`; paper does not say fast or dense; faiss: external dense `RandomRotationMatrix` | rotation, kind per remex |
| centering | against a centroid (global, or per-IVF-list); faiss: mean of training data | per remex; the stored norm is `||x||`, not `||x - c||` (centering not mentioned) |
| grid | shared uniform half-integer grid `±(k+0.5)`, then normalised to the unit sphere | Lloyd-Max levels for a Gaussian, fixed |
| per-vector fitting | rescale `t` searched per vector to maximise `cos(y, o')`, `O(2^B D log D)`; `t` is discarded | none; plain nearest-level per coordinate |
| stored per-vector scalars | `||o_r-c||` and `1/<y_bar,o'>` (2×fp32). Faiss adds `f_error` and 1-bit factors (+12 B at B≥2) | fp32 `||x||` only |
| estimator | `<o_bar,q> / <o_bar,o>` (unbiased over `P`) | `<u_hat/||u_hat||, q>` after renorm |

**The correction differs.**
- RaBitQ divides the estimated inner product by `<o_bar, o>`, the inner product between the unit codeword and the true unit vector. Equivalently it multiplies by `||o||/<y_bar,o'>`. This single per-vector scalar is what makes the estimator unbiased over the rotation.
- remex renorm divides by `||u_hat||`, which only makes `u_hat` unit length, i.e. it forms `o_bar`. That fixes `||o_bar||` (already 1 in RaBitQ), but it does not divide by `<o_bar,o>`.
- Hence a renormalised `u_hat` estimate of `<o,q>` is `<o_bar,q>`, which is expected to be shrunk by about `<o_bar,o>` relative to `<o,q>` (about 0.8 at 1 bit in RaBitQ, closer to 1 as B grows; the paper gives 0.8 only for B=1, and I did not measure it for larger B).
- With a Lloyd-Max codebook the centroid condition gives `E[<u,q> | u_hat] ≈ <u_hat, q>` (for the un-renormalised `u_hat`), so the un-renormalised decode is conditionally close to unbiased in the centroid sense, and renorm then rescales that by `1/||u_hat||`. This part is my reasoning, not from either paper. UNVERIFIED without an experiment.
- Distinct error geometry: RaBitQ's error bound has the factor `sqrt((1-<o_bar,o>²)/<o_bar,o>²)`, so the correction and the bound are tied. remex has no bound.

**Does faiss match the paper? Yes on the points checked.**
- Estimator: `dp_multiplier = ||r||/<o_bar,o>` at 1 bit, and `f_rescale_ex = -2||r||/<y_bar,o'>` at B≥2, are the paper's Eq. 12 + Eq. 2. I checked the stored values numerically.
- Grid: `±(code+0.5)` equals paper Eq. 7.
- Differences: (a) rotation is external and dense; (b) the `t` search is restricted to a tightened window with a heuristic `kTightStart` table; (c) 12 extra bytes at B≥2 for the two-stage prune; (d) at B≥2 the full distance uses the fp32 query regardless of `qb`; (e) the extra bits are stored as separate bit-planes (sign bits, then packed ex bits), not as a single interleaved B-bit integer.

**What to test when comparing with remex (suggestion, not a result).**
- Match bytes carefully. The paper's number is `B·D/8 + 8`; faiss is `+12` more at B≥2. remex at `B·D/8 + 4` has 4 fewer bytes than the paper's RaBitQ, which is 0.005 bytes/dim at d=768.
- Compare remex against faiss with a rotation (`RR768,RaBitQ`) and against faiss without, to separate the rotation effect from the codebook effect.
- Compare remex with and without renorm, and with an added `1/<u_hat,u>` scalar (the RaBitQ correction), to isolate the bias term.

## UNVERIFIED list
- Appendix A/B proofs (Lemma 3.1, Theorem 3.2).
- Whether the authors' released code uses a fast rotation.
- Default `B_q` in the original paper; query-quantisation width used by the extended method at B>1.
- `||y_u||` vs `||y_bar||` in the §5.1 factor description.
- Whether the faiss windowed `t` search always finds the paper's optimum.
- Derivation of `f_error` and the constant 1.9.
- IVF, FastScan and HNSW faiss variants (not read).
- Numeric values behind Fig. 3 / Fig. 4 / Table 3 other than those quoted.
- The Lloyd-Max shrinkage reasoning in section 6 (mine, not tested).
