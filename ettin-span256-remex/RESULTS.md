# RESULTS: Ettin-17m's 768-wide output folded to 256 dims with and without remex in the model

Ettin-17m is the 10 MB text embedder behind [austegard.com/bsky/embedders](https://austegard.com/bsky/embedders/)
([`ettin-native`](../ettin-native/RESULTS.md)). Its encoder is 256 wide. Its last layer is a plain matrix that widens
the pooled 256 numbers to 768, left over from distilling it toward a 768-dim teacher. So every vector it emits is a
linear function of 256 numbers.

The search page stores each of its 61,686 Bluesky posts as [remex](https://github.com/oaustegard/remex) codes: remex
rotates a vector and rounds each coordinate to a few bits. The live index uses 2 bits for each of the 768 coordinates,
which with two stored floats comes to 200 bytes per post. Three questions:

1. Can the model emit 256 numbers instead of 768 and return the same search results?
2. Are remex codes of those 256 numbers better or worse than the live index at the same size?
3. Can remex's rotation be built into the model's last layer, so that the model emits rotated coordinates?

Recall@10 below is the share of the exact float top 10 that a compressed index returns. It measures agreement with
uncompressed search, not relevance judged by a person. Hit@10 is how often a synthetic query finds the post it was
written from. "Real" queries are 1,000 MS MARCO questions on tech and AI topics. Byte counts include the 8 bytes of
stored floats.

## Answer

Yes to the first. Replacing the 256 → 768 matrix with a 256 → 256 matrix taken from its SVD returns
0.9999 of the 768-dim float top 10, <!-- @evidence[[results/span.json#fp32_folded.real.top10_recall]] -->
and a float vector takes 1,024 bytes where it took 3,072.

For the second, the folded codes are better at the same size and worse at the same bit width. Recall@10 on the real
queries, remex seed 42:

- live index, 2 bits × 768 dims (200 B): 0.857 <!-- @evidence[[results/recall.json#arms.full768_c_b2.recall_at_10_real.mean]] -->
- 2 bits × 256 folded (72 B): 0.763 <!-- @evidence[[results/recall.json#arms.span256_c_b2.recall_at_10_real.mean]] -->
- 3 bits × 256 folded (104 B): 0.868 <!-- @evidence[[results/recall.json#arms.span256_c_b3.recall_at_10_real.mean]] -->
- 4 bits × 256 folded (136 B): 0.927 <!-- @evidence[[results/recall.json#arms.span256_c_b4.recall_at_10_real.mean]] -->
- 8 bits × 256 folded (264 B): 0.993 <!-- @evidence[[results/recall.json#arms.span256_c_b8.recall_at_10_real.mean]] -->

Yes to the third. Multiplying remex's rotation into the 256 × 256 matrix makes the model emit rotated coordinates.
With that matrix stored as int8, the 4-bit index scores 0.931, <!-- @evidence[[results/recall.json#arms.integrated256_c_b4.recall_at_10_real.mean]] -->
which is within seed spread of the 0.927 above.

Hit@10 does not separate these configurations: each one at 2 bits or more is within 0.01 of float search. Queries
with relevance labels would show whether the recall differences matter.

## Findings

1. **The fold is exact.** The 768-dim outputs lie at most 2.0e-07 outside the basis read from the head weight, <!-- @evidence[[results/span.json#span.outside_basis_max_docs]] -->
   and float scores at 256 and 768 dims differ by at most 1.1e-06. <!-- @evidence[[results/span.json#fp32_folded.max_abs_score_diff]] -->
   The top-10 lists are identical for 997 of 1,000 real queries; <!-- @evidence[[results/span.json#fp32_folded.real.top10_identical_lists]] -->
   the other three swap near-tied neighbours. Hit@10 is 0.918 either way. <!-- @evidence[[results/span.json#fp32_folded.synth.hit_at_10.mean]] -->
   An ONNX file with the 256 → 256 weight in place of the 256 → 768 one, run by onnxruntime, matches the NumPy
   head to 8.9e-08. <!-- @evidence[[results/span.json#onnx_folded.max_abs_diff_to_numpy_head]] -->
   (`run/fold.py`)
2. **Folded codes win at equal or fewer bytes.** At 104 B, 3 bits × 256 scores 0.868 <!-- @evidence[[results/recall.json#arms.span256_c_b3.recall_at_10_real.mean]] -->
   against 0.721 for 1 bit × 768. <!-- @evidence[[results/recall.json#arms.full768_c_b1.recall_at_10_real.mean]] -->
   4 bits × 256 at 136 B beats the live 200 B index by 0.070 (95% interval 0.064 to 0.076). <!-- @evidence[[results/recall.json#arms.span256_c_b4.recall_at_10_real_minus_live.mean]] -->
   8 bits × 256 at 264 B scores 0.993, <!-- @evidence[[results/recall.json#arms.span256_c_b8.recall_at_10_real.mean]] -->
   above 3 bits × 768 at 296 B, which scores 0.919. <!-- @evidence[[results/recall.json#arms.full768_c_b3.recall_at_10_real.mean]] -->
   Across five remex seeds the standard deviation is 0.002, <!-- @evidence[[results/recall.json#seeds.span256_c_b4.recall_at_10_real.sd]] -->
   and the five-seed means are 0.855 for the live configuration <!-- @evidence[[results/recall.json#seeds.full768_c_b2.recall_at_10_real.mean]] -->
   and 0.928 for 4 bits × 256. <!-- @evidence[[results/recall.json#seeds.span256_c_b4.recall_at_10_real.mean]] -->
3. **At the same bit width, 256 dims lose.** Against 2 bits × 768, 2 bits × 256 changes recall by -0.094. <!-- @evidence[[results/recall.json#arms.span256_c_b2.recall_at_10_real_minus_live.mean]] -->
   It holds a third of the bits.
4. **The rotation is still needed after the fold.** At 4 bits, quantizing the 256 SVD coordinates with no rotation
   scores 0.892 <!-- @evidence[[results/recall.json#arms.span256_norot_c_b4.recall_at_10_real.mean]] -->
   against 0.927 with it. <!-- @evidence[[results/recall.json#arms.span256_c_b4.recall_at_10_real.mean]] -->
5. **A head with the rotation multiplied in produces remex's own codes.** With an fp32 head,
   99.9997% of the 4-bit codes equal the ones remex produces by rotating the rows itself. <!-- @evidence[[results/parity.json#fp32_head.codes_equal_share]] -->
   With the head rounded to int8, 97.9% are equal, <!-- @evidence[[results/parity.json#int8_head.codes_equal_share]] -->
   and every code that differs is one level away. (`run/parity.py`)
6. **Dropping the head is not the fold.** The pooled 256 numbers, used directly, agree with the model on
   0.602 of the top 10. <!-- @evidence[[results/span.json#control_pooled_no_head.real.top10_recall]] -->
   The head stretches some directions 10.9 times more than others, <!-- @evidence[[results/span.json#head.condition]] -->
   and the fold keeps that stretch.
7. **Truncation is not the fold either.** The first 256 of the 768 coordinates agree with the model on
   0.914 of the float top 10, <!-- @evidence[[results/span.json#control_prefix256.real.top10_recall]] -->
   and their 8-bit codes stay there, at 0.914. <!-- @evidence[[results/recall.json#arms.prefix256_c_b8.recall_at_10_real.mean]] -->
8. **A 256-wide int8 head needs codes in ±127 to be as faithful as the shipped head.** All three figures are top-10
   agreement with the fp32 checkpoint's head. The shipped 768-wide head, int8 in ±63, scores 0.989. <!-- @evidence[[results/span.json#vs_fp32_checkpoint_head.shipped_int8_768_L63.real.top10_recall]] -->
   The 256-wide head with the rotation multiplied in scores 0.985 at ±63 <!-- @evidence[[results/span.json#vs_fp32_checkpoint_head.folded_int8_256_L63_remex_rotated.real.top10_recall]] -->
   and 0.992 at ±127. <!-- @evidence[[results/span.json#vs_fp32_checkpoint_head.folded_int8_256_L127_remex_rotated.real.top10_recall]] -->
   The head shrinks from 199,680 bytes <!-- @evidence[[results/span.json#head_bytes.shipped_int8_768]] -->
   to 66,560. <!-- @evidence[[results/span.json#head_bytes.folded_int8_256]] -->
9. **Centring on the corpus mean does not change recall at 2 bits or more.** 4 bits × 256 scores 0.927 centred <!-- @evidence[[results/recall.json#arms.span256_c_b4.recall_at_10_real.mean]] -->
   and 0.928 uncentred. <!-- @evidence[[results/recall.json#arms.span256_u_b4.recall_at_10_real.mean]] -->
10. **The vectors occupy 255 dimensions, one fewer than the 256 the fold keeps.** The final LayerNorm has no bias, so
    it removes each token's mean, and mean pooling keeps that constraint: the 256th singular value of the pooled
    states is 1.0e-08 of the first. <!-- @evidence[[results/span.json#span.pooled_sv_ratio_256]] -->
11. **onnxruntime at its default optimization level, `ORT_ENABLE_ALL`, changes this model's output.** Its vectors
    have a median cosine of 0.9997 to the graph as written. <!-- @evidence[[results/embed.json#sets.docs.cos_to_ORT_ENABLE_ALL.median]] -->
    At `ORT_ENABLE_BASIC`, the model's output equals the pooled state times the dequantized weight to 2.1e-07. <!-- @evidence[[results/embed.json#sets.docs.head_parity_max_abs]] -->

## Method

**Model.** `onnx-ettin17m-tuned-t4-w8L63.zip` from release `ettin-native-20261021` (sha256 checked against
`SHA256SUMS`), the package the embedders page runs. The graph ends in mean pooling and one bias-free MatMul whose
256 × 768 weight is stored as int8 codes in ±63 with one scale per output column. The caller L2-normalizes. The fp32
head comes from `st-ettin17m-tuned-e17-cap-c-epoch1.zip` in the same release.

**The fold.** The model computes e = p W, with p the pooled state. Write W = A S Bᵀ by SVD. Then z = p (A S) holds e
in the 256 coordinates of the basis B, and because B's columns are orthonormal, unit(z) and unit(e) have the same
inner products. The folded head is the 256 × 256 matrix A S. Folding remex in multiplies it by remex's rotation for
d = 256 and seed 42, so the head becomes A S Rᵀ and the output is already rotated. `run/embed.py` adds the pooled
tensor as a second graph output, so every head is applied to the same pooled states.

**Data.** The 61,686 posts of the Bluesky AI-list index, texts from the published
`bsky/embedders/data/corpus.json.gz`. Queries from the embedding store set
`bsky-ai-list/embeddinggemma-2-q4/query-targets-v1`: `real` is 1,000 MS MARCO tech and AI queries, `synth` is 1,498
Haiku-written queries, each from one held-out post. Ground truth is the exact inner-product top 10 on the shipped
model's fp32 768-dim vectors.

**Arms.** remex 2.1.0 `Quantizer(d, bits, seed=42)` at 1, 2, 3, 4 and 8 bits, centred on the corpus mean of each
space, on: the 768-dim output; the folded 256 dims (remex rotating); the first 256 of the 768 coordinates; the
folded 256 dims with no rotation; and the output of the int8 ±127 head with the rotation folded in, encoded with
`rotation="none"`. The first two also run uncentred. Codes are scored as `search_adc` scores them: the query stays
in float and is multiplied against each post's decoded direction. Every arm matches `search_adc` itself on 20 queries. Bytes per vector are
d × bits / 8 plus the stored floats: scale and bias (8 B) centred, scale (4 B) uncentred.

**Intervals.** Bootstrap over queries, 2,000 resamples, paired for differences to the live arm. Seed spread from
five remex seeds on six arms. The table and the Answer use seed 42, the seed of the live index.

**Parity.** `run/parity.py` encodes the posts three ways from the fp32 checkpoint head: remex rotating the folded
rows itself, an fp32 head with the rotation multiplied in, and the int8 ±127 rounding of that head, the last two
encoded with `rotation="none"`.

**Not run.** remex has no 6-bit width, so no folded arm sits at exactly the live index's 200 B. The WGSL engine and
the live index are unchanged. leaf-ir's head has a bias (384 → 768), which puts its vectors in a 385-dim affine set
and needs a different fold.

**Prior work.** [`eg2-span512-remex`](../eg2-span512-remex/RESULTS.md) did the 768 → 512 version on EmbeddingGemma 2
vectors after the model; this folds the basis and the rotation into the model's own last layer. Two web searches
found rotation before scalar quantization (TurboQuant and its variants) and the observation that stacked linear
layers merge, and did not find this composition on an embedder's projection head.

**Reproduce.** Unzip the two release assets under `out/rel/onnx/` and `out/rel/st/`, put `corpus.json.gz` in `out/`
and the two query files in `out/q/`, then `python3 -I run/embed.py`, `run/fold.py`, `run/eval.py`, `run/parity.py`.
About 2, 1, 5 and 1 minutes on 4 vCPUs.

## Log

### Round 1 (2026-11-09): the fold and the remex arms

Asked: whether the widening can be reversed without penalty, tried with and without integrating remex.

`run/embed.py` first failed its own check: pooled state times dequantized weight differed from the model output by
0.003. At `ORT_ENABLE_ALL`, the default, onnxruntime rewrites the `DequantizeLinear → MatMul` pairs; at
`ORT_ENABLE_BASIC` the difference is 2e-07. All vectors here use `ORT_ENABLE_BASIC`.

The first int8 folded head was rounded at ±63 from the shipped head, which is itself a rounding, and agreed with the
shipped model on 0.984 of the real top 10. Judged against the fp32 checkpoint head, a ±63 head rounded from the
checkpoint did no better (0.985), and ±127 did (0.992), so the code range was the cause. The `rotation in head` rows
use the ±127 head rounded from the checkpoint. Its 8-bit row (0.986) sits below the exact fold's (0.993) because the
ground truth is the shipped model, and this head is a different rounding of the same fp32 weights. Its 4-bit row
(0.931) is above the exact fold's (0.927) by less than the range across seeds.

| space | bits | centred | bytes | recall@10 real | recall@10 synth | hit@10 synth | recall real minus live |
|---|---|---|---|---|---|---|---|
| fp32 768 | 32 | - | 3072 | 1.000 | 1.000 | 0.918 | |
| fp32 256 folded | 32 | - | 1024 | 1.000 | 1.000 | 0.918 | |
| 768 as shipped | 1 | yes | 104 | 0.721 | 0.754 | 0.905 | -0.137 [-0.144, -0.129] |
| 768 as shipped | 2 | yes | 200 | 0.857 | 0.870 | 0.917 | live |
| 768 as shipped | 3 | yes | 296 | 0.919 | 0.927 | 0.915 | +0.062 [+0.056, +0.067] |
| 768 as shipped | 4 | yes | 392 | 0.954 | 0.958 | 0.913 | +0.097 [+0.091, +0.103] |
| 768 as shipped | 8 | yes | 776 | 0.995 | 0.995 | 0.916 | +0.138 [+0.133, +0.144] |
| 768 as shipped | 1 | no | 100 | 0.716 | 0.755 | 0.907 | -0.141 [-0.149, -0.133] |
| 768 as shipped | 2 | no | 196 | 0.852 | 0.870 | 0.914 | -0.005 [-0.011, +0.001] |
| 768 as shipped | 3 | no | 292 | 0.919 | 0.926 | 0.917 | +0.061 [+0.055, +0.067] |
| 768 as shipped | 4 | no | 388 | 0.956 | 0.959 | 0.916 | +0.099 [+0.093, +0.105] |
| 768 as shipped | 8 | no | 772 | 0.995 | 0.995 | 0.918 | +0.137 [+0.132, +0.143] |
| 256 folded | 1 | yes | 40 | 0.559 | 0.602 | 0.891 | -0.298 [-0.307, -0.288] |
| 256 folded | 2 | yes | 72 | 0.763 | 0.791 | 0.909 | -0.094 [-0.102, -0.087] |
| 256 folded | 3 | yes | 104 | 0.868 | 0.886 | 0.910 | +0.011 [+0.004, +0.018] |
| 256 folded | 4 | yes | 136 | 0.927 | 0.934 | 0.914 | +0.070 [+0.064, +0.076] |
| 256 folded | 8 | yes | 264 | 0.993 | 0.992 | 0.917 | +0.136 [+0.130, +0.141] |
| 256 folded | 1 | no | 36 | 0.555 | 0.596 | 0.885 | -0.302 [-0.313, -0.293] |
| 256 folded | 2 | no | 68 | 0.768 | 0.793 | 0.916 | -0.089 [-0.096, -0.081] |
| 256 folded | 3 | no | 100 | 0.868 | 0.883 | 0.914 | +0.011 [+0.004, +0.017] |
| 256 folded | 4 | no | 132 | 0.928 | 0.932 | 0.915 | +0.071 [+0.065, +0.077] |
| 256 folded | 8 | no | 260 | 0.992 | 0.992 | 0.917 | +0.135 [+0.129, +0.141] |
| first 256 of 768 | 1 | yes | 40 | 0.539 | 0.606 | 0.885 | -0.318 [-0.328, -0.308] |
| first 256 of 768 | 2 | yes | 72 | 0.755 | 0.780 | 0.910 | -0.102 [-0.110, -0.094] |
| first 256 of 768 | 3 | yes | 104 | 0.852 | 0.869 | 0.912 | -0.005 [-0.012, +0.002] |
| first 256 of 768 | 4 | yes | 136 | 0.888 | 0.901 | 0.909 | +0.031 [+0.024, +0.037] |
| first 256 of 768 | 8 | yes | 264 | 0.914 | 0.919 | 0.912 | +0.057 [+0.051, +0.063] |
| 256 folded, no rotation | 1 | yes | 40 | 0.525 | 0.598 | 0.884 | -0.333 [-0.342, -0.323] |
| 256 folded, no rotation | 2 | yes | 72 | 0.720 | 0.768 | 0.910 | -0.138 [-0.146, -0.130] |
| 256 folded, no rotation | 3 | yes | 104 | 0.827 | 0.859 | 0.914 | -0.030 [-0.037, -0.023] |
| 256 folded, no rotation | 4 | yes | 136 | 0.892 | 0.913 | 0.915 | +0.034 [+0.028, +0.041] |
| 256 folded, no rotation | 8 | yes | 264 | 0.981 | 0.984 | 0.917 | +0.124 [+0.118, +0.130] |
| 256 folded, rotation in head (int8) | 1 | yes | 40 | 0.539 | 0.602 | 0.888 | -0.318 [-0.327, -0.309] |
| 256 folded, rotation in head (int8) | 2 | yes | 72 | 0.759 | 0.789 | 0.913 | -0.098 [-0.105, -0.090] |
| 256 folded, rotation in head (int8) | 3 | yes | 104 | 0.868 | 0.882 | 0.918 | +0.011 [+0.004, +0.017] |
| 256 folded, rotation in head (int8) | 4 | yes | 136 | 0.931 | 0.933 | 0.915 | +0.074 [+0.068, +0.080] |
| 256 folded, rotation in head (int8) | 8 | yes | 264 | 0.986 | 0.988 | 0.918 | +0.129 [+0.123, +0.134] |

Seed spread over five remex seeds (mean, sd), recall@10 real: 1 bit × 768 0.717, 0.0021; 2 bits × 768 0.855,
0.0019; 3 bits × 768 0.918, 0.0007; 3 bits × 256 0.871, 0.0018; 4 bits × 256 0.928, 0.0019; 8 bits × 256 0.993,
0.0005. Each range across the five seeds is 0.005 or less (4 bits × 256: 0.927 to 0.932). Hit@10 sd is 0.001 to 0.004, the size of the
differences between arms.

Wall clock: embedding 99 s for the posts, the whole of `eval.py` 5 min, all on 4 vCPUs. No paid compute.
