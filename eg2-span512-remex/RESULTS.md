# RESULTS: rotate EmbeddingGemma 2 into its 512-dim span before remex quantization

The live Bluesky search page ([`embeddinggemma2-webgpu-tail/web/bsky`](../embeddinggemma2-webgpu-tail/web/bsky))
stores 61,686 posts as EmbeddingGemma 2 (q4 ONNX) vectors at 768 dims. Each vector is compressed by remex, which applies a
random rotation and then quantizes each coordinate to 2 bits, so it takes 192 bytes. EG2's text vectors only occupy a
512-dim linear subspace (METHODS.md, from `rrc-negative-margin-drop`). Writing each vector in an orthonormal basis of
that subspace gives 512 numbers with no information lost. The question is whether the index can quantize those 512
rotated coordinates instead of the 768 originals and keep the same recall. Recall@10 here is the share of the exact
fp32 768-dim top 10 that the compressed index returns. It measures agreement with the uncompressed search, not
relevance judged by a person. The Matryoshka prefix, the first 512 of the 768 coordinates, is the comparison: EG2 was
trained so that prefix works on its own.

## Answer

It does not keep the same recall at the same bit width. At the same bytes per vector it does better than the live
index. Recall@10 on 1,000 real MS MARCO tech queries:

- live index, 2 bits × 768 dims (192 B): 0.837 <!-- @evidence[[results/recall.json#arms.full768_b2.recall_at_10.real_test]] -->
- 2 bits × 512 rotated (128 B): 0.808, a change of -0.030 <!-- @evidence[[results/recall.json#arms.rot512_b2.minus_full768_b2.real_test.mean]] -->
- 3 bits × 512 rotated (192 B, the same size as now): 0.894, a change of +0.056 <!-- @evidence[[results/recall.json#arms.rot512_b3.minus_full768_b2.real_test.mean]] -->

A likely reason, not tested here: 768 two-bit codes of a 512-dim signal carry 50% more bits than 512 do. The
Matryoshka prefix does worse than the rotation at every width, with 0.804 at 3 bits. <!-- @evidence[[results/recall.json#arms.prefix512_b3.recall_at_10.real_test]] -->
Synthetic queries and five more remex seeds agree.
On NanoBEIR's relevance labels (round 3), 3-bit codes of the 512-dim vectors change NDCG@10 by -0.35 <!-- @evidence[[results/nanobeir.json#diffs.svd512_b3_minus_svd512.mean_of_sets]] -->
against unquantized search and score 0.55 above 2-bit codes at 768, <!-- @evidence[[results/nanobeir.json#diffs.svd512_b3_minus_eg2_768_b2.mean_of_sets]] -->
within remex seed noise at 650 queries.

Round 2 switched the search page to 3-bit rotated codes (`embeddinggemma2-webgpu-tail` round 7). It needed a
512 × 768 query matrix on the client (1.5 MB) and two new WGSL kernels.

## Findings

1. The q4 ONNX outputs lie in a 512-dim span, as the fp32 model's do. The graph ends in a bias-free 4-bit
   MatMulNBits projection from 512 to 768, then mean pooling and an L2 norm. All three keep vectors in the
   projection's column space. On the 61,674 live posts the 513th singular value is 3.9e-8 of the first, <!-- @evidence[[results/span.json#posts.sv_ratio_513]] -->
   and no post lies further than 1.9e-7 outside the basis taken from the dequantized projection weight. <!-- @evidence[[results/span.json#outside_proj.posts.max]] -->
   Test queries lie at most 1.8e-7 outside it, <!-- @evidence[[results/span.json#outside_proj.real_test.max]] -->
   which is within float32 rounding. (`run/span.py`)
2. The basis can be computed from the model weights alone. The left singular vectors of the dequantized 768 × 512 projection weight and the top
   512 singular vectors of 30,000 held-out MS MARCO query vectors span the same space, with a smallest
   principal-angle cosine of 0.99999999999. <!-- @evidence[[results/span.json#principal_angle_cos_min]] -->
   Exact search in the rotated 512 coordinates finds 0.999 of the 768-dim top 10. <!-- @evidence[[results/recall.json#arms.fp32_rot512.recall_at_10.real_test]] -->
   Exact search on the 512-dim prefix finds 0.827. <!-- @evidence[[results/recall.json#arms.fp32_prefix512.recall_at_10.real_test]] -->
3. At 2 bits per coordinate the rotated codes give 0.808 against 0.837 for the live index. <!-- @evidence[[results/recall.json#arms.rot512_b2.recall_at_10.real_test]] -->
   The bootstrap 95% CI of the difference is -0.037 to -0.023. <!-- @evidence[[results/recall.json#arms.rot512_b2.minus_full768_b2.real_test.ci95.0]] -->
   On synthetic queries the difference is -0.022. <!-- @evidence[[results/recall.json#arms.rot512_b2.minus_full768_b2.synth_test.mean]] -->
4. At 192 bytes per vector, 3 bits × 512 (192 B) gives 0.894 on real queries <!-- @evidence[[results/recall.json#arms.rot512_b3.recall_at_10.real_test]] -->
   and 0.907 on synthetic ones, <!-- @evidence[[results/recall.json#arms.rot512_b3.recall_at_10.synth_test]] -->
   against 0.837 and 0.862 for the live index. <!-- @evidence[[results/recall.json#arms.full768_b2.recall_at_10.synth_test]] -->
   4 bits × 512 at 256 B (0.939) <!-- @evidence[[results/recall.json#arms.rot512_b4.recall_at_10.real_test]] -->
   beats 3 bits × 768 at 288 B (0.908). <!-- @evidence[[results/recall.json#arms.full768_b3.recall_at_10.real_test]] -->
   At 1 bit (64 B) the rotated codes give 0.623, below 1 bit × 768 at 96 B (0.685). <!-- @evidence[[results/recall.json#arms.rot512_b1.recall_at_10.real_test]] -->
5. The Matryoshka prefix loses to the rotation at every width. It holds 77% of each post's energy, <!-- @evidence[[results/span.json#prefix512_energy_posts]] -->
   and its codes trail the rotated codes by 0.055 to 0.118 on real queries from 2 to 4 bits, the gap growing with
   bits because the prefix's own truncation error stays put. 4 bits on the prefix
   (0.821) <!-- @evidence[[results/recall.json#arms.prefix512_b4.recall_at_10.real_test]] -->
   is below 2 bits at 768.
6. The rebuilt fp32 vectors are the ones the live index was built from. Re-encoding them with its quantizer
   reproduces its 47.4M codes except for 16 single flips. <!-- @evidence[[results/parity.json#codes_flipped]] -->
   (`run/parity.py`)
7. The differences between arms exceed seed noise. Across five more remex seeds, recall moves by at most 0.008 per arm. <!-- @evidence[[results/seeds.json#arms.rot512_b2.real_test.range]] -->
   The seed means are 0.831 for the live layout, <!-- @evidence[[results/seeds.json#arms.full768_b2.real_test.mean]] -->
   0.804 for rotated 2-bit <!-- @evidence[[results/seeds.json#arms.rot512_b2.real_test.mean]] -->
   and 0.889 for rotated 3-bit, so the seed-averaged gaps are -0.027 and +0.058. <!-- @evidence[[results/seeds.json#arms.rot512_b3.real_test.mean]] -->
   (`run/seeds.py`)
8. On NanoBEIR (13 English sets, NDCG@10 × 100) the unquantized columns of tomaarsen/embeddinggemma-2-512d's
   card reproduce to within 0.18 of its means: 63.24 for EG2 at 768, <!-- @evidence[[results/nanobeir.json#arms.eg2_768.mean_of_sets]] -->
   62.83 for the 512 prefix <!-- @evidence[[results/nanobeir.json#arms.prefix512.mean_of_sets]] -->
   and 63.21 for the SVD 512d model. <!-- @evidence[[results/nanobeir.json#arms.svd512.mean_of_sets]] -->
   remex 3-bit codes of the 512d vectors score 62.86 at 192 B, <!-- @evidence[[results/nanobeir.json#arms.svd512_b3.mean_of_sets]] -->
   2-bit codes at 768 score 62.31 at the same size, <!-- @evidence[[results/nanobeir.json#arms.eg2_768_b2.mean_of_sets]] -->
   and 2-bit codes of the 512d vectors 62.13 at 128 B. <!-- @evidence[[results/nanobeir.json#arms.svd512_b2.mean_of_sets]] -->
   The 192 B difference has a pooled 95% CI of -0.05 to +1.17, <!-- @evidence[[results/nanobeir.json#diffs.svd512_b3_minus_eg2_768_b2.ci95_pooled.0]] -->
   and over five remex seeds it shrinks to +0.23. <!-- @evidence[[results/nanobeir.json#seeds.svd512_b3_minus_eg2_768_b2_over_seeds]] -->
   Only 2-bit at 768 is clearly below unquantized search (-0.93). <!-- @evidence[[results/nanobeir.json#diffs.eg2_768_b2_minus_eg2_768.mean_of_sets]] -->
   (round 3, `run/nanobeir_eval.py`)

## Method

**Corpus.** The 61,686 posts of `web/bsky`, with text re-fetched by `eg2-query-student/run/hydrate_posts.py`.
12 have been deleted and are dropped, leaving 61,674. They were embedded by `run/embed_docs.py` with `model_q4.onnx` on
onnxruntime's CPU provider, using the document prompt `title: none | text: ` and 128 tokens. This is the same graph
and prompt as `build_bsky_index.py`; finding 6 checks the match.

**Queries.** The two held-out test sets of the embedding-store set `bsky-ai-list/embeddinggemma-2-q4/query-targets-v1`.
`real_test` holds 1,000 MS MARCO tech/AI queries. `synth_test` holds 1,498 Haiku queries written from held-out posts.
Ground truth is the exact inner-product top 10 over the fp32 768-dim vectors.

**Basis.** `run/span.py` dequantizes `/model/embedding_projection/MatMul_Quant` (4-bit, block 32, low nibble first) and
takes its left singular vectors. The control basis is the uncentred top 512 right singular vectors of `msmarco_train`,
30,000 non-tech queries. Neither is fitted on the vectors being evaluated.

**Arms.** remex 2.1.0 `Quantizer(d, bits, seed=42, mean=corpus mean)` with `rht` rotation, at 1 to 4 bits, on:

- the 768-dim vectors (2 bits is the live index)
- the 512 rotated coordinates `x @ B.T`
- the 512-dim prefix, renormalized, as `embeddinggemma2-webgpu-tail/run/eval_512.py` did

Codes are scored as `search_adc` scores them, as one matmul over decoded directions. Every arm checks that
scorer against `search_adc` on 20 queries, and all agree at 1.0.

**Metric.** Recall@10 against the ground truth. 95% CIs come from a paired bootstrap over queries (2,000 resamples)
for each arm minus the live index. `run/seeds.py` reruns the headline arms at five more remex seeds, because the
bootstrap holds the rotation fixed. Byte counts are code bytes. Each vector also carries 8 bytes of scale and bias.

**Cost.** Hydration took about 3 minutes. The CPU embed took 45 minutes on 4 cores (2,715 s). Each eval arm took under a minute.

## Log

### 2026-10-30 — round 1 (CCotw session delegated from the `rrc-negative-margin-drop` session)

Asked: does rotating into the 512 span let the Bluesky index drop to 512 dims at no recall cost? Compare at equal
byte budgets, at 1, 2 and 3 bits.

The fp32 document vectors behind the index were never uploaded, so they were rebuilt (hydrate, then embed for 45 minutes).
Parity with the committed codes was 16 flips in 47.4M. The first parity run reported agreement per post where it
meant per code. That was fixed before any number was used.

Prediction, from `rrc-negative-margin-drop`'s fp32 result (rotation exact, prefix 0.926 on Quora): rot512 at 2 bits would
match 768 at 2 bits. Measured: it is 0.030 lower. The prediction treated quantization error as a function of bits per
dimension; total bits per vector is the better predictor here. Rotation does remove all of the truncation error, which the prefix
pays even in fp32 (0.827 here, against 0.926 on Quora).

| arm | code B | real_test | synth_test |
|---|---:|---:|---:|
| fp32 rot512 | 2048 | 0.999 | 0.998 |
| fp32 prefix512 | 2048 | 0.827 | 0.859 |
| full768 1-bit | 96 | 0.685 | 0.738 |
| full768 2-bit (live) | 192 | 0.837 | 0.862 |
| full768 3-bit | 288 | 0.908 | 0.921 |
| full768 4-bit | 384 | 0.947 | 0.954 |
| rot512 1-bit | 64 | 0.623 | 0.683 |
| rot512 2-bit | 128 | 0.808 | 0.840 |
| rot512 3-bit | 192 | 0.894 | 0.907 |
| rot512 4-bit | 256 | 0.939 | 0.949 |
| prefix512 1-bit | 64 | 0.598 | 0.663 |
| prefix512 2-bit | 128 | 0.753 | 0.792 |
| prefix512 3-bit | 192 | 0.804 | 0.836 |
| prefix512 4-bit | 256 | 0.821 | 0.852 |
| rot512 2-bit, held-out basis | 128 | 0.803 | 0.831 |

The held-out basis and the weight basis span the same space, yet they differ by 0.004 and 0.010 at 2 bits. The two
bases are related by an orthogonal map, which only changes which random rotation remex applies. That difference is
seed noise. `run/seeds.py` confirms it: each arm's range over five seeds is 0.003 to 0.008, while the
seed-averaged arm gaps are 0.027 and 0.058.

**What switching the page would need.** Documents are encoded offline, so only queries need the transform.
`q512 = B @ q768` is 393k multiply-adds, negligible next to the model, in JS or as one small kernel.
`B` is 512 × 768: 1.5 MB as float32, 0.75 MB as float16 (float16 not tested).
`web/tail.js` hard-codes `D = 768` in its WGSL (workgroup arrays, LUT stride, RHT plan), and its pack and scan read
2-bit words. The 3-bit arm would need a d = 512 RHT plan from remex, 3-bit packing (8 values in 3 bytes, crossing
byte boundaries) in the scan, and a rebuilt `codes.u8.txt` (the same 11.8 MB) and `aux.f32.txt`. The 2-bit arm
needs only the d = 512 kernels and shrinks the codes to 7.9 MB, at -0.030 recall. The live page was not changed in this round.

### 2026-10-30 — round 2: the page switches

Oskar: "switch the bsky page to 3-bit rotated codes". Done in `embeddinggemma2-webgpu-tail`
(round 7 of its log): `run/build_bsky_span512.py` writes the index from this experiment's
`out/docs.npy` and `out/basis_proj.npy`, and `SpanTail` in `web/tail.js` scans it. remex's rotation
is folded into the basis, so the query side is one 512 × 768 product and needs no RHT plan. The
codes are stored as bit planes (three u32 words per 32 dims), so a post stays 48 words and the
existing corpus buffer and GPU top-k are reused. Headless, the page's top 10 equals remex's on
8 of 8 parity queries. `?span=0` loads the old 2-bit index.

### 2026-10-30 — round 3: NanoBEIR, as on tomaarsen/embeddinggemma-2-512d's card

Tom Aarsen published [tomaarsen/embeddinggemma-2-512d](https://huggingface.co/tomaarsen/embeddinggemma-2-512d),
which replaces EG2's 512 → 768 projection with `diag(S) @ Vh` from its SVD: the same rotation into the
512-dim span as here, shipped as a model. Oskar asked to repeat his NanoBEIR evaluation with remex codes
added before replying to him, so the comparison is on his benchmark and his model.

`run/nanobeir_encode.py` ran on a Colab L4 (bf16, sentence-transformers 6.2, `encode_query` /
`encode_document`, so the models' own query and document prompts): all 13 English NanoBEIR sets, with
`google/embeddinggemma-2` and with his model. That took about 20 minutes of GPU time and about one compute unit.
Two restarts were needed: NanoTouche2020's long documents ran out of GPU memory at batch 32, and the first
retry saved a stale array from the previous set before the guard caught it. Both were fixed and the file
re-encoded. `run/nanobeir_eval.py` scores on CPU, with NDCG@10 computed as sentence-transformers'
InformationRetrievalEvaluator does. The remex arms encode documents, keep queries as floats, and score as
`search_adc` does.

| arm | bytes | NDCG@10 mean of 13 | card |
|---|---:|---:|---:|
| EG2 768, float | 3072 | 63.24 | 63.36 |
| EG2 first 512, float | 2048 | 62.83 | 62.75 |
| SVD 512d model, float | 2048 | 63.21 | 63.39 |
| SVD 512d, remex 4-bit | 256 | 62.76 | |
| SVD 512d, remex 3-bit | 192 | 62.86 | |
| SVD 512d, remex 2-bit | 128 | 62.13 | |
| EG2 768, remex 3-bit | 288 | 62.88 | |
| EG2 768, remex 2-bit | 192 | 62.31 | |
| EG2 first 512, remex 3-bit | 192 | 62.80 | |

Per set, the reproduction differs from the card by up to 0.74 (EG2) and 1.32 (SVD model, NanoMSMARCO).
Each set has 50 queries, and bf16 on a different GPU with different batch padding moves a few ranks. The
means agree to 0.18. Paired bootstrap over the 650 queries:
- SVD 3-bit minus EG2 2-bit, both 192 B: +0.55 [-0.05, +1.17].
- SVD 3-bit minus unquantized SVD: -0.35 [-0.74, +0.05].
- EG2 2-bit minus unquantized EG2: -0.93 [-1.55, -0.36].
- SVD 3-bit minus prefix 3-bit: +0.05 [-0.54, +0.60].
Across remex seeds 0 to 4 the means move by 0.85 (SVD 3-bit) and 0.53 (EG2 2-bit), the same size as the
gaps being measured.

Reading: the direction matches rounds 1 and 2. At 192 B, 3-bit codes of the 512-dim vectors do better than
2-bit codes at 768 and stay closest to unquantized search. NanoBEIR's 50 queries per set are too few to
show it at this effect size, where the 1,000-query exact-search recall of round 1 did (0.894 against 0.837).
The prefix-versus-rotation gap that round 1 measured against exact search (0.090 at 3 bits) does not show in
NDCG either: on relevance labels the prefix loses only 0.40 even unquantized, as Tom's card also shows.

