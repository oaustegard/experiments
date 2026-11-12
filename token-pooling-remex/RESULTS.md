# token-pooling-remex — token pooling vs fewer bits at the same index size, on a pooling-aware ColBERT and its baseline

**Started:** 2026-11-11 · **Status:** done (SciFact, NFCorpus) · **Runtime:** encode ~22 min per model per corpus, bench 9–24 min per model per corpus, 4 vCPU

## Question

A late-interaction (ColBERT-style) index stores one vector per document token, so it is large. Two
ways to shrink it: **pool** (merge similar tokens, storing fewer vectors) or **quantize** (store fewer
bits per vector). Hugging Face's [token-pooling post](https://huggingface.co/blog/token-pooling)
(2026-11-11) trains a model with a loss that also scores pooled documents, and reports that pooling to
a quarter of the tokens then scores 51.23 nDCG@10 on NanoBEIR (13 datasets) against 49.03 for the
same recipe without that loss, a 2.2-point gap (97.8% vs 94.6% of each model's unpooled score).

This repo's `METHODS.md` said: "For a text token index, quantize and skip pooling." That came from two
earlier experiments on models never trained for pooling: `neomme-remex-quant` (NeoMME, 128-d tokens)
and `mxbai-edge-remex-quant` (mxbai-edge-colbert-v0-32m, 64-d tokens). Two questions:

1. At a fixed number of bytes per document, is it better to keep every token at 1 bit, or pool by 2 and
   use 2 bits, or pool by 4 and use 4 bits?
2. Does the pooling-aware checkpoint lose less to pooling than an identically trained baseline?

## Setup

- **Models:** `tomaarsen/mve-ModernBERT-base-token-pooling` ("aware") and
  `tomaarsen/mve-ModernBERT-base-token-pooling-baseline` ("baseline"), the post's own pair. Both are
  ModernBERT-base (~150M parameters) with a 128-d token projection, trained for one epoch on the same
  120k MS MARCO triplets with no tuning; only the loss differs (aware adds a pooled-document term at
  pool factor 4). Loaded with sentence-transformers 6.3.0 `MultiVectorEncoder` at the checkpoints' own
  settings, document length capped at 512 tokens. Every token vector is unit-norm.
- **Corpora:** BEIR SciFact (5,183 docs, 300 test queries, 299 tokens/doc) and NFCorpus (3,633 docs,
  323 queries, graded qrels, 326 tokens/doc).
- **Pooling** (`HierarchicalTokenPooling`, the post's implementation): Ward hierarchical clustering of
  each document's tokens on cosine distance, each cluster replaced by its mean and re-normalised to unit
  length; the first token ([CLS]) is never merged; queries are never pooled. Arms: `none`;
  `pf2`/`pf3`/`pf4` = pool factor, clustering each document down to about 1/2, 1/3 or 1/4 of its
  tokens; `mt32`/`mt16` = clustering each document down to at most 32 or 16 vectors. The module's third
  budget, an adaptive distance threshold, was not run.
- **Quantization:** remex 2.1.0, this account's vector codec: a random orthogonal rotation, then a
  Lloyd-Max scalar quantizer at 1–8 bits per coordinate. Every arm uses its **centered** mode
  (`Quantizer(128, bits, mean=corpus_mean(T))`), which subtracts the mean of the tokens being coded
  before rotating, so the levels are spent on what differs between tokens rather than on their shared
  direction. Default rotation (remex's randomized Hadamard) for every arm; the arms in the same-byte
  comparisons also run with a Haar-random rotation at seed 1 (rows marked `haar s1`) to show how much
  the codec's randomness moves the score. fp32 is the uncompressed reference.
- **Bytes per document** = tokens/doc × 128 × bits / 8. remex can store a norm per vector; every vector
  here is unit-norm, so none is stored. The single 512-byte mean per index is left out.
- **Scoring:** sum-of-MaxSim over every document (no candidate stage), nDCG@10 against qrels.
  Per-query Δ with a 5,000-resample paired bootstrap, 95% CI. Wherever an arm has a `haar s1` twin,
  the Findings and tables 2–3 use the per-query mean of its two runs; table 1 lists each run.

## Findings

1. **At about 5 KB per document, pooling by 4 and coding at 4 bits beats keeping every token at 1 bit,
   on both models and both corpora.** Against unpooled 1-bit:
   - aware: +0.044 [+0.026, +0.063] on SciFact, +0.022 [+0.011, +0.032] on NFCorpus;
   - baseline: +0.034 [+0.016, +0.052] on SciFact, +0.012 [+0.004, +0.022] on NFCorpus.

   Pooling by 2 at 2 bits ties pooling by 4 at 4 bits (every CI spans zero). It beats unpooled 1-bit in
   three of the four cells; the fourth is baseline on NFCorpus, +0.007 [−0.002, +0.015]. The 1-bit codec
   is what loses: going from 2 bits to 1 bit on unpooled tokens costs −0.045 / −0.024 (aware, SciFact /
   NFCorpus) and −0.040 / −0.019 (baseline). Pool factor 4 at 4 bits costs −0.011 / −0.008 (aware) and
   −0.016 / −0.011 (baseline) against unpooled fp32.

2. **At about 10 KB per document the trade is a wash.** Pool factor 2 at 4 bits against unpooled 2-bit:
   +0.010 [+0.001, +0.019] for aware on SciFact, and within ±0.005 with CIs spanning zero in the other
   three cells. Unpooled 2-bit already costs only −0.004 to −0.010 against fp32, so pooling has little
   to buy back.

3. **Pooling-aware training shrinks what pooling loses by about 0.01 on SciFact and by almost nothing on
   NFCorpus; one of twelve comparisons clears zero.** Pooled minus unpooled, aware minus baseline
   (table 3): SciFact +0.003 to +0.015, the one significant cell being pool factor 2 at 2 bits, +0.015
   [+0.003, +0.027]; NFCorpus −0.000 to +0.009. At pool factor 4, fp32, the comparison the post makes,
   it is +0.011 [−0.006, +0.028] on SciFact and −0.000 [−0.007, +0.006] on NFCorpus, against the post's
   +0.022 on NanoBEIR. The direction matches on SciFact; a corpus of about 300 queries cannot resolve
   an effect of this size.

4. **Unpooled, the two checkpoints tie** (+0.002 [−0.019, +0.022] on SciFact, −0.007 [−0.017, +0.003]
   on NFCorpus), so the extra loss term has no measurable cost when no pooling is applied. Both are
   one-epoch recipe demonstrations rather than tuned retrievers: unpooled SciFact is 0.658 here.

5. **Capping each document at 16 vectors is expensive on both models**: −0.056 / −0.037 for aware and
   −0.069 / −0.046 for baseline at fp32. A cap of 32 costs −0.013 / −0.018 (aware). At 4 bits, `mt32`
   scores 0.644 on SciFact at 2 KB per document, within 0.014 of fp32 at 1/75 of the bytes.

6. **The norm of the corpus mean is 0.54–0.57** for every pooling arm and both models. That number is
   remex's predictor for how much centering matters (1.0 = all tokens point the same way). It is
   moderate; mxbai-edge's tokens, where centering decided the result, sit at 0.94. Every arm here is
   centered and uncentered remex was not run, so this experiment does not measure what centering
   contributed.

## What this changes

Whether to pool a text token index depends on what 1 bit costs. Pooling by 4 costs about 0.01 here.
When the 2-bit-to-1-bit drop on unpooled tokens is larger than that, pool and keep more bits; when it is
near zero, quantize and skip pooling. This experiment's drop is −0.019 to −0.045. In
`neomme-remex-quant` the drop was about zero (2-bit and 1-bit both −0.013 on SciFact), which is why
quantizing won there. In `mxbai-edge-remex-quant` the centered drop was −0.044 / −0.020, as large as
here, but the pooled arm was coded *uncentered* (pool2 + uncentered 2-bit −0.108 against centered 1-bit
−0.058, at 2.7 vs 2.1 KB per document), so that comparison never gave pooling a fair codec.

A practical default for a 128-d text token index at 2–5 KB per document: Ward pool factor 4 or a
32-vector cap, centered 4-bit, rather than unpooled 1-bit. Measure the 2-to-1-bit drop first.

Not tested: the adaptive distance-threshold budget, uncentered codes, other corpora, and MaxSim
latency, which falls with token count and is the other reason to pool.

## Results

<!-- tables:start -->
### scifact: every arm (nDCG@10; Δ = arm − the same model's unpooled fp32; aware − baseline = paired per-query difference of absolute nDCG@10; all CIs 95% paired bootstrap)

| arm | tok/doc | B/doc | aware | aware Δ | baseline | baseline Δ | aware − baseline |
|---|---:|---:|---:|---|---:|---|---|
| none fp32 | 298.9 | 153,050 | 0.6577 | ref | 0.6560 | ref | +0.0017 [-0.0189, +0.0216] |
| none 4-bit | 298.9 | 19,131 | 0.6618 | +0.0040 [-0.0026, +0.0106] | 0.6581 | +0.0021 [-0.0037, +0.0081] | +0.0037 [-0.0147, +0.0219] |
| none 2-bit | 298.9 | 9,566 | 0.6493 | -0.0084 [-0.0188, +0.0020] | 0.6428 | -0.0132 [-0.0266, -0.0008] | +0.0065 [-0.0123, +0.0255] |
| none 2-bit haar s1 | 298.9 | 9,566 | 0.6461 | -0.0116 [-0.0227, -0.0003] | 0.6505 | -0.0055 [-0.0166, +0.0051] | -0.0044 [-0.0252, +0.0157] |
| none 1-bit | 298.9 | 4,783 | 0.6003 | -0.0574 [-0.0787, -0.0360] | 0.6103 | -0.0457 [-0.0696, -0.0223] | -0.0099 [-0.0381, +0.0190] |
| none 1-bit haar s1 | 298.9 | 4,783 | 0.6056 | -0.0522 [-0.0760, -0.0277] | 0.6028 | -0.0532 [-0.0741, -0.0337] | +0.0028 [-0.0266, +0.0329] |
| pf2 fp32 | 149.7 | 76,644 | 0.6591 | +0.0013 [-0.0086, +0.0115] | 0.6475 | -0.0085 [-0.0200, +0.0023] | +0.0116 [-0.0062, +0.0298] |
| pf2 4-bit | 149.7 | 9,581 | 0.6571 | -0.0006 [-0.0096, +0.0091] | 0.6462 | -0.0098 [-0.0225, +0.0021] | +0.0109 [-0.0079, +0.0298] |
| pf2 4-bit haar s1 | 149.7 | 9,581 | 0.6580 | +0.0003 [-0.0101, +0.0109] | 0.6463 | -0.0097 [-0.0214, +0.0013] | +0.0117 [-0.0071, +0.0304] |
| pf2 2-bit | 149.7 | 4,790 | 0.6452 | -0.0126 [-0.0247, -0.0004] | 0.6299 | -0.0261 [-0.0417, -0.0115] | +0.0152 [-0.0058, +0.0361] |
| pf2 2-bit haar s1 | 149.7 | 4,790 | 0.6512 | -0.0065 [-0.0214, +0.0083] | 0.6342 | -0.0218 [-0.0370, -0.0070] | +0.0170 [-0.0040, +0.0385] |
| pf2 1-bit | 149.7 | 2,395 | 0.5895 | -0.0683 [-0.0904, -0.0455] | 0.5933 | -0.0627 [-0.0860, -0.0392] | -0.0038 [-0.0317, +0.0240] |
| pf3 fp32 | 100.0 | 51,189 | 0.6500 | -0.0078 [-0.0184, +0.0029] | 0.6471 | -0.0089 [-0.0208, +0.0025] | +0.0028 [-0.0178, +0.0237] |
| pf3 4-bit | 100.0 | 6,399 | 0.6493 | -0.0084 [-0.0205, +0.0034] | 0.6454 | -0.0106 [-0.0227, +0.0010] | +0.0039 [-0.0173, +0.0255] |
| pf3 2-bit | 100.0 | 3,199 | 0.6384 | -0.0193 [-0.0325, -0.0067] | 0.6428 | -0.0132 [-0.0288, +0.0017] | -0.0044 [-0.0247, +0.0158] |
| pf3 1-bit | 100.0 | 1,600 | 0.5941 | -0.0636 [-0.0881, -0.0392] | 0.6053 | -0.0507 [-0.0742, -0.0272] | -0.0112 [-0.0394, +0.0175] |
| pf4 fp32 | 75.1 | 38,440 | 0.6520 | -0.0057 [-0.0173, +0.0062] | 0.6398 | -0.0162 [-0.0308, -0.0023] | +0.0122 [-0.0085, +0.0337] |
| pf4 4-bit | 75.1 | 4,805 | 0.6502 | -0.0076 [-0.0199, +0.0050] | 0.6404 | -0.0156 [-0.0314, -0.0008] | +0.0097 [-0.0106, +0.0307] |
| pf4 4-bit haar s1 | 75.1 | 4,805 | 0.6439 | -0.0138 [-0.0266, -0.0010] | 0.6403 | -0.0157 [-0.0311, -0.0016] | +0.0037 [-0.0180, +0.0259] |
| pf4 2-bit | 75.1 | 2,403 | 0.6317 | -0.0260 [-0.0405, -0.0112] | 0.6322 | -0.0238 [-0.0395, -0.0089] | -0.0005 [-0.0215, +0.0208] |
| pf4 1-bit | 75.1 | 1,201 | 0.5768 | -0.0809 [-0.1065, -0.0563] | 0.6041 | -0.0519 [-0.0774, -0.0256] | -0.0273 [-0.0600, +0.0040] |
| mt32 fp32 | 32.0 | 16,384 | 0.6449 | -0.0129 [-0.0293, +0.0043] | 0.6351 | -0.0209 [-0.0439, +0.0015] | +0.0097 [-0.0122, +0.0301] |
| mt32 4-bit | 32.0 | 2,048 | 0.6438 | -0.0139 [-0.0307, +0.0034] | 0.6312 | -0.0248 [-0.0477, -0.0023] | +0.0127 [-0.0105, +0.0339] |
| mt32 2-bit | 32.0 | 1,024 | 0.6289 | -0.0288 [-0.0461, -0.0111] | 0.6086 | -0.0474 [-0.0719, -0.0222] | +0.0203 [-0.0039, +0.0439] |
| mt32 1-bit | 32.0 | 512 | 0.5612 | -0.0966 [-0.1234, -0.0698] | 0.5622 | -0.0938 [-0.1235, -0.0639] | -0.0010 [-0.0337, +0.0307] |
| mt16 fp32 | 16.0 | 8,192 | 0.6013 | -0.0564 [-0.0772, -0.0350] | 0.5868 | -0.0692 [-0.0963, -0.0437] | +0.0145 [-0.0097, +0.0384] |
| mt16 4-bit | 16.0 | 1,024 | 0.6020 | -0.0557 [-0.0768, -0.0343] | 0.5882 | -0.0678 [-0.0951, -0.0419] | +0.0138 [-0.0117, +0.0395] |
| mt16 2-bit | 16.0 | 512 | 0.5824 | -0.0753 [-0.0990, -0.0519] | 0.5642 | -0.0918 [-0.1207, -0.0631] | +0.0182 [-0.0104, +0.0463] |
| mt16 1-bit | 16.0 | 256 | 0.5198 | -0.1379 [-0.1681, -0.1068] | 0.5158 | -0.1402 [-0.1753, -0.1065] | +0.0040 [-0.0286, +0.0367] |

### scifact: same bytes, split differently between token count and bits (ΔnDCG@10 a − b, 95% CI; an arm with a `haar s1` twin is the per-query mean of its default-rotation and haar-seed-1 runs)

| a | b | B/doc a / b | aware Δ | baseline Δ |
|---|---|---|---|---|
| pf2 2-bit | none 1-bit | 4,790 / 4,783 | +0.0452 [+0.0266, +0.0635] | +0.0255 [+0.0075, +0.0447] |
| pf4 4-bit | none 1-bit | 4,805 / 4,783 | +0.0441 [+0.0260, +0.0628] | +0.0338 [+0.0155, +0.0524] |
| pf4 4-bit | pf2 2-bit | 4,805 / 4,790 | -0.0011 [-0.0111, +0.0087] | +0.0083 [-0.0060, +0.0232] |
| pf2 4-bit | none 2-bit | 9,581 / 9,566 | +0.0098 [+0.0009, +0.0189] | -0.0004 [-0.0123, +0.0115] |

### scifact: does pooling-aware training shrink what pooling loses? (ΔnDCG@10 pooled − unpooled at the same codec, negative = pooling lost; same rotation averaging as above, so the 2/4-bit rows differ from the single-run Δs in the first table)

| pooled arm | unpooled ref | aware pooled − unpooled | baseline pooled − unpooled | aware − baseline |
|---|---|---|---|---|
| pf2 fp32 | none fp32 | +0.0013 [-0.0086, +0.0115] | -0.0085 [-0.0200, +0.0023] | +0.0098 [-0.0035, +0.0234] |
| pf4 fp32 | none fp32 | -0.0057 [-0.0173, +0.0062] | -0.0162 [-0.0308, -0.0023] | +0.0105 [-0.0065, +0.0283] |
| mt32 fp32 | none fp32 | -0.0129 [-0.0293, +0.0043] | -0.0209 [-0.0439, +0.0015] | +0.0080 [-0.0127, +0.0290] |
| mt16 fp32 | none fp32 | -0.0564 [-0.0772, -0.0350] | -0.0692 [-0.0963, -0.0437] | +0.0128 [-0.0134, +0.0408] |
| pf2 2-bit | none 2-bit | +0.0005 [-0.0094, +0.0108] | -0.0146 [-0.0251, -0.0043] | +0.0150 [+0.0027, +0.0274] |
| pf4 4-bit | none 4-bit | -0.0147 [-0.0267, -0.0024] | -0.0177 [-0.0318, -0.0046] | +0.0030 [-0.0131, +0.0204] |

scifact: 5,183 docs, 1,549,337 tokens (same tokenizer and length cap for both models), 300 queries. Norm of the corpus mean, aware / baseline, by pooling (remex's centering predictor): none 0.539 / 0.546; pf2 0.554 / 0.552; pf3 0.553 / 0.546; pf4 0.553 / 0.545; mt32 0.563 / 0.559; mt16 0.565 / 0.564. Bench time aware 1439 s, baseline 1432 s on 4 vCPU.

### nfcorpus: every arm (nDCG@10; Δ = arm − the same model's unpooled fp32; aware − baseline = paired per-query difference of absolute nDCG@10; all CIs 95% paired bootstrap)

| arm | tok/doc | B/doc | aware | aware Δ | baseline | baseline Δ | aware − baseline |
|---|---:|---:|---:|---|---:|---|---|
| none fp32 | 325.7 | 166,765 | 0.3011 | ref | 0.3079 | ref | -0.0068 [-0.0165, +0.0026] |
| none 4-bit | 325.7 | 20,846 | 0.3037 | +0.0026 [-0.0003, +0.0054] | 0.3077 | -0.0003 [-0.0035, +0.0030] | -0.0039 [-0.0138, +0.0058] |
| none 2-bit | 325.7 | 10,423 | 0.2965 | -0.0046 [-0.0108, +0.0018] | 0.3049 | -0.0031 [-0.0097, +0.0037] | -0.0083 [-0.0197, +0.0026] |
| none 2-bit haar s1 | 325.7 | 10,423 | 0.2949 | -0.0062 [-0.0117, -0.0008] | 0.3040 | -0.0040 [-0.0105, +0.0028] | -0.0090 [-0.0195, +0.0011] |
| none 1-bit | 325.7 | 5,211 | 0.2686 | -0.0326 [-0.0455, -0.0199] | 0.2823 | -0.0256 [-0.0366, -0.0146] | -0.0137 [-0.0285, +0.0011] |
| none 1-bit haar s1 | 325.7 | 5,211 | 0.2749 | -0.0262 [-0.0365, -0.0156] | 0.2877 | -0.0202 [-0.0309, -0.0094] | -0.0128 [-0.0245, -0.0009] |
| pf2 fp32 | 163.1 | 83,503 | 0.2968 | -0.0043 [-0.0103, +0.0008] | 0.3020 | -0.0060 [-0.0125, -0.0002] | -0.0052 [-0.0155, +0.0047] |
| pf2 4-bit | 163.1 | 10,438 | 0.2961 | -0.0051 [-0.0110, +0.0001] | 0.2994 | -0.0086 [-0.0156, -0.0024] | -0.0033 [-0.0141, +0.0072] |
| pf2 4-bit haar s1 | 163.1 | 10,438 | 0.2935 | -0.0077 [-0.0141, -0.0018] | 0.3002 | -0.0078 [-0.0150, -0.0014] | -0.0067 [-0.0179, +0.0043] |
| pf2 2-bit | 163.1 | 5,219 | 0.2891 | -0.0120 [-0.0207, -0.0038] | 0.2901 | -0.0178 [-0.0263, -0.0100] | -0.0010 [-0.0148, +0.0117] |
| pf2 2-bit haar s1 | 163.1 | 5,219 | 0.2903 | -0.0109 [-0.0183, -0.0040] | 0.2932 | -0.0147 [-0.0242, -0.0056] | -0.0029 [-0.0150, +0.0088] |
| pf2 1-bit | 163.1 | 2,609 | 0.2596 | -0.0415 [-0.0556, -0.0275] | 0.2695 | -0.0385 [-0.0506, -0.0261] | -0.0098 [-0.0271, +0.0064] |
| pf3 fp32 | 108.9 | 55,759 | 0.2955 | -0.0056 [-0.0121, +0.0005] | 0.3001 | -0.0079 [-0.0157, -0.0009] | -0.0046 [-0.0155, +0.0063] |
| pf3 4-bit | 108.9 | 6,970 | 0.2948 | -0.0064 [-0.0132, -0.0000] | 0.2990 | -0.0090 [-0.0165, -0.0020] | -0.0042 [-0.0152, +0.0065] |
| pf3 2-bit | 108.9 | 3,485 | 0.2821 | -0.0191 [-0.0282, -0.0103] | 0.2919 | -0.0160 [-0.0251, -0.0078] | -0.0098 [-0.0227, +0.0025] |
| pf3 1-bit | 108.9 | 1,742 | 0.2535 | -0.0476 [-0.0617, -0.0331] | 0.2686 | -0.0393 [-0.0511, -0.0280] | -0.0151 [-0.0307, +0.0005] |
| pf4 fp32 | 81.8 | 41,871 | 0.2915 | -0.0096 [-0.0178, -0.0024] | 0.2985 | -0.0095 [-0.0175, -0.0023] | -0.0070 [-0.0168, +0.0025] |
| pf4 4-bit | 81.8 | 5,234 | 0.2920 | -0.0091 [-0.0173, -0.0018] | 0.2979 | -0.0100 [-0.0185, -0.0023] | -0.0059 [-0.0156, +0.0037] |
| pf4 4-bit haar s1 | 81.8 | 5,234 | 0.2945 | -0.0066 [-0.0140, +0.0004] | 0.2970 | -0.0109 [-0.0192, -0.0031] | -0.0025 [-0.0122, +0.0068] |
| pf4 2-bit | 81.8 | 2,617 | 0.2836 | -0.0176 [-0.0275, -0.0080] | 0.2870 | -0.0209 [-0.0299, -0.0126] | -0.0034 [-0.0155, +0.0086] |
| pf4 1-bit | 81.8 | 1,308 | 0.2550 | -0.0461 [-0.0611, -0.0317] | 0.2661 | -0.0418 [-0.0545, -0.0292] | -0.0111 [-0.0266, +0.0044] |
| mt32 fp32 | 32.0 | 16,382 | 0.2834 | -0.0178 [-0.0280, -0.0077] | 0.2865 | -0.0215 [-0.0323, -0.0110] | -0.0031 [-0.0130, +0.0073] |
| mt32 4-bit | 32.0 | 2,048 | 0.2843 | -0.0169 [-0.0275, -0.0066] | 0.2831 | -0.0248 [-0.0353, -0.0145] | +0.0012 [-0.0086, +0.0113] |
| mt32 2-bit | 32.0 | 1,024 | 0.2750 | -0.0261 [-0.0371, -0.0152] | 0.2728 | -0.0351 [-0.0461, -0.0238] | +0.0022 [-0.0105, +0.0143] |
| mt32 1-bit | 32.0 | 512 | 0.2478 | -0.0533 [-0.0673, -0.0395] | 0.2518 | -0.0562 [-0.0710, -0.0413] | -0.0039 [-0.0205, +0.0117] |
| mt16 fp32 | 16.0 | 8,192 | 0.2640 | -0.0371 [-0.0498, -0.0248] | 0.2622 | -0.0458 [-0.0610, -0.0312] | +0.0019 [-0.0105, +0.0150] |
| mt16 4-bit | 16.0 | 1,024 | 0.2647 | -0.0365 [-0.0493, -0.0239] | 0.2625 | -0.0454 [-0.0603, -0.0311] | +0.0021 [-0.0099, +0.0153] |
| mt16 2-bit | 16.0 | 512 | 0.2561 | -0.0451 [-0.0584, -0.0322] | 0.2508 | -0.0571 [-0.0727, -0.0416] | +0.0052 [-0.0072, +0.0172] |
| mt16 1-bit | 16.0 | 256 | 0.2321 | -0.0691 [-0.0868, -0.0516] | 0.2214 | -0.0865 [-0.1050, -0.0684] | +0.0107 [-0.0058, +0.0277] |

### nfcorpus: same bytes, split differently between token count and bits (ΔnDCG@10 a − b, 95% CI; an arm with a `haar s1` twin is the per-query mean of its default-rotation and haar-seed-1 runs)

| a | b | B/doc a / b | aware Δ | baseline Δ |
|---|---|---|---|---|
| pf2 2-bit | none 1-bit | 5,219 / 5,211 | +0.0179 [+0.0092, +0.0267] | +0.0066 [-0.0017, +0.0152] |
| pf4 4-bit | none 1-bit | 5,234 / 5,211 | +0.0215 [+0.0113, +0.0319] | +0.0124 [+0.0038, +0.0218] |
| pf4 4-bit | pf2 2-bit | 5,234 / 5,219 | +0.0036 [-0.0034, +0.0106] | +0.0058 [-0.0002, +0.0120] |
| pf2 4-bit | none 2-bit | 10,438 / 10,423 | -0.0010 [-0.0076, +0.0050] | -0.0047 [-0.0119, +0.0021] |

### nfcorpus: does pooling-aware training shrink what pooling loses? (ΔnDCG@10 pooled − unpooled at the same codec, negative = pooling lost; same rotation averaging as above, so the 2/4-bit rows differ from the single-run Δs in the first table)

| pooled arm | unpooled ref | aware pooled − unpooled | baseline pooled − unpooled | aware − baseline |
|---|---|---|---|---|
| pf2 fp32 | none fp32 | -0.0043 [-0.0103, +0.0008] | -0.0060 [-0.0125, -0.0002] | +0.0016 [-0.0042, +0.0076] |
| pf4 fp32 | none fp32 | -0.0096 [-0.0178, -0.0024] | -0.0095 [-0.0175, -0.0023] | -0.0002 [-0.0070, +0.0063] |
| mt32 fp32 | none fp32 | -0.0178 [-0.0280, -0.0077] | -0.0215 [-0.0323, -0.0110] | +0.0037 [-0.0066, +0.0136] |
| mt16 fp32 | none fp32 | -0.0371 [-0.0498, -0.0248] | -0.0458 [-0.0610, -0.0312] | +0.0087 [-0.0061, +0.0254] |
| pf2 2-bit | none 2-bit | -0.0060 [-0.0121, -0.0008] | -0.0128 [-0.0198, -0.0063] | +0.0067 [-0.0005, +0.0143] |
| pf4 4-bit | none 4-bit | -0.0105 [-0.0180, -0.0035] | -0.0102 [-0.0180, -0.0030] | -0.0003 [-0.0075, +0.0068] |

nfcorpus: 3,633 docs, 1,183,313 tokens (same tokenizer and length cap for both models), 323 queries. Norm of the corpus mean, aware / baseline, by pooling (remex's centering predictor): none 0.549 / 0.543; pf2 0.563 / 0.547; pf3 0.562 / 0.54; pf4 0.562 / 0.539; mt32 0.565 / 0.55; mt16 0.566 / 0.558. Bench time aware 1141 s, baseline 523 s on 4 vCPU.

<!-- tables:end -->

## Reproduce

```
pip install "sentence-transformers>=6.3" "remex>=2.1" datasets fastcluster
python3 encode.py          # writes .work/<model>/<corpus>/ (gitignored), ~22 min per model per corpus on 4 vCPU
python3 bench.py           # results_<model>_<corpus>.json + per-query npz, ~20 min each
python3 report.py          # renders the tables above and summary.json
```
