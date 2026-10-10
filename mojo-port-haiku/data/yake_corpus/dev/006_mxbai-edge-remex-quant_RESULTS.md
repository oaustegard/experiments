# mxbai-edge-remex-quant — compressing mxbai-edge-colbert-v0-32m token vectors with remex, remax and PLAID

**Started:** 2026-09-27 · **Status:** done (SciFact, NFCorpus) · **Runtime:** encode 11 + 8 min, bench 20 + 10 min, follow-ups 7 min, 4 vCPU

## Question

mxbai-edge-colbert-v0-32m (Apache-2.0, 32M, 64-d per token) ties 33M single-vector
models on BEIR while storing about 45× the bytes per document (memory `2a7b4d12`).
How far down can remex push its token vectors before retrieval degrades, how does
that compare with the compression PLAID actually ships, and where does the curve
land against a single-vector model at matched bytes?

## Setup

- **Model:** `mixedbread-ai/mxbai-edge-colbert-v0-32m` through PyLate 1.5.0 with the
  card's own settings: query length 48, document length 512, no query expansion,
  document-side punctuation skiplist on. Tokens are unit-norm, 64-d.
- **Comparator:** `BAAI/bge-small-en-v1.5` (33M, 384-d, one vector per document), the
  model the mxbai card itself ties on BEIR.
- **Corpora:** BEIR SciFact (5,183 docs, 300 test queries, 267 tokens/doc after the
  skiplist) and NFCorpus (3,633 docs, 323 queries, graded qrels, 286 tokens/doc).
- **Scoring:** brute-force MaxSim over every document; no candidate stage.
- **Codecs:**
  - remex 0.8.0 `Quantizer` (Lloyd-Max on a rotated sphere, renorm on) at 8/4/3/2/1 bits.
  - remex centered mode (`mean=corpus_mean(T)`, one 256-byte mean for the whole index).
  - remax 0.3.0 `StackedSignBitQuantizer` k=1/2 with a float query (asym) and a binarised one (sym).
  - PLAID-style residual compression as ColBERTv2 specifies it: k-means centroids
    (K = 16,384 on SciFact, K = 8,192 on NFCorpus; 2^⌊log2(16·√tokens)⌋), uint16 centroid id per token, per-dimension residual
    buckets at 4/2/1 bits with quantile cutoffs, centroid-only as the floor.
  - PyLate's hierarchical token pooling at factor 2.
  - int8 with one global scale, and fp16.
- **Bytes:** per document as stored. Unit-norm token arms leave out remex's 4-byte
  per-vector norm (every norm is 1.0). Pooled arms include it, since pooled means are
  not unit-norm. PLAID rows show bytes with the centroid table amortised in parentheses.
- **Statistics:** nDCG@10 against qrels. Per-query ΔnDCG@10 against mxbai fp32 with a
  5,000-resample paired bootstrap, 95% CI. Top-10 overlap with the fp32 ranking is in
  `results_*.json`.
- **Seeds:** rht at 64-d is seed-invariant (ERRORS.md #1), so the seed floor uses haar
  at seeds 0 and 1.

## Findings

1. **Plain remex breaks at 2 bits and below on these tokens.** 1-bit costs −0.181
   nDCG@10 on SciFact and −0.161 on NFCorpus; 2-bit costs −0.051 and −0.033. On
   NeoMME's 128-d tokens the same codec cost −0.013 at 1 bit (`neomme-remex-quant`).
   The mxbai tokens sit in a narrow cone:
   - the corpus mean has norm 0.936 (SciFact) / 0.940 (NFCorpus);
   - two random tokens have cosine 0.876 on average, 0.789 at the 5th percentile;
   - after subtracting the mean, that falls to 0.009, and residual norms average 0.32.

   Uncentered codes spend most of their levels on the shared component. They are also
   unstable: uncentered 1-bit scored 0.5055 and 0.6132 on SciFact under two haar seeds.

2. **Centering with one stored mean recovers most of it.**
   - 4-bit: −0.004 / −0.000.
   - 2-bit: −0.014 / −0.003.
   - 1-bit: −0.058 / −0.022.

   Centered 1-bit beats uncentered 1-bit by +0.123 [+0.091, +0.155] (SciFact) and
   +0.139 [+0.116, +0.162] (NFCorpus). Its haar seed spread is 0.008 (SciFact) and
   0.005 (NFCorpus), against 0.108 and 0.050 uncentered. remex's own predictor
   (‖mean‖/mean‖x‖, 0.94 here) called this before any retrieval ran.

3. **Centered remex ties PLAID at 4 and 2 bits and is smaller.**
   - 2-bit: −0.001 [−0.013, +0.011] on SciFact and +0.004 [−0.004, +0.011] on
     NFCorpus, at 4,269 vs 4,803 B/doc on SciFact. PLAID also needs a centroid table
     (2 MB on SciFact, 405 B/doc amortised over 5,183 documents; 1 MB on NFCorpus, 289 B/doc),
     which shrinks per document as the corpus grows.
   - 1-bit: PLAID is ahead on SciFact, +0.029 [+0.009, +0.049], and ties on NFCorpus
     (+0.001 [−0.011, +0.013]). At 1 bit, the per-token centroid id carries information that a single
     corpus mean does not.
   - PLAID's centroid id alone (534 B/doc on SciFact) scores 0.687 on SciFact and
     0.275 on NFCorpus.

4. **8-bit, int8 and fp16 are free.** remex 8-bit +0.001 on both corpora, int8
   −0.003 / −0.001, fp16 +0.001 / +0.000, all inside their CIs. Uncentered 4-bit is
   −0.005 / −0.001, also inside.

5. **remax 1-bit behaves like uncentered remex 1-bit.** k=1 asym scores 0.5319 on
   SciFact (−0.213), inside uncentered remex 1-bit's haar seed range of 0.5055–0.6132.
   On NFCorpus it scores 0.1906 (−0.170) against a range of 0.1856–0.2352. At 1 bit on
   unit-norm input the two are the same code up to rotation (memory `abbd68ed`).
   - k=2 asym costs −0.083 / −0.101 at 2-bit bytes, behind centered remex 2-bit.
   - k=1 sym costs −0.305 / −0.204.
   - Centering remax by sign-coding the residual collapses retrieval (ERRORS.md #2).

6. **Pooling is a poor byte trade on text.** pool2 at fp32 costs −0.040 on SciFact
   and −0.005 on NFCorpus. pool2 + remex 2-bit (2,673 B/doc) scores −0.108 on SciFact,
   worse than centered 1-bit at 2,135 B/doc (−0.058). This matches the NeoMME text
   result: on text tokens, quantize and skip pooling.

7. **Against a single vector, compressed tokens keep a small, non-significant margin
   at over 20× the bytes.**
   - At fp32, mxbai beats bge-small by +0.032 [+0.005, +0.059] (SciFact) and
     +0.017 [−0.002, +0.036] (NFCorpus).
   - bge-small at remex 4-bit (192 B/doc) is within 0.005 of its own fp32 on both.
   - The smallest mxbai index that keeps most of its margin is centered 2-bit, at
     4,269 / 4,571 B/doc, 22–24× bge-small's 192. Over bge-small remex 4-bit it is
     +0.014 [−0.014, +0.041] (SciFact) and +0.016 [−0.002, +0.034] (NFCorpus).
   - At about 2.1–2.9 KB/doc (centered 1-bit, PLAID 1-bit), mxbai is level with
     bge-small fp32 (−0.026 to +0.003, every CI spanning zero).

   On these two short-text corpora, a first-stage index should be single-vector. The
   token index belongs as a reranker over a shortlist, or on long documents where the
   card's LongEmbed numbers (0.849 at 32k vs 0.441 for answerai-colbert-small) are the
   reason to use it.

## Results

<!-- tables:start -->
### Every arm, both corpora (nDCG@10; Δ vs mxbai fp32 with 95% paired-bootstrap CI; bytes per document as stored)

| arm | SciFact B/doc | SciFact nDCG@10 | SciFact Δ | NFCorpus B/doc | NFCorpus nDCG@10 | NFCorpus Δ |
|---|---:|---:|---|---:|---:|---|
| late fp32 | 68,304 | 0.7447 | ref | 73,134 | 0.3606 | ref |
| late fp16 | 34,152 | 0.7456 | +0.0009 [-0.0002, +0.0029] | 36,567 | 0.3608 | +0.0003 [-0.0002, +0.0010] |
| late int8 | 17,076 | 0.7414 | -0.0033 [-0.0077, +0.0009] | 18,284 | 0.3600 | -0.0005 [-0.0030, +0.0018] |
| late remex 8-bit | 17,076 | 0.7453 | +0.0006 [-0.0008, +0.0029] | 18,284 | 0.3612 | +0.0006 [-0.0009, +0.0023] |
| late remex 4-bit | 8,538 | 0.7401 | -0.0046 [-0.0143, +0.0044] | 9,142 | 0.3594 | -0.0012 [-0.0057, +0.0037] |
| late remex 3-bit | 6,404 | 0.7290 | -0.0157 [-0.0274, -0.0045] | 6,856 | 0.3519 | -0.0087 [-0.0160, -0.0016] |
| late remex 2-bit | 4,269 | 0.6932 | -0.0515 [-0.0741, -0.0304] | 4,571 | 0.3278 | -0.0328 [-0.0433, -0.0223] |
| late remex 1-bit | 2,134 | 0.5636 | -0.1811 [-0.2174, -0.1456] | 2,285 | 0.1995 | -0.1611 [-0.1857, -0.1368] |
| late remex 2-bit haar s0 | 4,269 | 0.6936 | -0.0511 [-0.0738, -0.0295] | 4,571 | 0.3345 | -0.0261 [-0.0360, -0.0166] |
| late remex 2-bit haar s1 | 4,269 | 0.6832 | -0.0615 [-0.0830, -0.0412] | 4,571 | 0.3216 | -0.0390 [-0.0519, -0.0267] |
| late remex 1-bit haar s0 | 2,134 | 0.5055 | -0.2392 [-0.2797, -0.1994] | 2,285 | 0.1856 | -0.1750 [-0.2009, -0.1495] |
| late remex 1-bit haar s1 | 2,134 | 0.6132 | -0.1315 [-0.1646, -0.0987] | 2,285 | 0.2352 | -0.1254 [-0.1460, -0.1049] |
| late remex 4-bit centered | 8,538 | 0.7405 | -0.0042 [-0.0084, -0.0008] | 9,142 | 0.3605 | -0.0001 [-0.0025, +0.0023] |
| late remex 2-bit centered | 4,269 | 0.7306 | -0.0141 [-0.0248, -0.0045] | 4,571 | 0.3580 | -0.0026 [-0.0086, +0.0036] |
| late remex 1-bit centered | 2,134 | 0.6868 | -0.0580 [-0.0789, -0.0379] | 2,285 | 0.3382 | -0.0224 [-0.0326, -0.0122] |
| late remax k=1 (asym) | 2,134 | 0.5319 | -0.2128 [-0.2517, -0.1748] | 2,285 | 0.1906 | -0.1700 [-0.1955, -0.1449] |
| late remax k=2 (asym) | 4,269 | 0.6617 | -0.0830 [-0.1099, -0.0575] | 4,571 | 0.2593 | -0.1013 [-0.1218, -0.0814] |
| late remax k=1 (sym) | 2,134 | 0.4395 | -0.3052 [-0.3504, -0.2594] | 2,285 | 0.1564 | -0.2042 [-0.2317, -0.1765] |
| late plaid 4-bit | 9,072 (9,476) | 0.7382 | -0.0066 [-0.0121, -0.0017] | 9,713 (10,002) | 0.3590 | -0.0016 [-0.0052, +0.0018] |
| late plaid 2-bit | 4,803 (5,207) | 0.7315 | -0.0132 [-0.0240, -0.0035] | 5,142 (5,431) | 0.3541 | -0.0064 [-0.0124, -0.0008] |
| late plaid 1-bit | 2,668 (3,073) | 0.7153 | -0.0294 [-0.0452, -0.0142] | 2,857 (3,145) | 0.3368 | -0.0237 [-0.0328, -0.0150] |
| late plaid centroid-only | 534 (938) | 0.6873 | -0.0575 [-0.0786, -0.0371] | 571 (860) | 0.2750 | -0.0856 [-0.1044, -0.0678] |
| late fp32 pool2 | 34,216 | 0.7050 | -0.0397 [-0.0565, -0.0245] | 36,631 | 0.3560 | -0.0046 [-0.0105, +0.0012] |
| late remex 2-bit pool2 | 2,673 | 0.6365 | -0.1082 [-0.1375, -0.0797] | 2,862 | 0.3093 | -0.0513 [-0.0650, -0.0383] |
| late remex 1-bit pool2 | 1,604 | 0.5100 | -0.2348 [-0.2752, -0.1954] | 1,717 | 0.1832 | -0.1774 [-0.2029, -0.1523] |
| dense bge-small fp32 | 1,536 | 0.7127 | -0.0320 [-0.0587, -0.0050] | 1,536 | 0.3436 | -0.0170 [-0.0358, +0.0018] |
| dense bge-small fp16 | 768 | 0.7127 | -0.0320 [-0.0587, -0.0050] | 768 | 0.3435 | -0.0171 [-0.0359, +0.0017] |
| dense bge-small int8 | 384 | 0.7097 | -0.0351 [-0.0617, -0.0082] | 384 | 0.3435 | -0.0171 [-0.0362, +0.0019] |
| dense bge-small remex 4-bit | 192 | 0.7170 | -0.0278 [-0.0557, +0.0006] | 192 | 0.3418 | -0.0187 [-0.0372, -0.0000] |
| dense bge-small remex 2-bit | 96 | 0.6800 | -0.0648 [-0.0950, -0.0343] | 96 | 0.3330 | -0.0276 [-0.0468, -0.0090] |
| dense bge-small remex 1-bit | 48 | 0.6172 | -0.1275 [-0.1617, -0.0934] | 48 | 0.2916 | -0.0690 [-0.0904, -0.0486] |
| dense bge-small remax k=1 (asym) | 48 | 0.6126 | -0.1321 [-0.1672, -0.0979] | 48 | 0.2963 | -0.0643 [-0.0849, -0.0441] |
| late remex 2-bit centered haar s0 |  | 0.7297 | -0.0151 [-0.0254, -0.0055] |  | 0.3522 | -0.0084 [-0.0148, -0.0020] |
| late remex 2-bit centered haar s1 |  | 0.7403 | -0.0044 [-0.0136, +0.0045] |  | 0.3563 | -0.0043 [-0.0102, +0.0020] |
| late remex 1-bit centered haar s0 |  | 0.6927 | -0.0521 [-0.0719, -0.0334] |  | 0.3272 | -0.0333 [-0.0450, -0.0220] |
| late remex 1-bit centered haar s1 |  | 0.7008 | -0.0439 [-0.0625, -0.0257] |  | 0.3219 | -0.0386 [-0.0508, -0.0267] |
| late remax k=1 (asym) centered |  | 0.0057 | -0.7390 [-0.7784, -0.6973] |  | 0.0186 | -0.3420 [-0.3781, -0.3067] |
| late remax k=2 (asym) centered |  | 0.0022 | -0.7426 [-0.7825, -0.7001] |  | 0.0198 | -0.3408 [-0.3771, -0.3058] |

plaid bytes in parentheses include the fp16 centroid table (K × 64 × 2 B; K = 16,384 SciFact, 8,192 NFCorpus) amortised over the corpus. Follow-up arms (extra.py) have the same token count as their bench.py counterparts.

### Head-to-head (paired bootstrap, per-query ΔnDCG@10, a − b)

| a | b | SciFact | NFCorpus |
|---|---|---|---|
| late remex 2-bit centered | late plaid 2-bit | -0.0010 [-0.0128, +0.0109] (31/31) | +0.0038 [-0.0038, +0.0114] (82/87) |
| late remex 1-bit centered | late plaid 1-bit | -0.0286 [-0.0490, -0.0088] (40/57) | +0.0014 [-0.0105, +0.0133] (105/89) |
| late remex 4-bit centered | late plaid 4-bit | +0.0023 [-0.0021, +0.0072] (17/13) | +0.0015 [-0.0024, +0.0054] (61/57) |
| late remex 1-bit centered | late remex 1-bit | +0.1232 [+0.0913, +0.1551] (98/27) | +0.1387 [+0.1160, +0.1624] (191/28) |
| late remex 2-bit centered | dense bge-small fp32 | +0.0179 [-0.0090, +0.0442] (55/49) | +0.0144 [-0.0039, +0.0328] (116/104) |
| late remex 2-bit centered | dense bge-small remex 4-bit | +0.0136 [-0.0140, +0.0413] (57/48) | +0.0161 [-0.0016, +0.0341] (117/101) |
| late plaid 1-bit | dense bge-small fp32 | +0.0026 [-0.0276, +0.0326] (53/59) | -0.0067 [-0.0258, +0.0125] (105/115) |
| late remex 1-bit centered | dense bge-small fp32 | -0.0259 [-0.0536, +0.0005] (43/60) | -0.0054 [-0.0237, +0.0127] (101/118) |
| late fp32 | dense bge-small fp32 | +0.0320 [+0.0050, +0.0587] (60/45) | +0.0170 [-0.0018, +0.0358] (118/99) |

### Token geometry

| | SciFact | NFCorpus |
|---|---:|---:|
| mean_norm_of_mean | 0.936 | 0.94 |
| random_pair_cos_mean | 0.876 | 0.883 |
| random_pair_cos_p05 | 0.789 | 0.804 |
| centered_pair_cos_mean | 0.009 | 0.011 |
| residual_norm_mean | 0.322 | 0.307 |

scifact: 5,183 docs, 1,382,899 tokens (266.81/doc, 133.66/doc after pool2), 300 queries, bench 1218 s.

nfcorpus: 3,633 docs, 1,037,874 tokens (285.68/doc, 143.09/doc after pool2), 323 queries, bench 592 s.
<!-- tables:end -->

## Reproduce

```
python3 encode.py            # writes .work/<corpus>/ (gitignored), ~19 min on 4 vCPU
python3 bench.py             # results_<corpus>.json + per-query npz, ~30 min
python3 extra.py             # results_extra_<corpus>.json, ~7 min
python3 report.py            # renders the tables above
python3 recheck.py           # prose vs artifacts, seconds
```

remax 0.3.0 is installed from `github.com/oaustegard/remax` (not on PyPI). remex 0.8.0 is on PyPI.
`bench.py` imports `seg_max_mean`, `paired_bootstrap` and `signs_pm1` from `../neomme-remex-quant/neomme_quant.py`.
