# neomme-remex-quant — remex and remax quantization of both NeoMME-260M-Retriever heads

**Started:** 2026-09-07 · **Status:** done (SciFact text; ViDoRe DocVQA + ShiftProject page images) · **Runtime:** SciFact 39 + 14 min; DocVQA 50 + 18 min; ShiftProject 83 min encode (two restarts) + 26 min bench, 4 vCPU

## Question

Tom Aarsen's Bluesky thread (2026-09-07) announced H Company's NeoMME: 260M and
800M multimodal encoders with a dense head (1024-d, Matryoshka 128/256/512/1024)
and a late-interaction head (128-d per token, MeanMaxSim), both from one forward
pass. Oskar asked whether H Company ships a quantized-vector version and, if not,
to build remex/remax quantization for it.

## Prior-art check

**Correction (2026-09-08).** The check below stopped at the model card and the
Hub artifacts and did not read the paper the card links
([arXiv:2609.01657](https://arxiv.org/abs/2609.01657)). Its abstract and
section 5.9 report exactly this study's second half on the token head: int8
documents at 3.9× cost 0.0002 nDCG@10; sign-bit binary documents with int8
queries cost 1.58 points at 32×; pool factor 8 + binary documents reach 6.0 kB
per page, 255×, at 95.19% retained (ViDoRe v3, 260M). What the paper does not
do: quantize the dense head (Matryoshka only), test on text, or rotate before
binarising. Section 5.9 also states the text-vs-pages pooling asymmetry (near-lossless to
factor 3 on text per Clavié et al., to factor 10 on pages) with the same
explanation as finding 9. Section 5.2 is the source of the dense-then-late
recipe finding 6 tests. Findings 1–4 and the dense half of 12 stand as novel;
findings 2, 6, 8, 9 and 11 are replications or tests of the paper's own
statements, and should have been framed that way from the start. `ERRORS.md`
#5.

H Company ships no quantized vectors. All ten repos in the `Hcompany/neomme`
collection hold `model.safetensors` and configs only. No `onnx/`, no int8, no
binary or GGUF siblings, no derivative repos (HF search for "neomme" returns
the ten official ones). The model cards mention no `precision=` option. The two
size levers the model does ship are Matryoshka truncation on the dense head and,
through Sentence Transformers, `HierarchicalTokenPooling` on the token head. The
bench below measures every codec against those two.

Sentence Transformers' own `quantize_embeddings` (int8 with per-dimension
calibration ranges, and binary = uncentered sign bits) applies to the dense
head, so it appears as an arm too.

## Setup

- **Model:** `Hcompany/NeoMME-260M-Retriever` via `NeoMMEForRetrieval`
  (transformers `5.17.0.dev0` from main; 5.16.1 has no NeoMME), float32 on CPU.
  Both heads from one pass; `encode.py`, settings in `data/scifact_enc/meta.json`.
- **Corpus:** BEIR SciFact, 5,183 docs (`title + " " + text`), 300 judged test
  queries, 339 qrels pairs. Docs average 320 tokens; `max_length=1024` truncates
  none of the 5,183 below the model's 16k context.
- **Codecs:** remex (`Quantizer`, Lloyd-Max on an RHT-rotated sphere, exact
  float32 norm stored) at 4/2/1 bits; remax (`StackedSignBitQuantizer`, k sign
  codes, RHT) with the query kept in float ("asym") or binarised ("sym");
  seed 0 for every arm.
- **Dense arms:** fp32 at each Matryoshka dim; remex at each bits × dim; remax
  k=1 at each dim, k=2 at 1024/512; ST int8 and ST binary at 1024.
- **Late arms:** per-token codes, scored with MeanMaxSim in the codec's
  rotated space (decode corpus once, rotate the query tokens, one GEMM per
  query, segmented max over document boundaries). Token pooling at factor 2
  and 4 re-uses `sentence_transformers`' `_hierarchical_pool_one`, so pooled
  indexes match an ST pipeline byte for byte.
- **Pipeline arms:** the thread's suggested deployment: dense top-100
  candidates, late-interaction rerank, with each side fp32 or quantized.
- **Metrics:** nDCG@10, R@10, R@100 against qrels; top-10 overlap with the
  fp32 full-width ranking of the same head (fidelity); bytes per document as
  stored (remex byte counts are direction only — every stored norm is 1.0 on
  these L2-normalised vectors, so the 4 B/vector remex writes anyway is left
  out of the table; add it back if you keep `CompressedVectors` as is). Paired
  bootstrap (5,000 resamples) on per-query ΔnDCG@10 against the same head's
  fp32 reference.

`neomme_quant.py` is the reusable part: `DenseIndex` and `MultiVectorIndex`
over NeoMME's two heads, self-tested against `sentence_transformers.util.mean_maxsim`.

## Results

### SciFact (text)

<!-- tables:start -->
### Dense head over 5,183 documents

| arm | bytes/doc | nDCG@10 | Δ vs fp32 [95% CI] | wins/losses | R@10 | R@100 | top-10 overlap w/ fp32 |
|---|---:|---:|---|---:|---:|---:|---:|
| dense fp32 d=1024 | 4.0 KB | 0.5527 | — | 0/0 | 0.678 | 0.864 | — |
| dense fp32 d=512 | 2.0 KB | 0.5463 | -0.0064 [-0.0171, +0.0046] | 24/32 | 0.678 | 0.861 | 0.859 |
| dense fp32 d=256 | 1.0 KB | 0.5381 | -0.0145 [-0.0292, +0.0002] | 28/43 | 0.675 | 0.856 | 0.781 |
| dense ST int8 d=1024 | 1.0 KB | 0.5525 | -0.0002 [-0.0005, +0.0000] | 0/2 | 0.678 | 0.864 | 0.996 |
| dense fp32 d=128 | 512 B | 0.5153 * | -0.0373 [-0.0598, -0.0147] | 33/59 | 0.646 | 0.853 | 0.657 |
| dense remex 4-bit d=1024 | 512 B | 0.5509 | -0.0018 [-0.0116, +0.0082] | 26/26 | 0.688 | 0.861 | 0.910 |
| dense remex 4-bit d=512 | 256 B | 0.5490 | -0.0036 [-0.0164, +0.0092] | 23/43 | 0.686 | 0.861 | 0.813 |
| dense remex 2-bit d=1024 | 256 B | 0.5305 * | -0.0221 [-0.0395, -0.0045] | 29/56 | 0.666 | 0.846 | 0.754 |
| dense remax k=2 d=1024 (asym) | 256 B | 0.5278 * | -0.0249 [-0.0397, -0.0103] | 22/54 | 0.642 | 0.853 | 0.752 |
| dense remex 4-bit d=256 | 128 B | 0.5300 * | -0.0226 [-0.0392, -0.0072] | 26/54 | 0.661 | 0.859 | 0.722 |
| dense remex 2-bit d=512 | 128 B | 0.5098 * | -0.0429 [-0.0624, -0.0236] | 26/62 | 0.631 | 0.846 | 0.658 |
| dense remex 1-bit d=1024 | 128 B | 0.5059 * | -0.0467 [-0.0662, -0.0276] | 20/65 | 0.632 | 0.829 | 0.657 |
| dense remax k=1 d=1024 (asym) | 128 B | 0.5120 * | -0.0407 [-0.0583, -0.0237] | 23/61 | 0.625 | 0.834 | 0.657 |
| dense remax k=1 d=1024 (sym) | 128 B | 0.4849 * | -0.0677 [-0.0914, -0.0443] | 24/78 | 0.604 | 0.808 | 0.553 |
| dense remax k=2 d=512 (asym) | 128 B | 0.5173 * | -0.0354 [-0.0577, -0.0150] | 29/55 | 0.630 | 0.823 | 0.647 |
| dense ST binary d=1024 (asym) | 128 B | 0.5055 * | -0.0472 [-0.0657, -0.0290] | 24/65 | 0.645 | 0.818 | 0.658 |
| dense ST binary d=1024 (sym) | 128 B | 0.4724 * | -0.0802 [-0.1058, -0.0558] | 24/76 | 0.591 | 0.803 | 0.551 |
| dense remex 4-bit d=128 | 64 B | 0.5021 * | -0.0505 [-0.0743, -0.0262] | 27/71 | 0.626 | 0.857 | 0.608 |
| dense remex 2-bit d=256 | 64 B | 0.4939 * | -0.0588 [-0.0869, -0.0312] | 30/77 | 0.619 | 0.839 | 0.544 |
| dense remex 1-bit d=512 | 64 B | 0.4819 * | -0.0707 [-0.0964, -0.0459] | 28/82 | 0.605 | 0.807 | 0.539 |
| dense remax k=1 d=512 (asym) | 64 B | 0.4822 * | -0.0705 [-0.0958, -0.0459] | 23/77 | 0.591 | 0.785 | 0.536 |
| dense remax k=1 d=512 (sym) | 64 B | 0.4425 * | -0.1101 [-0.1396, -0.0807] | 25/93 | 0.554 | 0.765 | 0.420 |
| dense remex 2-bit d=128 | 32 B | 0.4156 * | -0.1371 [-0.1691, -0.1049] | 21/106 | 0.527 | 0.746 | 0.394 |
| dense remex 1-bit d=256 | 32 B | 0.4042 * | -0.1485 [-0.1822, -0.1154] | 24/114 | 0.530 | 0.789 | 0.401 |
| dense remax k=1 d=256 (asym) | 32 B | 0.4272 * | -0.1255 [-0.1583, -0.0932] | 21/94 | 0.521 | 0.737 | 0.383 |
| dense remax k=1 d=256 (sym) | 32 B | 0.3814 * | -0.1713 [-0.2066, -0.1354] | 14/115 | 0.484 | 0.667 | 0.287 |
| dense remex 1-bit d=128 | 16 B | 0.3255 * | -0.2272 [-0.2660, -0.1880] | 14/130 | 0.422 | 0.683 | 0.260 |
| dense remax k=1 d=128 (asym) | 16 B | 0.3062 * | -0.2464 [-0.2885, -0.2038] | 13/138 | 0.386 | 0.636 | 0.240 |
| dense remax k=1 d=128 (sym) | 16 B | 0.2180 * | -0.3346 [-0.3798, -0.2895] | 13/167 | 0.312 | 0.535 | 0.159 |

### Late-interaction head scored with MeanMaxSim

| arm | bytes/doc | nDCG@10 | Δ vs fp32 [95% CI] | wins/losses | R@10 | R@100 | top-10 overlap w/ fp32 |
|---|---:|---:|---|---:|---:|---:|---:|
| late fp32 | 159.8 KB | 0.7198 | — | 0/0 | 0.841 | 0.943 | — |
| late fp32 pool2 | 80.0 KB | 0.7009 * | -0.0189 [-0.0294, -0.0094] | 14/40 | 0.826 | 0.925 | 0.863 |
| late fp32 pool4 | 40.1 KB | 0.6937 * | -0.0262 [-0.0419, -0.0118] | 25/46 | 0.807 | 0.932 | 0.736 |
| late remex 4-bit | 20.0 KB | 0.7152 | -0.0046 [-0.0109, +0.0010] | 10/19 | 0.835 | 0.940 | 0.941 |
| late remex 2-bit | 10.0 KB | 0.7066 * | -0.0132 [-0.0253, -0.0023] | 20/33 | 0.829 | 0.936 | 0.837 |
| late remax k=2 (asym) | 10.0 KB | 0.7070 * | -0.0129 [-0.0246, -0.0021] | 21/34 | 0.817 | 0.926 | 0.829 |
| late remex 2-bit pool2 | 5.0 KB | 0.6832 * | -0.0366 [-0.0517, -0.0224] | 16/52 | 0.800 | 0.930 | 0.780 |
| late remex 1-bit | 5.0 KB | 0.7070 | -0.0129 [-0.0283, +0.0022] | 25/42 | 0.816 | 0.936 | 0.764 |
| late remax k=1 (asym) | 5.0 KB | 0.6950 * | -0.0248 [-0.0401, -0.0108] | 23/47 | 0.824 | 0.930 | 0.759 |
| late remax k=1 (sym) | 5.0 KB | 0.6940 * | -0.0258 [-0.0450, -0.0082] | 28/51 | 0.816 | 0.931 | 0.702 |
| late remex 1-bit pool2 | 2.5 KB | 0.6767 * | -0.0432 [-0.0600, -0.0281] | 15/55 | 0.793 | 0.925 | 0.722 |
| late remax k=1 (asym) pool2 | 2.5 KB | 0.6840 * | -0.0359 [-0.0529, -0.0205] | 18/54 | 0.801 | 0.911 | 0.737 |

### Dense top-100 candidates reranked by late interaction

| arm | bytes/doc | nDCG@10 | Δ vs fp32 [95% CI] | wins/losses | R@10 | R@100 | top-10 overlap w/ fp32 |
|---|---:|---:|---|---:|---:|---:|---:|
| fp32 d=1024 -> late fp32 | 163.8 KB | 0.6854 | — | 0/0 | 0.786 | 0.864 | — |
| remex 2-bit d=1024 -> late fp32 | 160.1 KB | 0.6764 * | -0.0090 [-0.0194, -0.0010] | 4/9 | 0.765 | 0.846 | — |
| remax k=1 d=1024 (asym) -> late fp32 | 160.0 KB | 0.6758 | -0.0097 [-0.0220, +0.0004] | 8/7 | 0.768 | 0.834 | — |
| remex 2-bit d=256 -> late fp32 | 159.9 KB | 0.6795 | -0.0060 [-0.0223, +0.0080] | 17/11 | 0.770 | 0.839 | — |
| fp32 d=1024 -> late remex 2-bit | 14.0 KB | 0.6814 | -0.0041 [-0.0132, +0.0045] | 20/24 | 0.786 | 0.864 | — |
| remex 2-bit d=1024 -> late remex 2-bit | 10.2 KB | 0.6700 * | -0.0154 [-0.0295, -0.0031] | 18/32 | 0.763 | 0.846 | — |
| remax k=1 d=1024 (asym) -> late remex 2-bit | 10.1 KB | 0.6718 * | -0.0136 [-0.0286, -0.0003] | 19/29 | 0.765 | 0.834 | — |
| remex 2-bit d=256 -> late remex 2-bit | 10.1 KB | 0.6754 | -0.0101 [-0.0279, +0.0068] | 24/31 | 0.770 | 0.839 | — |
| fp32 d=1024 -> late remex 1-bit | 9.0 KB | 0.6807 | -0.0048 [-0.0177, +0.0070] | 22/30 | 0.777 | 0.864 | — |
| fp32 d=1024 -> late remax k=1 (asym) | 9.0 KB | 0.6722 * | -0.0133 [-0.0255, -0.0012] | 20/35 | 0.773 | 0.864 | — |
| remex 2-bit d=1024 -> late remex 1-bit | 5.2 KB | 0.6701 * | -0.0153 [-0.0313, -0.0008] | 22/37 | 0.760 | 0.846 | — |
| remex 2-bit d=1024 -> late remax k=1 (asym) | 5.2 KB | 0.6647 * | -0.0207 [-0.0352, -0.0074] | 17/39 | 0.758 | 0.846 | — |
| remax k=1 d=1024 (asym) -> late remex 1-bit | 5.1 KB | 0.6705 | -0.0150 [-0.0318, +0.0006] | 24/36 | 0.762 | 0.834 | — |
| remax k=1 d=1024 (asym) -> late remax k=1 (asym) | 5.1 KB | 0.6631 * | -0.0223 [-0.0394, -0.0066] | 20/40 | 0.760 | 0.834 | — |
| remex 2-bit d=256 -> late remex 1-bit | 5.1 KB | 0.6714 | -0.0140 [-0.0341, +0.0045] | 26/36 | 0.760 | 0.839 | — |
| remex 2-bit d=256 -> late remax k=1 (asym) | 5.1 KB | 0.6649 * | -0.0206 [-0.0396, -0.0026] | 24/43 | 0.763 | 0.839 | — |

`*` = 95% paired-bootstrap CI on ΔnDCG@10 excludes zero (300 queries, 5,000 resamples). Wins/losses count queries whose nDCG@10 moved vs the reference; ties omitted.
<!-- tables:end -->

### ViDoRe DocVQA page images

`vidore/docvqa_test_subsampled`: 500 scanned document pages, 500 real DocVQA
questions, one relevant page per query (its source page). Pages encoded as
images only, no OCR text, 2,921 tokens per page on average (`encode_vidore.py`,
settings in `data/vidore_docvqa_enc/meta.json`). Same arms, same seeds. A
page's fp32 token index is 1.5 MB; the dense vector is still 4 KB.

<!-- tables:docvqa:start -->
### Dense head over 5,183 documents

| arm | bytes/doc | nDCG@10 | Δ vs fp32 [95% CI] | wins/losses | R@10 | R@100 | top-10 overlap w/ fp32 |
|---|---:|---:|---|---:|---:|---:|---:|
| dense fp32 d=1024 | 4.0 KB | 0.3847 | — | 0/0 | 0.486 | 0.718 | — |
| dense fp32 d=512 | 2.0 KB | 0.3829 | -0.0019 [-0.0105, +0.0069] | 36/44 | 0.490 | 0.730 | 0.854 |
| dense fp32 d=256 | 1.0 KB | 0.3699 * | -0.0149 [-0.0262, -0.0035] | 36/64 | 0.472 | 0.734 | 0.782 |
| dense ST int8 d=1024 | 1.0 KB | 0.3834 | -0.0013 [-0.0034, +0.0000] | 0/2 | 0.484 | 0.718 | 0.997 |
| dense fp32 d=128 | 512 B | 0.3612 * | -0.0235 [-0.0361, -0.0110] | 36/73 | 0.466 | 0.720 | 0.688 |
| dense remex 4-bit d=1024 | 512 B | 0.3779 * | -0.0069 [-0.0134, -0.0003] | 21/33 | 0.478 | 0.722 | 0.913 |
| dense remex 4-bit d=512 | 256 B | 0.3813 | -0.0034 [-0.0127, +0.0062] | 36/43 | 0.486 | 0.728 | 0.824 |
| dense remex 2-bit d=1024 | 256 B | 0.3798 | -0.0049 [-0.0151, +0.0053] | 40/48 | 0.488 | 0.742 | 0.770 |
| dense remax k=2 d=1024 (asym) | 256 B | 0.3598 * | -0.0249 [-0.0369, -0.0126] | 31/62 | 0.466 | 0.730 | 0.756 |
| dense remex 4-bit d=256 | 128 B | 0.3687 * | -0.0160 [-0.0281, -0.0043] | 35/62 | 0.466 | 0.730 | 0.735 |
| dense remex 2-bit d=512 | 128 B | 0.3710 * | -0.0137 [-0.0263, -0.0011] | 44/70 | 0.486 | 0.730 | 0.692 |
| dense remex 1-bit d=1024 | 128 B | 0.3598 * | -0.0249 [-0.0397, -0.0110] | 38/67 | 0.468 | 0.738 | 0.675 |
| dense remax k=1 d=1024 (asym) | 128 B | 0.3509 * | -0.0338 [-0.0491, -0.0188] | 36/75 | 0.458 | 0.722 | 0.669 |
| dense remax k=1 d=1024 (sym) | 128 B | 0.3157 * | -0.0691 [-0.0875, -0.0507] | 31/106 | 0.428 | 0.696 | 0.582 |
| dense remax k=2 d=512 (asym) | 128 B | 0.3634 * | -0.0213 [-0.0361, -0.0063] | 45/72 | 0.476 | 0.728 | 0.658 |
| dense ST binary d=1024 (asym) | 128 B | 0.3630 * | -0.0217 [-0.0357, -0.0071] | 39/70 | 0.466 | 0.712 | 0.659 |
| dense ST binary d=1024 (sym) | 128 B | 0.3355 * | -0.0492 [-0.0657, -0.0323] | 36/97 | 0.444 | 0.700 | 0.563 |
| dense remex 4-bit d=128 | 64 B | 0.3563 * | -0.0284 [-0.0430, -0.0146] | 38/77 | 0.462 | 0.718 | 0.659 |
| dense remex 2-bit d=256 | 64 B | 0.3447 * | -0.0400 [-0.0570, -0.0235] | 42/90 | 0.456 | 0.716 | 0.594 |
| dense remex 1-bit d=512 | 64 B | 0.3334 * | -0.0513 [-0.0698, -0.0332] | 43/90 | 0.446 | 0.740 | 0.575 |
| dense remax k=1 d=512 (asym) | 64 B | 0.3394 * | -0.0454 [-0.0644, -0.0266] | 40/93 | 0.452 | 0.718 | 0.558 |
| dense remax k=1 d=512 (sym) | 64 B | 0.3064 * | -0.0783 [-0.0997, -0.0563] | 36/113 | 0.412 | 0.696 | 0.458 |
| dense remex 2-bit d=128 | 32 B | 0.3114 * | -0.0733 [-0.0952, -0.0521] | 41/115 | 0.436 | 0.706 | 0.477 |
| dense remex 1-bit d=256 | 32 B | 0.2889 * | -0.0958 [-0.1186, -0.0727] | 35/132 | 0.410 | 0.714 | 0.444 |
| dense remax k=1 d=256 (asym) | 32 B | 0.2923 * | -0.0924 [-0.1149, -0.0702] | 33/125 | 0.404 | 0.680 | 0.447 |
| dense remax k=1 d=256 (sym) | 32 B | 0.2396 * | -0.1451 [-0.1726, -0.1180] | 27/158 | 0.346 | 0.652 | 0.331 |
| dense remex 1-bit d=128 | 16 B | 0.2458 * | -0.1389 [-0.1658, -0.1104] | 28/150 | 0.344 | 0.676 | 0.319 |
| dense remax k=1 d=128 (asym) | 16 B | 0.2362 * | -0.1485 [-0.1766, -0.1210] | 30/151 | 0.332 | 0.626 | 0.328 |
| dense remax k=1 d=128 (sym) | 16 B | 0.1782 * | -0.2066 [-0.2375, -0.1754] | 19/183 | 0.268 | 0.602 | 0.225 |

### Late-interaction head scored with MeanMaxSim

| arm | bytes/doc | nDCG@10 | Δ vs fp32 [95% CI] | wins/losses | R@10 | R@100 | top-10 overlap w/ fp32 |
|---|---:|---:|---|---:|---:|---:|---:|
| late fp32 | 1460.3 KB | 0.5188 | — | 0/0 | 0.624 | 0.814 | — |
| late fp32 pool2 | 730.2 KB | 0.5150 | -0.0038 [-0.0096, +0.0019] | 21/25 | 0.612 | 0.820 | 0.926 |
| late fp32 pool4 | 365.3 KB | 0.5117 | -0.0070 [-0.0145, +0.0001] | 27/35 | 0.612 | 0.820 | 0.873 |
| late remex 4-bit | 182.5 KB | 0.5162 | -0.0025 [-0.0077, +0.0024] | 20/25 | 0.616 | 0.816 | 0.943 |
| late remex 2-bit | 91.3 KB | 0.5150 | -0.0037 [-0.0130, +0.0054] | 37/43 | 0.614 | 0.806 | 0.832 |
| late remax k=2 (asym) | 91.3 KB | 0.5134 | -0.0053 [-0.0143, +0.0036] | 32/41 | 0.612 | 0.804 | 0.822 |
| late remex 2-bit pool2 | 45.6 KB | 0.5126 | -0.0062 [-0.0155, +0.0025] | 37/39 | 0.618 | 0.800 | 0.808 |
| late remex 1-bit | 45.6 KB | 0.5146 | -0.0042 [-0.0148, +0.0065] | 38/46 | 0.614 | 0.810 | 0.738 |
| late remax k=1 (asym) | 45.6 KB | 0.5105 | -0.0083 [-0.0206, +0.0043] | 33/44 | 0.604 | 0.800 | 0.751 |
| late remax k=1 (sym) | 45.6 KB | 0.5083 | -0.0105 [-0.0231, +0.0019] | 35/52 | 0.614 | 0.806 | 0.699 |
| late remex 1-bit pool2 | 22.8 KB | 0.5149 | -0.0038 [-0.0149, +0.0076] | 38/42 | 0.614 | 0.808 | 0.719 |
| late remax k=1 (asym) pool2 | 22.8 KB | 0.5048 * | -0.0140 [-0.0253, -0.0025] | 31/46 | 0.600 | 0.802 | 0.727 |

### Dense top-100 candidates reranked by late interaction

| arm | bytes/doc | nDCG@10 | Δ vs fp32 [95% CI] | wins/losses | R@10 | R@100 | top-10 overlap w/ fp32 |
|---|---:|---:|---|---:|---:|---:|---:|
| fp32 d=1024 -> late fp32 | 1464.3 KB | 0.5082 | — | 0/0 | 0.604 | 0.718 | — |
| remex 2-bit d=1024 -> late fp32 | 1460.6 KB | 0.5145 * | +0.0063 [+0.0013, +0.0119] | 15/7 | 0.616 | 0.742 | — |
| remax k=1 d=1024 (asym) -> late fp32 | 1460.4 KB | 0.4983 * | -0.0099 [-0.0190, -0.0016] | 10/16 | 0.590 | 0.722 | — |
| remex 2-bit d=256 -> late fp32 | 1460.4 KB | 0.5025 | -0.0057 [-0.0154, +0.0028] | 17/13 | 0.594 | 0.716 | — |
| fp32 d=1024 -> late remex 2-bit | 95.3 KB | 0.5082 | -0.0000 [-0.0081, +0.0083] | 33/36 | 0.604 | 0.718 | — |
| remex 2-bit d=1024 -> late remex 2-bit | 91.5 KB | 0.5151 | +0.0069 [-0.0029, +0.0167] | 39/35 | 0.616 | 0.742 | — |
| remax k=1 d=1024 (asym) -> late remex 2-bit | 91.4 KB | 0.4961 * | -0.0121 [-0.0238, -0.0010] | 32/45 | 0.588 | 0.722 | — |
| remex 2-bit d=256 -> late remex 2-bit | 91.3 KB | 0.5051 | -0.0031 [-0.0158, +0.0085] | 40/36 | 0.602 | 0.716 | — |
| fp32 d=1024 -> late remex 1-bit | 49.6 KB | 0.5030 | -0.0051 [-0.0156, +0.0052] | 35/42 | 0.592 | 0.718 | — |
| fp32 d=1024 -> late remax k=1 (asym) | 49.6 KB | 0.5002 | -0.0079 [-0.0199, +0.0042] | 31/41 | 0.580 | 0.718 | — |
| remex 2-bit d=1024 -> late remex 1-bit | 45.9 KB | 0.5081 | -0.0000 [-0.0112, +0.0109] | 40/45 | 0.600 | 0.742 | — |
| remex 2-bit d=1024 -> late remax k=1 (asym) | 45.9 KB | 0.5050 | -0.0032 [-0.0158, +0.0096] | 34/44 | 0.590 | 0.742 | — |
| remax k=1 d=1024 (asym) -> late remex 1-bit | 45.8 KB | 0.4939 * | -0.0143 [-0.0273, -0.0021] | 35/51 | 0.580 | 0.722 | — |
| remax k=1 d=1024 (asym) -> late remax k=1 (asym) | 45.8 KB | 0.4945 | -0.0137 [-0.0281, +0.0006] | 32/45 | 0.574 | 0.722 | — |
| remex 2-bit d=256 -> late remex 1-bit | 45.7 KB | 0.4992 | -0.0090 [-0.0227, +0.0038] | 38/47 | 0.588 | 0.716 | — |
| remex 2-bit d=256 -> late remax k=1 (asym) | 45.7 KB | 0.4958 | -0.0123 [-0.0271, +0.0018] | 34/45 | 0.580 | 0.716 | — |

`*` = 95% paired-bootstrap CI on ΔnDCG@10 excludes zero (500 queries, 5,000 resamples). Wins/losses count queries whose nDCG@10 moved vs the reference; ties omitted.
<!-- tables:docvqa:end -->

### ViDoRe ShiftProject page images

`vidore/shiftproject_test`: 1,000 pages of environmental reports (the Shift
Project), 100 queries generated from the pages, one relevant page each. 3,010
tokens per page (every page is 1654×2339 px, so every page is 3,010 patches);
fp32 token index 1.5 MB/page. An easy corpus: the fp32 token index gets
R@10 0.99 and R@100 1.00, so intervals are wide (100 queries) and ceiling
effects apply.

<!-- tables:shift:start -->
### Dense head over 5,183 documents

| arm | bytes/doc | nDCG@10 | Δ vs fp32 [95% CI] | wins/losses | R@10 | R@100 | top-10 overlap w/ fp32 |
|---|---:|---:|---|---:|---:|---:|---:|
| dense fp32 d=1024 | 4.0 KB | 0.7581 | — | 0/0 | 0.930 | 0.990 | — |
| dense fp32 d=512 | 2.0 KB | 0.7536 | -0.0044 [-0.0186, +0.0084] | 6/10 | 0.930 | 0.990 | 0.894 |
| dense fp32 d=256 | 1.0 KB | 0.7304 * | -0.0277 [-0.0557, -0.0000] | 8/19 | 0.910 | 0.970 | 0.838 |
| dense ST int8 d=1024 | 1.0 KB | 0.7581 | +0.0000 [+0.0000, +0.0000] | 0/0 | 0.930 | 0.990 | 0.995 |
| dense fp32 d=128 | 512 B | 0.7227 * | -0.0353 [-0.0671, -0.0061] | 10/20 | 0.890 | 0.990 | 0.778 |
| dense remex 4-bit d=1024 | 512 B | 0.7538 | -0.0043 [-0.0210, +0.0126] | 7/12 | 0.920 | 0.990 | 0.943 |
| dense remex 4-bit d=512 | 256 B | 0.7383 * | -0.0198 [-0.0400, -0.0010] | 4/14 | 0.920 | 0.990 | 0.859 |
| dense remex 2-bit d=1024 | 256 B | 0.7549 | -0.0032 [-0.0364, +0.0302] | 11/18 | 0.920 | 0.990 | 0.836 |
| dense remax k=2 d=1024 (asym) | 256 B | 0.7503 | -0.0077 [-0.0266, +0.0115] | 11/12 | 0.910 | 0.980 | 0.814 |
| dense remex 4-bit d=256 | 128 B | 0.7362 | -0.0219 [-0.0493, +0.0026] | 10/17 | 0.930 | 0.980 | 0.798 |
| dense remex 2-bit d=512 | 128 B | 0.7294 * | -0.0286 [-0.0588, -0.0003] | 10/20 | 0.920 | 0.990 | 0.767 |
| dense remex 1-bit d=1024 | 128 B | 0.6958 * | -0.0622 [-0.0937, -0.0333] | 7/27 | 0.880 | 0.990 | 0.753 |
| dense remax k=1 d=1024 (asym) | 128 B | 0.7357 | -0.0224 [-0.0507, +0.0056] | 8/20 | 0.910 | 0.990 | 0.746 |
| dense remax k=1 d=1024 (sym) | 128 B | 0.7035 * | -0.0546 [-0.0942, -0.0164] | 11/25 | 0.880 | 0.980 | 0.697 |
| dense remax k=2 d=512 (asym) | 128 B | 0.7229 * | -0.0351 [-0.0683, -0.0022] | 12/17 | 0.900 | 0.980 | 0.748 |
| dense ST binary d=1024 (asym) | 128 B | 0.7357 | -0.0224 [-0.0532, +0.0081] | 7/20 | 0.900 | 0.970 | 0.748 |
| dense ST binary d=1024 (sym) | 128 B | 0.7189 * | -0.0392 [-0.0761, -0.0017] | 9/24 | 0.880 | 0.970 | 0.659 |
| dense remex 4-bit d=128 | 64 B | 0.7175 * | -0.0406 [-0.0761, -0.0068] | 11/23 | 0.870 | 0.990 | 0.737 |
| dense remex 2-bit d=256 | 64 B | 0.6731 * | -0.0850 [-0.1307, -0.0408] | 8/29 | 0.870 | 0.980 | 0.636 |
| dense remex 1-bit d=512 | 64 B | 0.6587 * | -0.0994 [-0.1440, -0.0557] | 8/35 | 0.870 | 0.980 | 0.664 |
| dense remax k=1 d=512 (asym) | 64 B | 0.6981 * | -0.0600 [-0.1018, -0.0165] | 11/30 | 0.900 | 0.990 | 0.647 |
| dense remax k=1 d=512 (sym) | 64 B | 0.6581 * | -0.1000 [-0.1488, -0.0538] | 11/34 | 0.840 | 0.990 | 0.563 |
| dense remex 2-bit d=128 | 32 B | 0.5979 * | -0.1602 [-0.2252, -0.0951] | 13/43 | 0.790 | 0.960 | 0.545 |
| dense remex 1-bit d=256 | 32 B | 0.6065 * | -0.1516 [-0.2110, -0.0941] | 11/43 | 0.810 | 0.970 | 0.538 |
| dense remax k=1 d=256 (asym) | 32 B | 0.6176 * | -0.1405 [-0.1986, -0.0834] | 9/41 | 0.810 | 0.970 | 0.551 |
| dense remax k=1 d=256 (sym) | 32 B | 0.5383 * | -0.2198 [-0.2849, -0.1585] | 5/48 | 0.750 | 0.980 | 0.463 |
| dense remex 1-bit d=128 | 16 B | 0.4579 * | -0.3001 [-0.3672, -0.2335] | 3/59 | 0.680 | 0.920 | 0.427 |
| dense remax k=1 d=128 (asym) | 16 B | 0.4816 * | -0.2765 [-0.3421, -0.2132] | 7/57 | 0.690 | 0.960 | 0.430 |
| dense remax k=1 d=128 (sym) | 16 B | 0.4268 * | -0.3312 [-0.4069, -0.2566] | 4/59 | 0.590 | 0.920 | 0.321 |

### Late-interaction head scored with MeanMaxSim

| arm | bytes/doc | nDCG@10 | Δ vs fp32 [95% CI] | wins/losses | R@10 | R@100 | top-10 overlap w/ fp32 |
|---|---:|---:|---|---:|---:|---:|---:|
| late fp32 | 1504.9 KB | 0.9234 | — | 0/0 | 0.990 | 1.000 | — |
| late fp32 pool2 | 752.5 KB | 0.9308 | +0.0074 [-0.0074, +0.0221] | 3/1 | 0.990 | 1.000 | 0.955 |
| late fp32 pool4 | 376.5 KB | 0.9228 | -0.0007 [-0.0185, +0.0178] | 3/4 | 0.990 | 1.000 | 0.904 |
| late remex 4-bit | 188.1 KB | 0.9256 | +0.0022 [-0.0080, +0.0128] | 4/1 | 0.990 | 1.000 | 0.955 |
| late remex 2-bit | 94.1 KB | 0.9243 | +0.0009 [-0.0226, +0.0254] | 7/6 | 0.990 | 1.000 | 0.883 |
| late remax k=2 (asym) | 94.1 KB | 0.9320 | +0.0085 [-0.0100, +0.0283] | 5/3 | 0.990 | 1.000 | 0.866 |
| late remex 1-bit | 47.0 KB | 0.9395 | +0.0161 [-0.0062, +0.0396] | 7/3 | 0.990 | 1.000 | 0.827 |
| late remax k=1 (asym) | 47.0 KB | 0.9143 | -0.0091 [-0.0339, +0.0148] | 7/9 | 0.990 | 1.000 | 0.812 |
| late remax k=1 (sym) | 47.0 KB | 0.9127 | -0.0108 [-0.0394, +0.0171] | 8/9 | 0.990 | 1.000 | 0.788 |
| late remex 2-bit pool2 | 47.0 KB | 0.9169 | -0.0065 [-0.0250, +0.0115] | 4/4 | 0.990 | 1.000 | 0.883 |
| late remex 1-bit pool2 | 23.5 KB | 0.9268 | +0.0034 [-0.0193, +0.0274] | 6/7 | 0.990 | 1.000 | 0.814 |
| late remax k=1 (asym) pool2 | 23.5 KB | 0.8923 * | -0.0311 [-0.0619, -0.0011] | 7/13 | 0.990 | 1.000 | 0.817 |

### Dense top-100 candidates reranked by late interaction

| arm | bytes/doc | nDCG@10 | Δ vs fp32 [95% CI] | wins/losses | R@10 | R@100 | top-10 overlap w/ fp32 |
|---|---:|---:|---|---:|---:|---:|---:|
| fp32 d=1024 -> late fp32 | 1508.9 KB | 0.9134 | — | 0/0 | 0.980 | 0.990 | — |
| remex 2-bit d=1024 -> late fp32 | 1505.2 KB | 0.9134 | +0.0000 [+0.0000, +0.0000] | 0/0 | 0.980 | 0.990 | — |
| remax k=1 d=1024 (asym) -> late fp32 | 1505.1 KB | 0.9134 | +0.0000 [+0.0000, +0.0000] | 0/0 | 0.980 | 0.990 | — |
| remex 2-bit d=256 -> late fp32 | 1505.0 KB | 0.9034 | -0.0100 [-0.0300, +0.0000] | 0/1 | 0.970 | 0.980 | — |
| fp32 d=1024 -> late remex 2-bit | 98.1 KB | 0.9143 | +0.0009 [-0.0226, +0.0254] | 7/6 | 0.980 | 0.990 | — |
| remex 2-bit d=1024 -> late remex 2-bit | 94.3 KB | 0.9143 | +0.0009 [-0.0226, +0.0254] | 7/6 | 0.980 | 0.990 | — |
| remax k=1 d=1024 (asym) -> late remex 2-bit | 94.2 KB | 0.9143 | +0.0009 [-0.0226, +0.0254] | 7/6 | 0.980 | 0.990 | — |
| remex 2-bit d=256 -> late remex 2-bit | 94.1 KB | 0.9043 | -0.0091 [-0.0413, +0.0206] | 7/7 | 0.970 | 0.980 | — |
| fp32 d=1024 -> late remex 1-bit | 51.0 KB | 0.9295 | +0.0161 [-0.0062, +0.0396] | 7/3 | 0.980 | 0.990 | — |
| fp32 d=1024 -> late remax k=1 (asym) | 51.0 KB | 0.9043 | -0.0091 [-0.0339, +0.0148] | 7/9 | 0.980 | 0.990 | — |
| remex 2-bit d=1024 -> late remex 1-bit | 47.3 KB | 0.9295 | +0.0161 [-0.0062, +0.0396] | 7/3 | 0.980 | 0.990 | — |
| remex 2-bit d=1024 -> late remax k=1 (asym) | 47.3 KB | 0.9043 | -0.0091 [-0.0339, +0.0148] | 7/9 | 0.980 | 0.990 | — |
| remax k=1 d=1024 (asym) -> late remex 1-bit | 47.2 KB | 0.9295 | +0.0161 [-0.0062, +0.0396] | 7/3 | 0.980 | 0.990 | — |
| remax k=1 d=1024 (asym) -> late remax k=1 (asym) | 47.2 KB | 0.9043 | -0.0091 [-0.0339, +0.0148] | 7/9 | 0.980 | 0.990 | — |
| remex 2-bit d=256 -> late remex 1-bit | 47.1 KB | 0.9195 | +0.0061 [-0.0250, +0.0357] | 7/4 | 0.970 | 0.980 | — |
| remex 2-bit d=256 -> late remax k=1 (asym) | 47.1 KB | 0.8943 | -0.0191 [-0.0513, +0.0107] | 7/10 | 0.970 | 0.980 | — |

`*` = 95% paired-bootstrap CI on ΔnDCG@10 excludes zero (100 queries, 5,000 resamples). Wins/losses count queries whose nDCG@10 moved vs the reference; ties omitted.
<!-- tables:shift:end -->

## Findings

**1. The late-interaction head at 1 bit per dimension beats the dense head at
fp32, at comparable bytes.** Late remex 1-bit stores 5.1 KB/doc and scores
nDCG@10 0.7070; dense fp32 stores 4.1 KB/doc and scores 0.5527. The paired
difference is +0.154 [+0.116, +0.192], 105 queries up, 27 down. Pooling to
half the tokens and then coding at 1 bit (2.5 KB/doc, 60% of the dense fp32
budget) still holds +0.124 [+0.085, +0.162]. For NeoMME the dense head is not
the way to save index space; a quantized token index is smaller and far better.
The card's BEIR-15 gap (late 0.488 vs dense 0.306) shows on SciFact as 0.720
vs 0.553 at fp32.

**2. On the token head, quantizing tokens costs less than pooling them per
byte saved.** `HierarchicalTokenPooling` at factor 2 halves storage (82 KB/doc)
for −0.019 [−0.029, −0.009] nDCG@10; factor 4 (41 KB/doc) costs −0.026
[−0.042, −0.012]. remex 2-bit cuts storage 16× (10.2 KB/doc) for −0.013
[−0.025, −0.002], and remex 1-bit 32× (5.1 KB/doc) for −0.013 [−0.028,
+0.002]. Against pool2 at 8× fewer bytes, remex 2-bit is +0.006 [−0.005,
+0.016]; against pool4 at 8× fewer bytes, remex 1-bit is +0.013 [−0.004,
+0.030]. remex 4-bit (8×, 20 KB/doc) is within −0.005 [−0.011, +0.001] of
fp32 with 0.94 top-10 overlap. Pooling has a use once the token count itself is
the cost (MaxSim compute scales with tokens; bits do not change the GEMM shape
in this decode-then-dot implementation) — measure compute separately if that is
the constraint.

**3. On the dense head, 4-bit on a truncated Matryoshka dimension wins every
byte budget from 64 B to 512 B.** At 128 B: remex 4-bit d=256 scores 0.5300
against remex 1-bit d=1024 0.5059 (+0.024 [+0.001, +0.048]), remax k=1
d=1024 asym 0.5120 (+0.018 [−0.001, +0.038]) and ST binary 0.5055 (+0.025
[+0.005, +0.045]). At 256 B: remex 4-bit d=512 0.5490 against remex 2-bit
d=1024 0.5305 (+0.019 [+0.000, +0.037]). At 512 B: remex 4-bit d=1024 0.5509
is indistinguishable from fp32 d=1024 (−0.002 [−0.012, +0.008], 26/26) and
beats fp32 d=128 at the same 512 B by +0.036 [+0.014, +0.057]. At 64 B the
ordering is the same (4-bit d=128 0.5021 > 2-bit d=256 0.4939 > 1-bit d=512
0.4819) with CIs crossing zero.

This runs against the "quantize wide rather than truncate narrow" entry in
`METHODS.md`. That rule
compared 2-bit wide against *fp32* narrow and found the bits cheaper than the
dims. Here the model was trained at exactly the truncation dims, so truncation
is cheap (fp32 1024→256 costs 0.015 nDCG) while dropping below 4 bits is not
(1024-d 4→2→1 bits costs 0.020, then 0.025 more). The frontier on an
MRL-trained model is 4 bits at the dimension the budget allows, and 1-bit at
full width is the wrong end of it. Both findings are consistent: bits below 4
are expensive on both models; truncation was expensive on the older ones and is
cheap on this one.

**4. ST int8 is lossless and twice the size of an equivalent remex 4-bit.**
ST int8 d=1024 (1,024 B) matches fp32 to −0.0002 with 0.996 top-10 overlap.
remex 4-bit d=1024 (512 B) is −0.002 from fp32. Both are free; one is half the
bytes.

**5. remex 1-bit and remax k=1 (asym) are the same code up to the rotation
draw, and the gap between them is the seed-noise floor of this bench.** On
unit-norm inputs, remex at 1 bit decodes to ±c in rotated space and remax
asym scores the float query against ±1 in rotated space — identical rankings
for an identical rotation. The two use independently seeded RHTs, and they
land 0.006 apart on the dense head (remax ahead, 47/47) and 0.012 apart on
the token head (remex ahead, 40/34), both CIs spanning zero. Any dense-head
difference under about ±0.02 nDCG@10 on 300 SciFact queries is inside that
floor; the 128 B finding in (3) clears it against 1-bit and ST binary and
does not clear it against remax asym.

**6. The dense→late pipeline (the paper's own large-corpus recipe, section 5.2,
and the thread's) caps below a full late-interaction scan, and quantizing its
rerank stage is free.** Dense fp32 top-100 → late fp32 rerank
scores 0.6854, −0.034 [−0.056, −0.015] under a full late scan (0.7198),
because dense R@100 is 0.864 against late's 0.943: a relevant document the
dense head ranks past 100 does not reach the reranker. Swapping the rerank
stage to remex 1-bit costs −0.005 [−0.018, +0.007] on top of that, and remex
2-bit −0.004 [−0.013, +0.005]. A full scan over remex 1-bit tokens (5.1
KB/doc, 0.7070) beats the all-fp32 pipeline (168 KB/doc, 0.6854) by +0.022
[−0.001, +0.045]. On a corpus this size the pipeline buys nothing; at a scale
where a token scan is unaffordable, widen the candidate set past 100 before
spending bytes on the reranker's precision.

**7. Asymmetric scoring holds up on the token head too.** Binarising the query
(remax sym) costs −0.001 on the token head and −0.027 on the dense head at
k=1 d=1024. Keeping the query in float is free at index time and is the default
in `neomme_quant.py`.

**8. On page images the 1-bit token index costs nothing measurable.** Late
fp32 0.5188; remex 1-bit 0.5146, −0.004 [−0.015, +0.007], at 46.7 KB/page
against 1.5 MB (32×). remex 4-bit −0.003, 2-bit −0.004. Against the dense
head at fp32 (0.3847, 4 KB), the 1-bit token index is +0.130 [+0.103, +0.158],
145 queries up, 29 down. On SciFact the same comparison was +0.154; the gap
between heads survives the switch from text to pixels, at 11× the bytes for the
token side rather than 1.25×.

**9. Pooling is also nearly free on pages, so the two compose.** pool2 costs
−0.004 [−0.010, +0.002] and pool4 −0.007 [−0.015, +0.000] on DocVQA, against
−0.019 and −0.026 on SciFact. Ward clustering of ~2,900 patch vectors finds
many near-duplicates (white margins, repeated background) that 320 text tokens
do not have. pool2 + 1-bit is 23.4 KB/page, 64× under fp32, at −0.004 [−0.015,
+0.008]. This changes the SciFact verdict on pooling (finding 2): on text it
costs more than quantization per byte; on pages it is free and stacks.

**10. The dense frontier is flat from 128 B to 512 B on DocVQA.** fp32 →
remex 4-bit d=512 (256 B) −0.003, 2-bit d=1024 (256 B) −0.005, 4-bit d=1024
(512 B) −0.007 [−0.013, −0.000]; at 128 B, 4-bit d=256, 2-bit d=512 and ST
binary sit within 0.006 of each other, all CIs spanning zero. What holds from
SciFact: remex 1-bit at full width loses (−0.025 [−0.040, −0.011]) and so does
fp32 truncated to 128 dims (−0.024 [−0.036, −0.011]) — the two ways to spend
512 B or 128 B badly are the same on both corpora. What does not transfer: the
strict 4-bit-narrow-beats-2-bit-wide ordering; here the two tie. Binarising the
query still costs −0.035 on the dense head and −0.002 on the token head.

**11. The pipeline cap is smaller but still there.** Dense fp32 top-100 → late
fp32 rerank 0.5082, −0.011 [−0.020, −0.002] under the full late scan, with dense
R@100 0.718 against late 0.814. A 1-bit reranker costs a further −0.005 (CI
spans zero). One oddity: candidates from remex 2-bit d=1024 rerank to 0.5145,
+0.006 [+0.001, +0.012] over fp32 candidates, because that arm's R@100 came out
0.742 against 0.718 — a quantized candidate stage that retrieves *more* of the
relevant pages into the top 100 on this corpus. Treat as noise until a second
corpus repeats it.

**12. ShiftProject repeats DocVQA on the token head and brings the SciFact
dense ordering back.** Token head: fp32 0.9234; remex 1-bit 0.9395, +0.016
[−0.006, +0.040] at 48.2 KB/page (32×); 4-bit +0.002, 2-bit +0.001; pool2
+0.007 [−0.007, +0.022], pool4 −0.001; pool2 + 1-bit +0.003 [−0.019, +0.027]
at 24.1 KB (64×). Nothing in the token ladder loses; 1-bit tokens are +0.182
[+0.122, +0.244] over dense fp32 (35 queries up, 3 down). remax k=1 asym
(−0.009) trails remex 1-bit by 0.025 [−0.001, +0.054] — the same-code-different-
rotation pair from finding 5, at the edge of its interval on 100 queries.
Dense head: fp32 0.7581; 4-bit d=1024 −0.004 and 2-bit d=1024 −0.003 (ties);
1-bit d=1024 −0.062 [−0.094, −0.033]; fp32 d=128 −0.035 [−0.067, −0.006];
fp32 d=256 −0.028. 4-bit d=256 beats 1-bit d=1024 by +0.040 [+0.003, +0.077] —
the SciFact ordering (finding 3), which DocVQA had as a tie. Pipeline: dense
R@100 is 0.99 here, so dense top-100 → late rerank is −0.010 under the full
scan (one query), and the finding-11 oddity (quantized candidates beating fp32
candidates) does not repeat: remex 2-bit d=1024 candidates rerank to exactly
the fp32-candidate score.

Across the three corpora, then: (a) the token head at 1 bit per coordinate is
within noise of fp32 on all three (−0.013, −0.004, +0.016) at 32× smaller;
(b) pooling costs on text (−0.019) and is free on pages (−0.004, +0.007);
(c) on the dense head, 4-bit at full width matches fp32 everywhere (−0.002,
−0.007, −0.004), 1-bit at full width and fp32 at 128 dims lose everywhere, and
4-bit-narrow beats 1-bit-wide on two of three with a tie on the third.

## Cost

- Encode: 39.3 min for 5,183 docs + 300 queries at float32 on 4 vCPU
  (2.24 docs/s over the run; length-sorted batches of 16, longest first).
  1.66M document tokens; `mv_tokens.npy` is 424 MB at float16.
- ShiftProject: 83 min wall for 1,000 pages across three launches (nohup died at 114, Monitor at 380, `run_in_background` finished); bench 26 min after an OOM-killed first attempt (exit 137: remex codec caches held 1.5 GB per arm on 3M tokens; `_release()` in `bench.py` fixes it) and a resume fix (fidelity reference rebuilt from fp32 on resume).
- DocVQA: 50.2 min for 500 pages at 0.17 pages/s (batch 2, no length sort; a
  2,048-px page is ~3,000 patches); bench 18 min, pooling 107 s / 92 s.
- Bench (SciFact): dense arms under 1 s each; each late arm 25–80 s (300 queries × one
  GEMM against 1.66M × 128 decoded tokens); pooling 15 s per factor over the
  whole corpus; 14 min total.
- Peak memory ≈ 3 GB (decoded token matrix at float32 plus the remax k=2
  ±1 matrix).

## Caveats

- Three corpora (SciFact text; DocVQA and ShiftProject page images), one seed
  per codec, 300 / 500 / 100 queries. ShiftProject's queries are generated from
  the pages and its fp32 token index is at R@10 0.99, so it can only confirm
  "no loss"; it cannot rank codecs finely. The 128 B dense finding clears the seed floor against
  1-bit and ST binary and not against remax asym.
- Both ViDoRe tasks have one relevant page per query, so per-query nDCG@10 is
  coarse (1/log2(rank+1) or 0).
- nDCG@10 against SciFact's binary qrels only; no throughput or compute
  measurement. The MaxSim implementation here decodes to float32 and dots, so
  it saves storage and not FLOPs; a popcount kernel over remax codes is the
  compute-side answer and is not benchmarked.
- The 260M model only. The 800M adds a 1,792-d Matryoshka step and may shift
  the dense frontier.
- `pytrec_eval` was not used; `recheck.py` recomputes the fp32 dense nDCG@10
  through a second code path (Python `sorted`, explicit `math.log2`) and runs a
  shuffled-qrels negative control.

## Files

- `neomme_quant.py`: `DenseIndex`, `MultiVectorIndex`, `pool_tokens`, metrics; `python3 neomme_quant.py` runs the self-test.
- `encode.py`: resumable two-head encode; writes `data/scifact_enc/` (gitignored, 450 MB) and `meta.json`.
- `bench.py`: all arms; resumable; writes `results.json` + `results_perquery.npz`.
- `report.py`: renders the tables above from those two files.
- `recheck.py`: prose-vs-artifact fixture (tables match, independent nDCG recompute, negative control).
- `smoke.py`: builds a partial dataset from checkpoints for exercising `bench.py` mid-encode; not a result.
- `ERRORS.md`: what went wrong and which way it pushed.
