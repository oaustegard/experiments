# neomme-remex-quant — remex and remax quantization of both NeoMME-260M-Retriever heads

**Started:** 2026-09-07 · **Status:** done · **Runtime:** 39 min encode + 14 min bench on 4 vCPU

## Question

Tom Aarsen's Bluesky thread (2026-09-07) announced H Company's NeoMME: 260M and
800M multimodal encoders with a dense head (1024-d, Matryoshka 128/256/512/1024)
and a late-interaction head (128-d per token, MeanMaxSim), both from one forward
pass. Oskar asked whether H Company ships a quantized-vector version and, if not,
to build remex/remax quantization for it.

## Prior-art check

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

**6. The dense→late pipeline caps below a full late-interaction scan, and
quantizing its rerank stage is free.** Dense fp32 top-100 → late fp32 rerank
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

## Cost

- Encode: 39.3 min for 5,183 docs + 300 queries at float32 on 4 vCPU
  (2.24 docs/s over the run; length-sorted batches of 16, longest first).
  1.66M document tokens; `mv_tokens.npy` is 424 MB at float16.
- Bench: dense arms under 1 s each; each late arm 25–80 s (300 queries × one
  GEMM against 1.66M × 128 decoded tokens); pooling 15 s per factor over the
  whole corpus; 14 min total.
- Peak memory ≈ 3 GB (decoded token matrix at float32 plus the remax k=2
  ±1 matrix).

## Caveats

- One corpus (SciFact, scientific claims against abstracts), one seed per
  codec, 300 queries. The 128 B dense finding clears the seed floor against
  1-bit and ST binary and not against remax asym.
- Text only. NeoMME's headline use is page images, whose token count per
  document runs into the thousands and where pooling has a different
  cost/benefit; nothing here measures that.
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
