# Jev 256-tag document encoder: results

## Answer

Can one Jev call asking 256 "The document is about <tag>." Nouls produce a useful named document
vector? For topic labelling, yes; for retrieval, no. On 600 arXiv abstracts with 24 cross-listed
categories, mapping each category to its tags gives micro-F1 0.636 [0.616, 0.659] with no labels.
Given labels, a logistic probe on the 256 raw probabilities beats a gemini-embedding-2 probe at 50
labels (0.566 vs 0.517) and trails it at 200 (0.671 vs 0.714) and 500 (0.697 vs 0.778). On
SciFact the tags alone score nDCG@10 0.18 to 0.34, and as a third RRF leg they cost 0.087 [0.059,
0.115] against BM25 + dense. A 3-bit code (sign plus logit-spaced confidence buckets) matches the
floats on retrieval (−0.011 [−0.031, +0.009]); one bit keeps zero-shot F1 exactly and loses 0.18
micro-AP. Each doc gets 7.5 active tags; 7,449 calls ran at p50 0.59 s and ~$0.00017; reruns
flip 5 of 7,680 values across 0.5.

What would change this: a taxonomy finer than the corpus (SciFact uses 58 of 256 tags), or labels
defined the way the tags are phrased instead of by arXiv cross-listing habits.

## Findings

1. **Zero-shot tags sit between a 50-label and a 200-label dense probe.** micro-F1 0.636 vs dense
   0.517 (@50), 0.714 (@200), 0.778 (@500), unscaled probes; micro-AP 0.610 vs 0.519 / 0.795 /
   0.855. Per-label oracle thresholds lift zero-shot to 0.705. (`results/step3_classify.json`,
   `results/probe_codes.json`)
2. **With a few labels, the tag vector is the better feature set.** A probe on the 256 raw
   probabilities: 0.566 [0.547, 0.585] @50 vs dense 0.517 [0.497, 0.536]; 0.671 vs 0.714 @200;
   0.697 vs 0.778 @500. It passes zero-shot from 200 labels (+0.035 [0.018, 0.054]).
   (`results/probe_codes.json`)
3. **Standardizing Jev probabilities before a probe costs up to 0.22 micro-F1.** StandardScaler
   gives a rare tag wandering between 0.01 and 0.03 unit variance. Scaled vs raw: 0.347 vs 0.566
   @50, 0.620 vs 0.697 @500. Dense embeddings lose only at @50 (0.471 vs 0.517). Quantizing to
   1 to 3 bits recovers part of it (0.625 to 0.667 @500). (Round 2, `results/probe_codes.json`)
4. **The tag vector hurts every retrieval fusion on SciFact.** RRF(BM25, dense, Jev) 0.688 at best
   (Bernoulli score) vs RRF(BM25, dense) 0.774; RRF(BM25, Jev) is 0.072 under BM25 alone. Bernoulli
   log-likelihood is the best Jev score (0.337 alone), dot 0.184, Hamming on 0.5 bits 0.102. "The
   query asks about" changes nothing (within 0.015). SciFact spreads over 58 tags that fire on
   ≥ 1% of docs; "scientific study" fires on 99%. (`results/step4_retrieve.json`)
5. **3 bits per tag keep the ranking signal; 1 bit keeps only the labels.** Zero-shot F1 at 0.5 is
   identical under every code. micro-AP: float 0.610, 3-bit logit 0.567, 2-bit logit 0.527, 1-bit
   0.433. SciFact Bernoulli alone: float 0.337, 3-bit logit 0.325, 2-bit logit 0.276, 1-bit
   0.137. Logit-spaced cuts beat uniform cuts at the same bits on AP and on Bernoulli retrieval (3-bit
   uniform 0.283); on the dot score uniform cuts are 0.01 to 0.02 better.
   96 bytes per doc at 3 bits. (Round 2, `results/quant_eval.json`)
6. **Where zero-shot fails, the tag and the label mean different things.** cs.LG: recall 0.98,
   precision 0.29. "Machine learning" fires on any paper that uses ML; arXiv cross-lists only some
   of them. Distinct fields map cleanly: hep 0.94, astro-ph 0.89, cs.RO 0.88.
   (`results/step3_classify.json` per_label_zs)
7. **"is about" is the right phrasing of the three.** On 50 arXiv papers "mentions" raises active
   tags from 5.7 to 8.3 per doc and costs 0.088 [0.038, 0.138] micro-F1; the additions are
   off-topic ("statistics and data" +0.54, "architecture" read as model architecture) and
   gold-label recall does not rise (0.887 → 0.859). "is substantially about" ties "is about"
   (−0.014 [−0.039, +0.007]). (`results/step5_phrasing.json`)
8. **The taxonomy is sparse and mostly live.** 7.5 [7.2, 7.9] active tags per doc on 199 mixed
   news and Usenet docs, none with zero, none always-on ("news report" 50%). Across all 6,719
   encoded docs 27 tags never fire, all topics absent from news, Usenet and science (cooking,
   golf, esports, venture capital, ...). (`results/step2_fit.json`)
9. **Jev is near-deterministic, not bit-deterministic.** 30 docs re-encoded ~2.5 h later, gateway
   cache skipped: 84.7% of values identical, mean |Δ| 0.0019, max 0.10, 5 of 7,680 values flip
   across 0.5, all from within 0.05 of it. (`results/step6_determinism.json`)

## Method

**Encoder.** `jev.py`: one POST per text to Workers AI `typesafe/jev` through the Cloudflare AI
Gateway (TypeSafe key stored gateway-side; `cf-aig-skip-cache: true` on every call), `state =
{"document": text}`, 256 Noul questions `tNNN` = flattened `tags.txt` index. jev-1.13.0 answered
every call. Vectors are 256 floats in tag order, cached per (set, phrasing) in `vectors/*.parquet`.
The gateway is configured at 50 requests per 60 s (fixed window), so the client paces at 48/min;
the full run took ~2.5 h wall clock.

**Fixtures** (`data.py`, seed 98). Step 2: 100 AG News test (25 per class) + 100 20 Newsgroups
test (5 per group, headers/footers/quotes removed, ≥200 chars), truncated at 4,000 chars. Steps 3
and 5: the 60 newest arXiv papers for each of 24 categories (title + abstract), deduplicated to
1,337, labels = every one of the 24 categories the paper is listed in (29% multi-label), 600 test
and 737 train. Each label maps to one or two tags (`data.LABELS`); its score is the max over them.
Step 4: BEIR SciFact, 5,183 docs, 300 test queries.

**Arms.** Dense: gemini-embedding-2 (3,072-d, task type CLASSIFICATION / RETRIEVAL_*). Probes: one
logistic regression per label, `class_weight=balanced`, C by 3-fold CV on the training sample,
5 random training samples per size. Round 1 standardized features and searched C in
{0.01, 0.1, 1, 10}; Round 2 added raw-feature probes with C up to 1000. The headline probe numbers
are the raw-feature ones for both Jev and dense. BM25:
`bm25s` with English stopwords. RRF k = 60 over full rankings. Jev retrieval scores: dot of
probabilities; Bernoulli log-likelihood Σ q log d + (1−q) log(1−d) with d clipped to [0.01, 0.99];
negative Hamming distance on 0.5 bits.

**Metrics.** micro/macro-F1 at 0.5 and micro/macro-AP (AP because a fixed threshold ranks arms
partly by calibration — `sparseup-tag-probe` ERRORS #5); nDCG@10. 95% CIs from 2,000 bootstrap
resamples over test documents or queries; differences are paired on the same resamples. Probe arms
average the 5 training samples inside each resample. No rate here comes from a judge.

**Quantization** (`quant.py`). Sign bit at 0.5 plus confidence |2p−1| split into 2 (2-bit) or 4
(3-bit) buckets per side, cut uniformly in confidence or evenly in |logit p|. Each code
dequantizes to its bucket's mean p on the mixed corpus, which is disjoint from the arXiv and
SciFact eval sets. Reported against the 2-decimal floats Jev returns (100 distinct values).

**Prior art.** QA-Emb (Benara et al. 2024, arXiv 2405.16714) builds embeddings from binary LLM
answers to task-generated yes/no questions and evaluates on fMRI encoding. This run uses a fixed
general taxonomy, calibrated Noul probabilities from one call, and evaluates zero-shot labelling
and retrieval fusion.

**Cost.** 7,449 successful calls, 1 WAF block (a 20 Newsgroups post), mean 4,127 input tokens
(~3.8k of it the 256 question strings). TypeSafe bills input only, $0.042/Mtok: ~$1.30 for the
run. The 4,868 "output tokens" per call are the fixed size of the 256-entry answer map and are
not billed.

## Log

### Round 1 (2026-09-23): six steps from the issue

Asked: oaustegard/experiments#98, run the README's six steps from code. Blocker check: no
TypeSafe key in the container, but the CF AI Gateway holds one; direct Workers AI without the
gateway returns 402 "Insufficient balance".

Ran: `data.py` → `encode_all.py` (background, resumable) → `fit.py`, `classify.py`,
`retrieve.py`, `ablate.py`, `quant_eval.py`. The first encode launch died on a missing `pyarrow`
after step 2's 200 docs; the second backed off exponentially into the gateway's 50/min limit and
ran at ~0.4 docs/s until a client-side pacer replaced the backoff.

Step 3 (600 arXiv test papers):

| arm | micro-F1 | macro-F1 | micro-AP | macro-AP |
|---|---|---|---|---|
| jev_zs@0.5 | 0.636 [0.616, 0.659] | 0.693 [0.665, 0.713] | 0.610 [0.580, 0.644] | 0.757 [0.738, 0.786] |
| jev_zs@oracle-glob (0.6) | 0.639 [0.618, 0.662] | 0.693 [0.665, 0.714] | 0.610 | 0.757 |
| jev_zs@oracle-lab | 0.705 [0.686, 0.725] | 0.747 [0.724, 0.765] | 0.610 | 0.757 |
| dense+probe@50 | 0.471 [0.449, 0.494] | 0.419 [0.395, 0.434] | 0.569 [0.545, 0.593] | 0.652 [0.638, 0.676] |
| dense+probe@200 | 0.720 [0.702, 0.739] | 0.738 [0.717, 0.753] | 0.784 [0.764, 0.806] | 0.813 [0.798, 0.834] |
| dense+probe@500 | 0.779 [0.760, 0.797] | 0.803 [0.781, 0.818] | 0.859 [0.839, 0.878] | 0.868 [0.852, 0.887] |
| jev256+probe@50 | 0.347 [0.331, 0.363] | 0.333 [0.315, 0.346] | 0.232 [0.219, 0.248] | 0.301 [0.298, 0.334] |
| jev256+probe@200 | 0.557 [0.536, 0.579] | 0.585 [0.561, 0.604] | 0.418 [0.389, 0.455] | 0.482 [0.467, 0.536] |
| jev256+probe@500 | 0.620 [0.596, 0.643] | 0.651 [0.623, 0.673] | 0.533 [0.483, 0.590] | 0.584 [0.552, 0.654] |

Paired zero-shot minus dense micro-F1: +0.165 [0.138, 0.192] vs @50, −0.084 [−0.106, −0.061] vs
@200, −0.143 [−0.165, −0.119] vs @500. CV picked C = 0.01 in most dense@500 samples and swung
between 0.01 and 10 at @50. Seed SD of micro-F1 was 0.028 at dense@50 and ≤ 0.007 from @200 up.

Step 4 (SciFact, 300 queries, nDCG@10):

| run | nDCG@10 | vs RRF(BM25, dense) |
|---|---|---|
| BM25 | 0.662 [0.616, 0.706] | −0.113 |
| dense (gemini-embedding-2) | 0.898 [0.872, 0.923] | +0.123 |
| RRF(BM25, dense) | 0.774 [0.737, 0.813] | — |
| Jev alone: about/dot, bernoulli, hamming | 0.184 / 0.337 / 0.102 | |
| Jev alone: query/dot, bernoulli, hamming | 0.184 / 0.322 / 0.089 | |
| RRF(BM25, dense, Jev about/bernoulli) | 0.688 [0.648, 0.727] | −0.087 [−0.115, −0.059] |
| RRF(BM25, dense, Jev about/dot) | 0.679 | −0.095 |
| RRF(BM25, dense, Jev about/hamming) | 0.622 | −0.152 |

BM25 reproduces BEIR's published 0.665. Dense at 0.898 sits above published SciFact numbers for
other embedders; gemini-embedding-2 may have seen SciFact in training, which would make the
dense-relative deltas pessimistic for any third leg. The BM25-only comparison does not depend on
that and also loses (−0.072 [−0.110, −0.036] for the best Jev score). SciFact docs activate 7.5
tags each, but only 58 tags fire on ≥ 1% of the corpus. "scientific study" fires on 99% of docs,
"regulation" on 38% (read as gene regulation). 3,759 distinct 1-bit codes cover 5,183 docs, and
the largest bucket holds 61. At this granularity the tags separate biology from medicine, not one
claim's evidence from another's.

Prediction (issue): standalone would lose, the question was whether the leg adds anything. It
does not, under any of the three scores or two query phrasings.

Step 5 (50 arXiv test papers):

| phrasing | active/doc | gold-label recall | micro-F1 | macro-F1 |
|---|---|---|---|---|
| is about | 5.70 | 0.887 | 0.704 [0.626, 0.774] | 0.814 [0.709, 0.877] |
| mentions | 8.30 | 0.859 | 0.616 [0.548, 0.685] | 0.709 [0.614, 0.793] |
| is substantially about | 5.76 | 0.845 | 0.690 [0.610, 0.764] | 0.806 [0.683, 0.874] |

The claude.ai probe had flagged "is about" as gating on primary topic (doping → pharmacology
0.14), and "mentions" was the candidate fix. On arXiv categories it adds tags without adding
gold labels. A category is a primary-topic label by construction, so the primary-topic phrasing
fits it best; a corpus labelled for secondary topics could rank the phrasings differently.

Step 6: 30 mixed docs re-encoded after the SciFact corpus (~2.5 h after the first pass).
1,172 of 7,680 values changed, max |Δ| 0.10; 5 flips across 0.5 (pricing 0.47→0.52, United
Kingdom 0.53→0.49, announcement 0.45→0.50, controversy 0.47→0.52, diplomacy 0.51→0.49).

### Round 2 (2026-09-23): 2- and 3-bit codes and probe scaling

Asked (Oskar, mid-run): test 2-bit and 3-bit versions, using confidence buckets for the extra
bits. Also asked why 4,868 output tokens: TypeSafe bills input only, and output_tokens is the
fixed size of the answer map (~19 tokens per Noul), identical on every call.

Ran `quant_eval.py` on the cached floats (no new Jev calls). Codes: sign bit plus confidence
|2p−1| in 2 or 4 buckets per side, cut uniformly or evenly in |logit p|, dequantized to bucket
means fit on the mixed corpus. Jev's 2-decimal floats put 87.5% of arXiv values at ≤ 0.02, so
uniform 2-bit cuts leave 96% of values in one bucket; logit cuts spread them (93.6% / 3.5% / 1.4%
/ 1.5% on SciFact).

| code | bytes/doc | zs micro-AP | Δ vs float | SciFact Bernoulli alone | Δ vs float |
|---|---|---|---|---|---|
| float (2-dec.) | 256 | 0.610 | — | 0.337 | — |
| 1-bit sign | 32 | 0.433 | −0.177 [−0.202, −0.153] | 0.137 | −0.199 [−0.237, −0.162] |
| 2-bit uniform | 64 | 0.495 | −0.115 [−0.136, −0.092] | 0.206 | −0.130 [−0.163, −0.099] |
| 2-bit logit | 64 | 0.527 | −0.083 [−0.101, −0.065] | 0.276 | −0.060 [−0.092, −0.028] |
| 3-bit uniform | 96 | 0.531 | −0.080 [−0.099, −0.061] | 0.283 | −0.053 [−0.078, −0.029] |
| 3-bit logit | 96 | 0.567 | −0.043 [−0.057, −0.031] | 0.325 | −0.011 [−0.031, +0.009] |

Zero-shot micro-F1 at 0.5 is 0.636 under every code. The RRF deltas move with the standalone
score: 3-bit logit as a third leg is −0.085 against RRF(BM25, dense), float −0.087.

The single-seed probe column in that run inverted Round 1's Finding: the probe on 1-bit codes
had micro-AP 0.694 against 0.514 on the floats. `probe_codes.py` tested the explanation that
StandardScaler inflates near-constant rare tags, with 5 seeds and raw-feature controls:

| features | @50 micro-F1 | @200 | @500 | @500 micro-AP |
|---|---|---|---|---|
| Jev float, standardized (Round 1) | 0.347 | 0.557 | 0.620 | 0.533 |
| Jev float, raw | 0.566 [0.547, 0.585] | 0.671 [0.655, 0.688] | 0.697 [0.678, 0.717] | 0.776 |
| Jev 1-bit, standardized | 0.501 | 0.615 | 0.625 | 0.693 |
| Jev 2-bit logit, standardized | 0.447 | 0.637 | 0.667 | 0.706 |
| Jev 3-bit logit, standardized | 0.400 | 0.603 | 0.648 | 0.624 |
| dense, standardized (Round 1) | 0.471 | 0.720 | 0.779 | 0.859 |
| dense, raw | 0.517 [0.497, 0.536] | 0.714 [0.697, 0.732] | 0.778 [0.758, 0.796] | 0.855 |

Round 1's "a probe over the 256 tags never beats the hand mapping" was the scaler, and is
withdrawn. Raw-feature probes are the fair comparison and now carry Findings 1 and 2.
