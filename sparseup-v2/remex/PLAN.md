# PLAN: SPARSEUP-v2 value compression ("remex treatment" for a sparse vector)

Written 2026-11-25, before any step-2 arm was scored. Predictions are scored in
REPORT.md; none are edited after scoring.

## Question

How few bytes per document can SPARSEUP-v2's sparse document vectors take before
retrieval quality drops, and at a fixed byte budget is it better to keep fewer
terms at more bits per value or more terms at fewer bits?

remex compresses a dense vector with three parts: the exact fp32 norm, a random
rotation, and a Lloyd-Max scalar codebook. A rotation would turn a vector with
about 290 nonzeros out of 50,370 dimensions into a dense one, so it does not apply.
This study applies the other two parts to the nonzero term values: a codebook
(uniform, Lloyd-Max, or Lloyd-Max in the log domain) and an optional exact
per-document scale (fp16 document max, or the remex-style exact L2 norm with the
decoded direction renormalised to it). Combined with per-document top-k pruning
(the model was trained for this with Matryoshka top-k levels 64/128/256).

## Fixture

SciFact, NFCorpus, FiQA-2018, ArguAna test sets from `mteb/*`. Docs `title + " " + text`,
`encode_document` ([D]), max 512 tokens; queries `encode_query` ([Q]) unpruned.
bf16 weights (the card's precision), fp32 output. Exact dot-product scoring over
the whole corpus, nDCG@10 (trec_eval linear gain) and R@100; BEIR's
`ignore_identical_ids` (drops the query's own id from its ranking; affects ArguAna).
Paired bootstrap over queries, 1,000 resamples, vs fp32 at full k.

## Observation made before predicting (from the 400-doc Kaggle smoke run)

Document values span 0.118 to 3.88, median 1.06; mean 286 nonzeros. Only 168
distinct values occur in 400 documents: the activation is computed in bf16, so
`logit - 18` lives on a 0.125 grid near the threshold and values are `log1p` of
a few hundred discrete points. Top-128 holds 71% of a document's L1 mass and
top-256 holds 97%.

## Predictions

| # | prediction |
|---|---|
| P1 | fp16 values at full k give exactly the fp32 rankings (delta 0.0000 on every dataset): every bf16 value is exactly representable in fp16. |
| P2 | 8-bit uniform (no scale) at full k: macro delta nDCG@10 within +/-0.002, CI spanning 0. |
| P3 | Anserini-style 8-bit (round(w/max*255), zeros dropped) at full k: within +/-0.002; drops no terms because the minimum value 0.118 is far above max/510. |
| P4 | 4-bit Lloyd-Max (any scale) at full k: macro delta >= -0.005. |
| P5 | 2-bit: the best Lloyd/log-Lloyd arm with a per-doc scale loses <= 0.015 macro; 2-bit uniform without scale loses > 0.02. |
| P6 | 1-bit binary presence (no scale) at full k loses between 0.02 and 0.08 macro (the v1 tag probe lost 0.021 micro-AP; ranking is more sensitive). |
| P7 | fp32 top-k pruning, macro delta: k=256 within 0.002; k=128 -0.005 to -0.015; k=64 -0.02 to -0.04; k=32 -0.05 to -0.08; k=16 below -0.10. |
| P8 | 16-bit term ids cost 2 B per term and dominate the bytes, so on the macro frontier above 100 B/doc every point uses 3 or 4 bits per value; no 8-bit or fp16 arm is on the frontier except the lossless top end, and no 1-bit arm is on it (60% confidence on the 1-bit half). |
| P9 | Delta-varint ids cost 1.5-1.8 B per term at full k, rising toward 2 B at k=16. |
| P10 | 8-bit queries (per-query uniform) at full k: within +/-0.001 of fp32 queries. |
| P11 | At 2 bits, a per-doc scale (docmax or l2) beats no scale on Lloyd by >= 0.003 macro; docmax and l2 are within 0.003 of each other. |
| P12 | Sentence Transformers `max_active_dims=k` gives the same vectors as top-k truncation of the full vector (exact, up to ties). |
| P13 | fp32 full-k baseline within 1.0 nDCG@10 point of the paper's Table 1 (SciFact 75.1, NFCorpus 38.9, FiQA 50.9, ArguAna 56.1); the paper used Seismic (approximate) and did not state the document length. |

Step 3 (model-weight fake quantization, SciFact + NFCorpus, fp32 compute):

| # | prediction |
|---|---|
| P14 | int8 per-row weights: mean cosine to fp32 vectors >= 0.995, top-32 overlap >= 0.95, nDCG@10 delta within +/-0.005. |
| P15 | int4 group-32 weights: cosine 0.95-0.99, top-32 overlap 0.75-0.90, nDCG@10 delta -0.005 to -0.03. |
