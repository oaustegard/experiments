# RESULTS: quantizing and pruning EmbeddingGemma 2's token table

Run 2026-10-20 in a 4-vCPU CCotw container, CPU only, about 25 minutes of evaluation.
Everything below is from `results/arms.json` (per-query cosines and overlaps in
`results/perquery/`), produced by `run/arms.py`.

## Answer

Can EmbeddingGemma 2's (EG2) token table be shrunk without moving query vectors off the
corpus? Yes, down to 4 bits per value. Below that it fails. Int8 with one scale per row
passes easily: top-10 overlap 0.994 and 1st-percentile cosine 0.99998 against fp32 EG2.
Two 4-bit layouts pass narrowly: asymmetric groups of 32 at 0.959 / 0.9977, and remex
4-bit (rotation, Lloyd-Max codebook, one norm per row) at 0.950 / 0.9966. Every 3-bit and
2-bit variant fails, and so does 4-bit with a single scale per row. remex at 2 bits, which
costs corpus vectors almost nothing in `embeddinggemma2-remex-quant`, gets 0.832 / 0.958
here.

Pruning the table to the tokens the corpus and training queries use keeps 88,847 of
262,144 rows. 2.1% of the 2,000 held-out queries contain a token outside that set and go to
the full table. With that fallback, pruning is exact: every query has cosine ≥ 0.9999999 to
fp32 and overlap 1.000. The combined model (pruned table, quantized, with a fallback to the
quantized full table) scores the same as the quantization arm alone, because both arms
apply the same per-row quantization. Its table is 45.7 MB at int8, 28.4 MB at int4-g32 and
22.9 MB at remex4, against 268.4 MB for the full table in bf16.

The test's reference is the fp32 model. The corpus itself was encoded by `model_q4.onnx`,
which agrees with fp32 on only 0.865 of the top 10 (1st-percentile cosine 0.977). Every
passing variant here sits closer to fp32 than the corpus's own encoder does.

## Ground truth that differs from the brief

- **The table is 262,144 × 512, not × 768.** EG2's text model runs at hidden size 512, and
  a final `embedding_projection` maps to 768. The table holds 134.2M parameters and the 24
  text layers 130.1M, so the table is half the text model, not two thirds. The checkpoint is
  bf16, and "fp32 EG2" here means bf16 weights computed in fp32. EG2 scales by √512 after
  the lookup (`EmbeddingGemma2TextScaledWordEmbedding`), so swapping the weight dequantizes
  before the scaling.
- **The table feeds every layer, not just the first.** EG2's per-layer embeddings (`ple`)
  are a projection of the looked-up rows, gated into each of the 24 layers. Table error
  therefore enters each layer directly. Corpus-vector quantization error enters only one
  final dot product, which averages it over 768 dimensions. This is the likely reason 2 bits
  works for corpus vectors and fails for the table.
- **The shipped q4 ONNX already stores the table at 4 bits** (`GatherBlockQuantized`,
  quant / scales / zero-point initializers). Arm 1 as specified (int8) would grow the
  deployed table. The useful question is whether it can go below 4 bits, and it cannot at
  this bar.

## Results

2,000 held-out queries: all 1,000 real ones (MS MARCO tech/AI, held out by topic) and 1,000
of the 1,498 synthetic test queries (seed 0). Scored against the committed 61,686-post
Bluesky index (EG2 q4, remex 2-bit codes). Query prompt `task: search result | query: `,
768-d, L2-normalized, max 128 tokens: the same as the corpus-side query targets. Pass
means mean top-10 overlap ≥ 0.95 and 1st-percentile per-query cosine ≥ 0.995.

### Arm 1: table quantization, full vocabulary

| variant | table MB | overlap@10 [95% CI] | cos p01 | cos min | real / synth overlap | pass |
|---|---|---|---|---|---|---|
| bf16 (as shipped in safetensors) | 268.4 | 1 (reference) | 1 | 1 | — | — |
| int8, one scale per row | 134.7 | 0.994 [0.992, 0.995] | 0.99998 | 0.9997 | 0.994 / 0.993 | yes |
| int4, asymmetric, groups of 32 | 83.9 | 0.959 [0.956, 0.961] | 0.9977 | 0.9939 | 0.960 / 0.957 | yes |
| remex 4-bit | 67.6 | 0.950 [0.947, 0.953] | 0.9966 | 0.9860 | 0.952 / 0.948 | yes, by 0.0001 |
| int4, one scale per row | 67.6 | 0.928 [0.925, 0.931] | 0.9912 | 0.9792 | 0.929 / 0.927 | no |
| int3, groups of 32 | 67.1 | 0.919 [0.915, 0.922] | 0.9890 | 0.9713 | 0.921 / 0.917 | no |
| remex 3-bit | 50.9 | 0.910 [0.906, 0.914] | 0.9850 | 0.9496 | 0.910 / 0.909 | no |
| int2, groups of 32 | 50.3 | 0.819 [0.813, 0.825] | 0.9478 | 0.9094 | 0.822 / 0.815 | no |
| remex 2-bit, centered | 34.1 | 0.835 [0.829, 0.840] | 0.9590 | 0.9279 | 0.836 / 0.833 | no |
| remex 2-bit | 34.1 | 0.832 [0.826, 0.837] | 0.9578 | 0.9339 | 0.829 / 0.834 | no |
| *context: `model_q4.onnx` (whole model, corpus encoder)* | — | 0.865 [0.860, 0.869] | 0.9771 | 0.9592 | — | no |

Table MB counts code bits plus fp16 scales (per row), or fp16 scale and zero point (per
group of 32). remex 4-bit passes the overlap bar with 0.0001 to spare, which is inside its
confidence interval: treat it as borderline. At equal size (67.6 MB), remex 4-bit beats
per-row int4 by 0.022 overlap, so rotation plus a Lloyd-Max codebook does what it does for
vectors. It is just not enough below 4 bits. Centering the table helps at 2 bits by 0.003,
inside the noise.

### Arm 2: pruning to corpus-plus-query coverage

| | |
|---|---|
| token ids the full tokenizer emits on 61,686 posts + 109,274 training queries | 78,225 |
| kept rows after adding the 256 byte tokens and added tokens, closed under BPE merges | 88,847 (33.9%) |
| table, bf16 | 91.0 MB (from 268.4) |
| tokenizer.json gzip, pruned / full | 1.6 MB / 5.4 MB |
| held-out queries routed to the full table | 42 of 2,000 (2.1%): 2.9% of real, 1.3% of synthetic |
| covered route (1,958): overlap / cos min | 1.000 / 0.9999999 |
| fallback route (42): overlap / cos min | 1.000 / 0.9999999 |
| fallback queries forced through the pruned tokenizer instead | overlap 0.755 [0.707, 0.798], cos p01 0.932: fail |

The training queries are MS MARCO non-tech, synthetic train and the GPU-round synthetic
queries. No test query is among them. Rows are copied bit-for-bit, so the covered route
reproduces fp32 exactly. The fallback route *is* fp32. The no-fallback row shows the cost
of the fallback on the 42 queries that need it. Without it they lose a quarter of their
top 10.

The pruned tokenizer (merges whose parts or result were cut are dropped, byte fallback on)
reproduces the full tokenizer's ids on only 97.5% of covered queries. So routing has to
run the full tokenizer, or an equivalent check, before choosing the table. The pruned
tokenizer alone cannot tell which queries need the fallback. I did not find out why it
diverges on the other 2.5%. A plausible cause is a merge whose result string is reachable
by more than one parent pair.

### Combined: pruned table, quantized, fallback to the quantized full table

| table variant | pruned table MB | all: overlap / cos p01 | covered (1,958) | fallback (42) | pass |
|---|---|---|---|---|---|
| int8 per row | 45.7 | 0.994 / 0.99998 | 0.994 / 0.99998 | 0.993 / 0.99998 | yes |
| int4 g32 | 28.4 | 0.959 / 0.9977 | 0.959 / 0.9977 | 0.948 / 0.9982 | yes (fallback overlap 0.948 on n = 42) |
| remex 4-bit | 22.9 | 0.950 / 0.9966 | 0.951 / 0.9966 | 0.929 / 0.9971 | borderline |
| remex 3-bit | 17.2 | 0.910 / 0.9850 | 0.910 / 0.9850 | 0.890 / 0.9802 | no |

As expected, the combined rows equal arm 1 to within batching noise (remex 4-bit: 0.95005
against 0.95010), because quantization is per row or per group within a row. The fallback
route's lower overlap rests on 42 queries, with confidence intervals about ±0.02 wide.

## Verdict

- **Arm 1, int8 per row: pass, with a wide margin.** It halves bf16, but the deployed q4
  ONNX is already at 4 bits, so it saves nothing over what ships.
- **4 bits is the floor for the table.** int4 with groups of 32 passes and remex 4-bit
  passes at the edge. Nothing at 3 or 2 bits passes. The theory that the table would
  quantize like corpus vectors holds to 4 bits and breaks below. 2-bit remex loses 17% of
  the top 10.
- **Arm 2: pass, exact.** It needs the full tokenizer (or a dropped-merge check) for
  routing, and a full table somewhere for 2.1% of queries. The obvious place is the hosted
  `eg2-query-embedder` endpoint, so the browser model never carries the full table.
- **Combined: pass** at int8 (45.7 MB table) and int4-g32 (28.4 MB). remex 4-bit
  (22.9 MB) is borderline.

## What this means for the size budget

The eg2-layer-student session (experiments#103, `eg2-layer-student/results/quant_full.json`)
measured the transformer side against the q4 teacher, so its numbers are not on this test's
fp32 scale. The q4 layers are about 88 MB. At 3 bits (GPTQ, 85.3 MB file) top-10 agreement
falls to 0.742, and at 2 bits to 0.390. Neither half of EG2 tolerates going below 4 bits at
a top-10-agreement bar. With the table pruned and at 4 bits (23–28 MB), the layers are now
three quarters of the download. A smaller model needs fewer layers, not fewer bits. The
same session's 4-layer cut is 29.0 MB.

The bar itself is strict. Top-10 overlap against fp32 penalizes reorderings that leave
retrieval quality unchanged: #102 found hit@10 moved 0.009 at 0.770 overlap. The corpus's
own q4 encoder scores 0.865 against fp32. If the deployment's real reference is the q4
corpus encoder rather than fp32, re-score the 3-bit variants against `real_test.eg2.npy` /
`synth_test.eg2.npy` (the q4 targets in the embedding store) before ruling them out.

## Method notes

- No corpus re-embedding. The doc side is the committed 2-bit codes of
  `embeddinggemma2-webgpu-tail/web/bsky`, scored as remex `search_adc` scores them (the
  `Index` class from eg2-query-student).
- Post text was re-fetched from Bluesky (`run/hydrate_posts.py`, 0 of 61,686 missing). It
  is used only for vocabulary coverage and is kept out of git (`.work/`).
- The fp32 pipeline (`run/common.py` `EG2`) matches `SentenceTransformer.encode` with the
  query prompt to cosine 0.9999999.
- The query sets come from the embedding store, `bsky-ai-list/embeddinggemma-2-q4/query-targets-v1`.

## Reproduce

```bash
pip install "transformers>=4.57" sentence-transformers==6.1.0 remex==2.1.0 tokenizers
mkdir -p .work/queries
B=https://pub-38421218e82c4d4ba2bc53974c28b55d.r2.dev/embeddings/bsky-ai-list/embeddinggemma-2-q4/query-targets-v1
for f in real_test synth_test msmarco_train synth_train gpu_more; do curl -sSfA x -o .work/queries/$f.jsonl $B/$f.jsonl; done
for f in real_test synth_test; do curl -sSfA x -o .work/queries/$f.eg2.npy $B/$f.eg2.npy; done
python3 -c "from huggingface_hub import snapshot_download as s; s('google/embeddinggemma-2', local_dir='.work/hf')"
python3 -I run/hydrate_posts.py      # ~3 min
python3 -I run/arms.py               # ~25 min on 4 vCPU
```
