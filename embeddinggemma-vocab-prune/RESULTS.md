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

## Follow-up: table bits on the cut students (experiments#103), with QAT

Run 2026-10-21, 4 vCPU, about 3.5 h. Question: if EG2 is already cut to 2–8 layers, the
41,741-row table (#102's 32K cut) is 47–63% of the download. Can it go below 4 bits there, and does
quantization-aware training (QAT) help? Metric is the board's `ASYM` one (experiments#104):
the query encoder varies and the doc side stays EG2 q4's committed 2-bit codes. The reference
is the q4 teacher. It is scored with #103's `run/bench.py`: top-10 overlap with EG2 on
1,000 real_test queries, and synth_test source-post hit@10 on 1,498 queries. Tables are
asymmetric min/max in groups of 32 (GatherBlockQuantized's layout). "q4 layers" fake-quantizes
every transformer linear to 4 bits in groups of 32, round-to-nearest. That reproduces #103's
shipped q4 files to within 0.001 (k=2 2-epoch: 0.5463 / 0.6680 / 0.8885 against the shipped
0.5463 / 0.6674 / 0.8892).

### QAT does not help

`run/qat.py`, on k=2 reproduced from #103 (2 epochs, 0.546 / 0.6723 / 0.8865, exact match).
Each arm gets one more epoch: lr 1e-3, batch order seed 2, the same 164,588 targets. The control
uses a float table. The QAT arms use a fake-quantized table with a straight-through estimator.
All rows below have q4 layers. The file is `results/qat_k2_<arm>.json`.

| arm | table | real overlap | synth overlap | hit@10 |
|---|---|---|---|---|
| control (+1 epoch, float table) | float | 0.568 | 0.678 | 0.899 |
| control | 3-bit after training | 0.564 | 0.675 | 0.896 |
| QAT 3-bit | 3-bit | 0.564 | 0.679 | 0.899 |
| control | 2-bit after training | 0.548 | 0.662 | 0.894 |
| QAT 2-bit | 2-bit | 0.549 | 0.667 | 0.887 |

Paired differences, bootstrap 95% CI. Overlap CIs come from `.work`; hit@10 CIs come from
`results/k2_*_perquery.json`:

| comparison | real overlap | synth overlap | hit@10 |
|---|---|---|---|
| 2 bits after training vs float | −0.020 [−0.025, −0.015] | −0.016 [−0.020, −0.013] | −0.005 [−0.010, 0.000] |
| 2-bit QAT vs 2 bits after training | +0.001 [−0.005, +0.006] | +0.005 [+0.001, +0.010] | −0.007 [−0.014, −0.001] |
| 3 bits after training vs float | −0.005 [−0.008, −0.002] | −0.004 [−0.006, −0.001] | −0.003 [−0.006, 0.000] |
| 3-bit QAT vs 3 bits after training | +0.000 [−0.004, +0.004] | +0.004 [+0.001, +0.007] | +0.003 [−0.002, +0.007] |

QAT recovers none of the real-query loss and about a quarter of the synthetic one. At
2 bits it costs hit@10. The likely reason is that the student already trains its table, so the
rows that matter have been fitted, and quantizing them afterwards is close to what QAT finds.
I tried one QAT recipe (a single epoch at lr 1e-3, STE on min/max groups), so this does not
rule out a longer or learned-scale QAT.

### Quantizing the table after training, on the 10-epoch Mac builds

Checkpoints are from release `eg2-layer-mac-20261020` (`out/l1/mac_k{2,4,8}_v32k/final.pt`),
files `results/qat_mac_k*.json`. All rows have q4 layers. Download is #103's q4 file size, minus
the table bits saved. A 41,741 × 512 table saves 2.67 MB per bit.

| k | table | download | real overlap | synth overlap | hit@10 |
|---|---|---|---|---|---|
| 2 | 4-bit (as shipped) | 21.8 MB | 0.577 | 0.686 | 0.898 |
| 2 | 3-bit | ~19.1 MB | 0.572 | 0.684 | 0.898 |
| 2 | 2-bit | ~16.5 MB | 0.554 | 0.661 | 0.893 |
| 4 | 4-bit (as shipped) | 29.0 MB | 0.629 | 0.721 | 0.906 |
| 4 | 3-bit | ~26.3 MB | 0.624 | 0.718 | 0.906 |
| 4 | 2-bit | ~23.7 MB | 0.607 | 0.702 | 0.903 |
| 8 | 4-bit (as shipped) | ~44 MB | 0.653 | 0.747 | 0.911 |
| 8 | 3-bit | ~41.4 MB | 0.649 | 0.747 | 0.910 |
| 8 | 2-bit | ~38.8 MB | 0.641 | 0.734 | 0.907 |

The same pattern holds at every depth. 3 bits costs about 0.005 real overlap and nothing
measurable on hit@10. 2 bits costs 0.012–0.023 real overlap and 0.003–0.006 hit@10. The 2-bit
loss is the same at k=2 and k=4 (−0.023) and about half that at k=8 (−0.012). The 32K cut keeps the full tokenization of 84.1% of real_test and
83.0% of synth_test (`results/perquery_summary.json`). The rest are re-split, and the student is
trained on the cut tokenizer. There is no fallback.

### Verdict on the theory

On a cut student, the table goes to 3 bits for free on the task metric, saving 2.7 MB
(k=2: 21.8 → 19.1 MB). 2 bits saves 5.3 MB for about −0.005 hit@10 and −0.02 top-10 agreement.
That is a real trade. It is smaller than the full-EG2 result, where 2 bits lost 17% of the
top 10, because a 2–8 layer student has less depth to amplify table error and was trained on its
table. QAT is not the lever. Quantizing after training does as well. These are download
formats: a 3- or 2-bit table needs a standard-op unpack to run on onnxruntime-web wasm
(GatherBlockQuantized covers 4 and 8 bits), which is not built yet.

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
# follow-up (needs eg2-layer-student's run/fetch_data.py, run/vocab.py and a checkpoint):
python3 -I run/qat.py --k 2 --init ../eg2-layer-student/out/l1/k2_ep2_v32k/final.pt --bits 2 --tag k2_b2
python3 -I run/perquery.py
```
