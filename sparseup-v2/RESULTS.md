# RESULTS: SparseUp v2 under the remex treatment and in a browser pack

`Linkup-Platform/linkup-sparseup-embed-v2` is a SPLADE-style learned sparse retriever on
ModernBERT-base (paper arXiv 2611.09680): every text becomes a few dozen to a few hundred
weighted vocabulary terms, scored by dot product. Two sessions worked on it on 2026-11-25:
[`remex/`](remex/REPORT.md) measured how small its document vectors can get, and
[`pack/`](pack/REPORT.md) runs the encoder in a browser tab on hand-written WGSL. This file
joins the two; each subdirectory's REPORT.md has the full tables.

## Answer

Does SparseUp v2 survive the remex treatment (compressing its stored document vectors) and
a browser pack? Mostly: averaged over SciFact, NFCorpus, FiQA and ArguAna neither loses measurable
retrieval quality (nDCG@10, higher is better), with one caveat each.

Index side: bf16 output holds under 200 distinct weight values per corpus, so
8 bits per value is exactly lossless (836 B/doc against 1,672 at fp32). <!-- @evidence[[remex/REPORT.md]] -->
Keeping each document's top 128 terms at 2 bits gives 287 B/doc, 5.8x smaller, at
−0.0016 [−0.0067, +0.0035]; FiQA alone loses 0.022. At matched bytes, more terms at fewer bits beats fewer terms at
more bits by 0.026. <!-- @evidence[[remex/results/paired.json#k128_lloyd2_minus_k64_fp16.d]] -->
remex's random rotation does not apply (it would make the sparse vector dense) and its
exact per-document norm costs 0.006 to 0.007 at 2 bits.

Encoder side: the browser encoder, in WGSL (WebGPU's shader language), matches int8
PyTorch to cosine 0.99999988 on 47 texts, <!-- @evidence[[pack/results/summary.json#int8.kernel_vs_dequant_cos_min]] -->
runs a 512-token document in 34.4 ms on a laptop GPU, <!-- @evidence[[pack/results/summary.json#int8.mac_ms_512]] -->
and moves SciFact nDCG@10 by −0.0027 [−0.0068, +0.0015] from a 137.5 MB (brotli) int8
pack. 4-bit saves 33 MB and is no faster.

Corpora where top-128 already loses, as FiQA does, or documents past 512 tokens (untested)
would change this.

## Findings

1. **8-bit values are lossless by construction.** (Round 1, `remex/results/summary.json`)
   The activation is computed in bf16, so 171 to 177 distinct values occur per corpus and an
   8-bit table holds them all: delta 0.0000. Anserini-style 8-bit impacts, the prior art,
   are −0.0002. <!-- @evidence[[remex/REPORT.md]] -->
2. **Term count, not value precision, is what costs.** (Round 1, `remex/results/paired.json`)
   At ~256 to 287 B/doc, top-128 at 2 bits beats top-64 at fp16 by +0.026 [+0.019, +0.032]. <!-- @evidence[[remex/results/paired.json#k128_lloyd2_minus_k64_fp16.d]] -->
   Every fp16 or 8-bit arm is dominated by a lower-bit arm with more terms.
3. **Top-256 pruning is free; top-128 is free only on average; below that it costs.**
   (Round 1, `remex/REPORT.md`) The model was trained to tolerate top-k cutoffs at 256, 128
   and 64 terms. Top-256 +0.0042 macro, top-128 +0.0002 macro (FiQA −0.013 offset by
   ArguAna +0.022), top-64 −0.027, top-32 −0.088. <!-- @evidence[[remex/REPORT.md]] -->
4. **remex's exact per-document norm hurts here.** (Round 1, `remex/results/paired.json`)
   At full k and 2 bits, an fp16 doc max costs −0.0072 [−0.0112, −0.0030] against no scale, <!-- @evidence[[remex/results/paired.json#full_lloyd2_docmax_minus_none.d]] -->
   and the exact L2 norm −0.0061 [−0.0104, −0.0020]. <!-- @evidence[[remex/results/paired.json#full_lloyd2_l2_minus_none.d]] -->
   The values already share one scale; normalising each document moves them off the shared grid.
5. **The WGSL kernels add no measurable error.** (Round 1, `pack/results/summary.json`)
   Against PyTorch on the same dequantized int8 weights, 47 texts: cosine min 0.99999988,
   max weight difference 2.4e-5. <!-- @evidence[[pack/results/summary.json#int8.kernel_vs_dequant_max_abs]] -->
   The reference itself is the model's own `modeling_splade.py`, which agrees with the
   pack's `ref.py` bit for bit on 38 of 47 texts; <!-- @evidence[[pack/results/official_vs_ref.json#identical_texts]] -->
   on the other 9 the largest weight difference is 1.1e-5. <!-- @evidence[[pack/results/official_vs_ref.json#max_abs_max]] -->
6. **int8 weights stay within noise of fp32 on retrieval, and the two sessions reproduce
   each other's int8 number.** (Round 1, `pack/results/summary.json`) int8 vs fp32 cosine
   median 0.9992, <!-- @evidence[[pack/results/summary.json#int8.vs_fp32_cos_median]] --> bf16 vs fp32 0.9998. <!-- @evidence[[pack/results/summary.json#bf16.vs_fp32_cos_median]] -->
   SciFact int8 delta −0.0027 in the pack session <!-- @evidence[[pack/results/scifact.json#vs_fp32.int8.delta_ndcg10]] -->
   and −0.0027 in the remex session (same fake-quantization recipe, so a replication), <!-- @evidence[[remex/results/weightquant.json#datasets.scifact.int8.d_ndcg10]] -->
   NFCorpus +0.0009, <!-- @evidence[[remex/results/weightquant.json#datasets.nfcorpus.int8.d_ndcg10]] --> both CIs spanning zero.
7. **4-bit is not worth it.** (Round 1, `remex/results/weightquant.json`) All-4-bit group-32
   weights cost NFCorpus −0.0100 [−0.0157, −0.0046]. <!-- @evidence[[remex/results/weightquant.json#datasets.nfcorpus.int4g32.d_ndcg10]] -->
   The pack's q4 (attention and MLP-up at 4 bits, the rest int8)
   is inconclusive on SciFact (+0.0024, CI about ±0.01) and keeps only 90% of fp32's top 10, <!-- @evidence[[pack/results/scifact.json#vs_fp32.q4.top10_overlap]] -->
   and runs 36.1 ms at 512 tokens against int8's 34.4. <!-- @evidence[[pack/results/summary.json#q4.mac_ms_512]] -->
8. **Speed on a laptop GPU.** (Round 1, `pack/results/summary.json`) Apple Metal through
   Chrome: 8.6 ms at 128 tokens, <!-- @evidence[[pack/results/summary.json#int8.mac_ms_128]] -->
   13.3 ms for 16 queries in one batch. <!-- @evidence[[pack/results/summary.json#int8.mac_ms_batch16q]] -->
   SwiftShader (WebGPU emulated on CPU) takes about 0.5 s per token and is no guide to GPU speed.
9. **The paper already covers pruning, not quantization.** (Round 1, `remex/REPORT.md`)
   arXiv 2611.09680 reports Matryoshka top-k training and its nDCG cost, with non-zeros as
   its storage measure. It does not report value quantization, bytes per document, posting
   compression or weight quantization.

## Method

- **Model and data.** The card's Sentence Transformers snippet, `encode_query` /
  `encode_document`, docs truncated at 512 tokens. BEIR test splits from `mteb/*`:
  SciFact (300 queries, 5,183 docs), NFCorpus (323, 3,633), FiQA (648, 57,638), ArguAna
  (1,406, 8,674). Baselines reproduce the paper's Table 1 within 0.4 points.
- **Index compression** (`remex/compress.py`): docs only, queries fp32. Top-k per doc ×
  value codings (fp32, fp16, 8-bit table, Anserini 8-bit, uniform or Lloyd-Max (levels fitted to the value distribution) at 2 to 8
  bits, binary), each with no scale, an fp16 doc max, or remex-style L2 renormalisation;
  262 arms per dataset. Bytes counted with 16-bit term ids (and delta-varint ids
  separately). Paired bootstrap over queries, 1,000 resamples, against fp32 at full k.
- **Weight quantization** (`remex/weightquant.py`, `pack/ref.py`): int8 per output row,
  symmetric, on every linear layer, the token table and the tied decoder; 4-bit symmetric
  in groups of 32. Fake quantization in fp32 compute, re-encoding SciFact and NFCorpus.
- **Browser pack** (`pack/pack.py`, `pack/web/`): a `SeqEngine` subclass on the shared
  `_lib/wgsl/` kernels, plus a new `splade` kernel that does the activation, the
  per-position top-12 (all ties at the cutoff kept), max-pooling and the case/space fold
  in one pass with atomic max. Parity metrics per text: cosine of the folded vectors, top-32 Jaccard, max
  weight difference.
- **Compute.** CPU in the sessions' containers; FiQA, the weight-quantization re-encodes and
  the pack's SciFact runs on Kaggle's free T4s; GPU parity and timing on Oskar's Mac through
  the GPU Bridge (his tool giving a session remote use of the Mac's browser GPU), which he
  opened and asked for. No paid compute.

## Log

### Round 1, 2026-11-25: both treatments

Oskar: "give this hot new model both the remex and the packing treatment: delegate work to
cheaper models as you deem Pareto appropriate". The coordinating session read the model's
code, decided what remex means for a sparse vector, and wrote two briefs; Oskar started
them as two Claude Code sessions after the coordinating session
could not (it runs where Hugging Face and GitHub are blocked, and may only start sessions
in its own environment). Pre-registered predictions are in `remex/PLAN.md`: 9 of 15 right,
4 half right, 2 wrong. The two that failed shared one assumption, that per-value precision
is the scarce resource.

**Delegation.** Gemini 4 Hydrogen wrote four mechanical pieces: the remex test suite and plot
script, the pack converter and the SciFact script, $0.18 in all. Claude subagents read the
paper, reviewed the remex report as a fresh reader (Sonnet) and wrote the demo page (Sonnet).
Kernel design, references and every check stayed with the two working sessions. Ledgers:
[`remex/DELEGATION.md`](remex/DELEGATION.md), [`pack/DELEGATION.md`](pack/DELEGATION.md).

**What did not work.** Kaggle containers have no graphics capability, so no real-GPU WebGPU
ran there; the Mac through the GPU Bridge supplied it. Loading the model's own code with
`trust_remote_code` was refused in the pack session's automatic mode until Oskar asked for
it; it then ran sandboxed with no network. A bf16 CPU encode depends on batch composition
(cosine 0.9995 between batchings), and `max_active_dims` equals top-k only up to ties at
the k-th value, which 70 to 99% of rows have.

**Not run.** Query-side top-k, documents past 512 tokens, an end-to-end search engine
(Seismic, Lucene) and its latency. The pack's full 47-text SwiftShader run was still going
when this was written. No README index row or METHODS.md entry yet.
