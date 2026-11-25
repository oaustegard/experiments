# sparseup-v2: README index row, notes section and METHODS entries

For a session with a git checkout of oaustegard/experiments. README.md and METHODS.md
are ~500 KB each, too big to rewrite through the API. Grep METHODS.md first and skip
any entry already there; match the neighbouring format in both files. Apply on main,
delete this file and the `sparseup-v2-index` branch.
Commit message ends with: Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

## 1. README.md index row (top of the table; newest first)

| [`sparseup-v2/`](sparseup-v2/RESULTS.md) | 2026-11-25 | **done — Linkup's SparseUp v2 sparse retriever under the remex treatment and in a browser pack. Index: bf16 output has under 200 distinct values per corpus, so 8 bits per value is exactly lossless (836 B/doc vs 1,672 at fp32); top-128 terms at 2-bit Lloyd-Max is 287 B/doc (5.8x) at macro nDCG@10 −0.0016 [−0.0067, +0.0035] over SciFact, NFCorpus, FiQA and ArguAna (FiQA −0.022); at matched bytes more terms at fewer bits beats fewer terms at fp16 by 0.026. remex's per-document norm costs 0.006–0.007. Browser: a hand-written WGSL encoder with a fused SPLADE kernel matches int8 PyTorch to cosine 0.99999988 and runs a 512-token document in 34.4 ms on a Mac GPU from a 137.5 MB int8 pack; 4-bit saves 33 MB and is no faster** | [`RESULTS.md`](sparseup-v2/RESULTS.md) + [`remex/REPORT.md`](sparseup-v2/remex/REPORT.md), [`pack/REPORT.md`](sparseup-v2/pack/REPORT.md) | Oskar: "give this hot new model both the remex and the packing treatment: delegate work to cheaper models as you deem Pareto appropriate" (after a Tom Aarsen post) |

Also bump the experiment count and date in the intro paragraph if that is the convention.

## 2. README.md, under "Per-experiment notes"

### sparseup-v2

Two sessions on `Linkup-Platform/linkup-sparseup-embed-v2` (SPLADE head on
ModernBERT-base). `remex/` measures how small the stored document vectors get
(262 arms per dataset: top-k × value coding × per-document scale, paired bootstrap
CIs); `pack/` is a `SeqEngine` encoder on `_lib/wgsl/` plus a new `splade` kernel
(activation, per-position top-12 with ties kept, max-pool and case/space fold in one
pass). Gemini 4 Hydrogen wrote four mechanical pieces for $0.18; no paid compute.
GPU parity and timing ran on Oskar's Mac through the GPU Bridge. Not run: query-side
top-k, documents past 512 tokens, an end-to-end search engine.

## 3. METHODS.md entries (put each in the matching topical section)

- **Count distinct values before picking a codebook for a bf16 model's output.**
  SparseUp v2 computes its activation in bf16, so only 171–177 distinct weights
  occur per corpus; an 8-bit index into a table of them is exactly lossless, where a
  generic 8-bit uniform code is merely close. Any bf16-native encoder output may have
  the same property. (`sparseup-v2/remex`)

- **For a learned-sparse index on a byte budget, spend bytes on terms, not bits.**
  At ~256–287 B/doc, top-128 terms at 2 bits beats top-64 at fp16 by +0.026
  nDCG@10 [+0.019, +0.032]; every fp16 or 8-bit arm was dominated by a lower-bit arm
  with more terms. (`sparseup-v2/remex/results/paired.json`)

- **A per-document scale hurts sparse vectors whose values already share one scale.**
  remex's exact-norm idea (quantize v/‖v‖, rescale) cost −0.0061, and an fp16 doc max
  −0.0072, against no scale at 2 bits: normalising moves the same raw weight onto
  different code levels in different documents. Rotation does not apply at all (it
  densifies). (`sparseup-v2/remex`)

- **A bf16 CPU encode depends on batch composition.** The same 300 docs at batch 4 in
  corpus order vs length-sorted batches give cosine 0.9995 (min 0.9953). Compare
  encodes made with the same batching before calling a difference real. Relatedly,
  Sentence Transformers' `max_active_dims` equals top-k only up to ties at the k-th
  value, which 70–99% of SPLADE rows have. (`sparseup-v2/remex`)

- **Quantized retrieval arms create score ties; break them as trec_eval does** (doc id
  descending, `ignore_identical_ids`), or tie order adds noise to every delta. ArguAna
  holds duplicate documents with identical scores. (`sparseup-v2/remex`)

- **Kaggle GPU containers have no graphics capability, so no real-GPU WebGPU runs
  there.** Use a real browser GPU (the GPU Bridge to a Mac) for WGSL parity and timing;
  SwiftShader is ~0.5 s per token and no guide to GPU speed. (`sparseup-v2/pack`)
