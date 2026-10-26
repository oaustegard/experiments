# moederncolbert-bench: SauerkrautLM-MoEdernColBERT in the browser, and whether its sizes share a space

**Date:** 2026-10-26 · **Status:** done (benchmark). A distilled cross-size query encoder was proposed to Oskar and is not started.

VAGOsolutions released SauerkrautLM-MoEdernColBERT on 2026-10-23: multilingual ColBERT retrievers at
1.5M, 27M and 115M parameters on a new encoder, MoEdernBERT (a hashed, factorized token embedding,
attention layers with a top-2-of-8 mixture-of-experts MLP, and in the 115M size gated short-convolution
layers). Two questions from Oskar: how fast do the two smallest sizes run next to the query
embedders the Embedded Embedder project already runs in the browser (EG2 cut, a 4-layer cut of the
EG2 query encoder, and Ettin-17m, both on hand-written WebGPU engines), and do the sizes share a
vector space, so that documents indexed with 115M could be searched with 1.5M queries?

Terms used below. *Parity* is the minimum per-token cosine between the browser engine's output and
the PyTorch reference. *The bridge* is a page on one of Oskar's devices that runs timing jobs on its
real GPU and posts the results back. SwiftShader is Chrome's software GPU, used for parity only.

## Answer

The sizes do not share a vector space. On 13 English NanoBEIR sets each size matches its
model card alone (nDCG@10 0.590 / 0.619 / 0.642 <!-- @evidence[[results/summary.json#/ndcg/1.5M>1.5M]] -->), and every
mixed pair scores near zero (random ranking): 1.5M queries against 115M documents give 0.004 <!-- @evidence[[results/summary.json#/ndcg/1.5M>115M]] -->.
Token vectors agree at cosine −0.009 <!-- @evidence[[results/summary.json#/align/1.5M~0115M]] -->. An
orthogonal map into 115M's space recovers only 0.086 for 1.5M → 115M <!-- @evidence[[results/summary.json#/ndcg/1.5M*map>115M]] -->. A small query model would need training into 115M's space.

On Oskar's Mac (Chrome 155, Metal 3) a 32-token query takes 0.464 ms <!-- @evidence[[results/summary.json#/mac/1.5M_query_ms]] --> on 1.5M and 0.963 ms on 27M <!-- @evidence[[results/summary.json#/mac/27M_query_ms]] -->. On ~32-token queries EG2 cut takes 0.997 ms <!-- @evidence[[results/summary.json#/mac/eg2_cut_near32_ms]] --> and Ettin-17m 0.667 ms <!-- @evidence[[results/summary.json#/mac/ettin_near32_ms]] -->. Documents: 1,161,965 tokens/s on 1.5M <!-- @evidence[[results/summary.json#/mac/1.5M_tokens_per_s]] -->
and 174,497 on 27M <!-- @evidence[[results/summary.json#/mac/27M_tokens_per_s]] -->. An iPhone was 6× slower per query and 16× slower on documents.

A distilled 1.5M query model scoring well on 115M documents would change this.

## Findings

1. **No shared space.** All six mixed query/document pairs score 0.002–0.005 nDCG@10, against
   0.59–0.642 for the matched pairs (`results/ndcg.json`). The three sizes are separately trained
   encoders that share only the tokenizer and the 128-d head width.
2. **The token-level check agrees.** All sizes tokenize identically, so a document's row *i* is
   the same token in the same context in every size. Mean cosine between sizes is −0.013 to 0.003 <!-- @evidence[[results/summary.json#/align/27M~0115M]] -->
   over the six pairs (20,000 rows per set); 1.5M against 115M is −0.009.
3. **A linear map does not rescue it.** Orthogonal Procrustes (128×128, fit leave-one-set-out on
   matching document tokens, then applied to the query side's tokens) gives 0.444 for 27M → 115M <!-- @evidence[[results/summary.json#/ndcg/27M*map>115M]] --> (below 27M's own 0.619) and 0.086 for 1.5M → 115M.
4. **Browser query latency is in the same range as the existing engines.** Mac, back-to-back loops,
   queries of about 32 tokens: 1.5M 0.464 ms, 27M 0.963 ms, EG2 cut 0.997 ms, Ettin-17m 0.667 ms. 1.5M
   is fastest and 27M ties EG2 cut. All four are under 1 ms and dominated by dispatch and readback.
   For queries of 20 tokens or fewer, EG2 cut takes 0.95 ms <!-- @evidence[[results/summary.json#/mac/eg2_cut_short_ms]] --> and Ettin 0.812 ms <!-- @evidence[[results/summary.json#/mac/ettin_short_ms]] -->.
5. **Document throughput needs a batch kernel.** The shared 8-row matmul (`kernels.js`) gives 27M
   130,702 tokens/s <!-- @evidence[[results/summary.json#/mac/27M_tokens_per_s_template]] -->. A 64×64 tiled kernel lifts that to 174,497, but makes single queries
   slower (27M 2.086 ms <!-- @evidence[[results/summary.json#/mac/27M_query_ms_tiled]] -->). An engine should pick the kernel by batch size.
6. **Device matters more than model.** iPhone (iOS 27.0.1, Chrome for iOS, so WebKit's WebGPU):
   1.5M 3 ms, 27M 5 ms per query <!-- @evidence[[results/summary.json#/iphone/27M_query_ms]] --> (one significant figure: that run predates the amortized loops), and
   27M documents at 8,122 tokens/s, with only the 8-row kernel. 1.5M parity (minimum cosine) there is 0.9958 <!-- @evidence[[results/summary.json#/iphone/1.5M_parity_min_cos]] --> (1.0 on
   the Mac and on SwiftShader). The likely cause is the router: 1.5M has a top-2/top-3 probability
   gap of 5.6e-8 on one fixture token, small enough for a rounding difference to flip an expert.
   This is not yet confirmed per row.
7. **Server-side encoding (plain PyTorch fp32, no Triton).** 4 vCPUs, 1.5M: 217 docs/s <!-- @evidence[[results/summary.json#/encode/cpu_1.5M_docs_per_s]] -->. Colab L4: 27M
   576 docs/s <!-- @evidence[[results/summary.json#/encode/l4_27M_docs_per_s]] -->, 115M 194 docs/s <!-- @evidence[[results/summary.json#/encode/l4_115M_docs_per_s]] -->. Server timings include tokenization. The Mac's browser engine, on the 64 fixture documents (314 tokens on average) and without tokenization (3,699 docs/s for 1.5M <!-- @evidence[[results/summary.json#/mac/1.5M_docs_per_s]] -->, 555.6 for 27M <!-- @evidence[[results/summary.json#/mac/27M_docs_per_s]] -->) is on
   par with the L4 for 27M.
8. **A pipeline trap in sentence-transformers 6.2.** `encode_query` inserts the `[Q]` marker
   token (id 30000) after `[CLS]`, but `preprocess(task="query")` does not. A reimplementation copied
   from `preprocess` gets cosine ~0.5 per token. The model detects a query by that marker and then
   attends over all 32 positions, `[MASK]` padding included.

## Method

- **Reference.** `run/mcb.py` reimplements the forward pass from `model.safetensors` and
  `1_Dense/model.safetensors` without running the repo's remote code. Its outputs match the official
  sentence-transformers 6.2 pipeline at cosine 1.0000 for all three sizes. That pipeline ran in
  `scripts/jail.sh` (no network), since it executes `modeling_modernbert_hash_moe_v10.py`.
- **Retrieval.** 13 English NanoBEIR sets (`zeta-alpha-ai/Nano*`), 56,723 docs and 649 queries in total,
  document length 512, punctuation skiplist on the document side, queries padded to 32 with
  `[MASK]`. Brute-force MaxSim and nDCG@10 per set, then the mean over sets (the card's convention). The
  reported 1.5M numbers come from a CPU encode, 27M and 115M from a Colab L4. 1.5M was also encoded on the L4 as a check: 0.5904 there, 0.5907 on CPU.
- **Cross-space arms.** Query tokens of one size against document tokens of another. Alignment is
  the mean cosine of matching document rows. The Procrustes map is fit on 20,000 sampled rows from
  each of the other 12 sets.
- **Browser engine.** `web/engine.js` extends `_lib/wgsl/runtime.js` and reads the checkpoint's
  safetensors directly (f32: 5.9 MB for 1.5M, 108.5 MB for 27M <!-- @evidence[[results/summary.json#/weights_mb_f32/27M]] -->). The MoE is sparse: a router kernel
  appends each (token, slot) to its expert's list and grouped matmuls walk only those rows.
  Sequences go up to 512 tokens. The 115M size's conv layers are not implemented. Parity:
  `node run/check.js` (SwiftShader) and on-device in every bridge job, per token row of
  `web/fixtures.json` (16 SciFact queries and 64 documents, 20,614 rows).
- **Timing.** GPU bridge (`gpu-bridge/jobs/moedern`, `moedern-refs`). `embed()` calls run back to
  back with readback included, total divided by count (the worker clock ticks in ~1 ms), median of 5
  rounds. Documents: the 64 fixture docs in 3 packed batches of ≤ 8,192 tokens. Tokenization is not
  timed and the browser has no tokenizer yet. The reference engines use their own fixtures
  (short = ≤ 20 tokens).

## Log

### Round 1 (2026-10-26): model, retrieval, space

Read the cards and configs. 1.5M and 27M are 2 attention+MoE layers (hidden 128 / 640). 115M has 10
layers, 4 of them attention+MoE and 6 gated conv. All share a 30,002-token Unigram tokenizer and a
128-d ColBERT head. Reimplemented the model. The first parity attempt (no `[Q]` marker, copied from
`preprocess`) gave cosine ~0.5. Adding the marker gave 1.0000. 1.5M on CPU: 261 s for the 13 sets,
NanoBEIR-en 0.5907 (card 0.591). Encoding 27M on CPU projected to about an hour, so 27M and 115M
went to a Colab L4 (9.6 min for both). The 3×3 matrix, alignment and Procrustes ran there too,
because the fp16 token files for all sizes come to about 4 GB.

| query ↓ / docs → | 1.5M | 27M | 115M |
|---|---:|---:|---:|
| 1.5M | 0.590 | 0.004 | 0.004 |
| 27M | 0.003 | 0.619 | 0.002 |
| 115M | 0.005 | 0.003 | 0.642 |

Mapped: 27M → 115M 0.444, 1.5M → 115M 0.086.

### Round 2 (2026-10-26): WGSL engine and the bridge

Wrote `web/engine.js` on the shared runtime with the 8-row `kernels.js` matmul. SwiftShader parity
for 1.5M: cosine 1.0 on all 20,614 rows. First bridge job `moedern-1` ran on an iPhone (the bridge
then reported only `apple apple apple`; page v6 now records device and user agent): 1.5M 3 ms, 27M
5 ms per query, 27M documents at 8,122 tokens/s. 1.5M parity 0.9958. The job's reference section
crashed (inside an EG2 `embed`, likely the tab going to the background).

Added a 64×64 tiled matmul (`mmt`) for dense and expert-grouped rows. `moedern-2` and `moedern-3`
ran on the Mac with both kernels and agreed within 15%:

| Mac | query ms (8-row / tiled) | 16 queries ms | doc tokens/s (8-row / tiled) |
|---|---:|---:|---:|
| 1.5M | 0.464 / 0.731 | 0.67 / 0.80 | 1,025,612 / 1,161,965 |
| 27M | 0.963 / 2.086 | 3.86 / 3.92 | 130,702 / 174,497 |

`moedern-refs-3`, same method on the same Mac: EG2 cut 0.95 ms short / 0.997 ms near 32 tokens /
2.135 ms at 118 tokens. Ettin-17m 0.812 / 0.667 / 1.105 ms at 128 tokens. Two retries before it
(`moedern-refs-1`, `-2`) failed on a transient jsDelivr 403 for `_lib/wgsl/runtime.js`. The second
failed because the persistent worker had cached the failed module import for that URL. Re-running at
another commit with the same engine code fixed it.
