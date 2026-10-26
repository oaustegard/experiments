# moederncolbert-bench: SauerkrautLM-MoEdernColBERT in the browser, and whether its sizes share a space

**Date:** 2026-10-26 · **Status:** done (benchmark, and one distillation run).

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
orthogonal map into 115M's space recovers only 0.086 for 1.5M → 115M <!-- @evidence[[results/summary.json#/ndcg/1.5M*map>115M]] -->. Training 1.5M's query side into 115M's space reaches 0.468 <!-- @evidence[[results/summary.json#/distill/step8000]] -->, below 1.5M's own 0.590.

On Oskar's Mac (Chrome 155, Metal 3) a 32-token query takes 0.464 ms <!-- @evidence[[results/summary.json#/mac/1.5M_query_ms]] --> on 1.5M and 0.963 ms on 27M <!-- @evidence[[results/summary.json#/mac/27M_query_ms]] -->. On ~32-token queries EG2 cut takes 0.997 ms <!-- @evidence[[results/summary.json#/mac/eg2_cut_near32_ms]] --> and Ettin-17m 0.667 ms <!-- @evidence[[results/summary.json#/mac/ettin_near32_ms]] -->. Documents: 1,161,965 tokens/s on 1.5M <!-- @evidence[[results/summary.json#/mac/1.5M_tokens_per_s]] -->
and 174,497 on 27M <!-- @evidence[[results/summary.json#/mac/27M_tokens_per_s]] -->. An iPhone was 6× slower per query and 16× slower on documents.

Queries closer to the weak sets (multi-hop, duplicate-question, argument) could change the distillation result.

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
8. **Distilling 1.5M queries into 115M's space reaches 0.468 and beats 1.5M alone only near its training data.**
   Student initialised from 1.5M, trained 8,000 steps × 256 MS MARCO and NQ queries to match 115M's query
   token vectors (cosine loss, final 0.203 <!-- @evidence[[results/summary.json#/distill/final_loss]] -->). It plateaus by step 6,000 (0.470 <!-- @evidence[[results/summary.json#/distill/step6000]] -->). MS MARCO 0.628 <!-- @evidence[[results/summary.json#/distill/MSMARCO]] --> beats 1.5M
   alone (0.593 <!-- @evidence[[results/summary.json#/distill/self15_MSMARCO]] -->); HotpotQA 0.344 <!-- @evidence[[results/summary.json#/distill/HotpotQA]] --> (1.5M alone 0.871 <!-- @evidence[[results/summary.json#/distill/self15_HotpotQA]] -->) and Quora 0.611 <!-- @evidence[[results/summary.json#/distill/QuoraRetrieval]] --> (0.9 <!-- @evidence[[results/summary.json#/distill/self15_QuoraRetrieval]] -->) collapse.
   One L4 run took 38.5 minutes <!-- @evidence[[results/summary.json#/distill/total_min]] -->, 7.6 of them encoding NanoBEIR with 115M.
9. **On the project's Bluesky task 1.5M is the weakest model tried.** hit@10 on 1,498 synthetic
   queries over 61,686 posts: 1.5M 0.82 <!-- @evidence[[results/summary.json#/bsky/1.5M_hit10]] -->, 27M 0.879 <!-- @evidence[[results/summary.json#/bsky/27M_hit10]] -->, against EG2 0.918, the EG2 4-layer cut 0.907,
   stock leaf-ir 0.889 and tuned Ettin-17m 0.918 (`../ettin-native/RESULTS.md`). Its exact fp16
   token index takes 10,576 bytes per post <!-- @evidence[[results/summary.json#/bsky/bytes_per_doc]] -->, against 192–264 for the single-vector models' 2-bit codes.
9. **A pipeline trap in sentence-transformers 6.2.** `encode_query` inserts the `[Q]` marker
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

### Round 3 (2026-10-26): distilling a 1.5M query encoder into 115M's space

Oskar asked what it would take to train 1.5M against 115M's document tokens. `run/distill.py`: the
student is every 1.5M weight, the teacher 115M's query encoder (frozen). The two share token ids, so the
loss is 1 − cosine per query position, all 32 rows. 1,110,793 MS MARCO and Natural Questions queries
with the 649 NanoBEIR query texts removed; AdamW lr 1e-3, cosine schedule, batch 256, 8,000 steps
(about 2 passes). Eval every 2,000 steps: student queries against 115M documents, all 13 sets.

The first run lost its VM after a 24-minute gap between `colab exec` calls; its step-2000 eval was
0.389. The second ran with a `colab exec` and `colab download` every 4 minutes and finished.

| step | 0 | 2000 | 4000 | 6000 | 8000 |
|---|---:|---:|---:|---:|---:|
| nDCG@10 | 0.0035 | 0.389 | 0.458 | 0.470 | 0.468 |
| loss (1 − cos) | | 0.252 | 0.218 | 0.205 | 0.203 |

Per set at step 8000, against 1.5M alone: MS MARCO 0.628 / 0.593, NQ 0.637 / 0.628, FEVER 0.798 / 0.904,
HotpotQA 0.344 / 0.871, Quora 0.611 / 0.900, ArguAna 0.273 / 0.448. The student matches or beats 1.5M
where the training queries look like the test queries and falls far short elsewhere. Mean token cosine
to the teacher stops at about 0.80, which points at the query mix or the student's capacity, not the
step count. Not tried: queries generated from the eval corpora's domains, a MaxSim score loss against 115M
documents, more layers.

### Round 4 (2026-10-26): the Bluesky task

Oskar asked how 1.5M compares with the models the project has tried. Those were all scored on
eg2-native-small's Bluesky task, not NanoBEIR, so `run/bsky.py` runs MoEdernColBERT on it:
61,686 AI-list posts re-fetched with `eg2-query-student/run/hydrate_posts.py` (12 since deleted,
scored as empty), documents truncated to 128 tokens, the 1,498 synthetic test queries, exact MaxSim
over fp32 token vectors, stock checkpoints. The other rows come from their own writeups and use
2-bit remex codes, which cost them about 0.004 against fp32 (leaf-ir 0.893 vs 0.889).

| model | hit@10 | hit@1 | weights MB | index bytes/post | Mac query ms |
|---|---:|---:|---:|---:|---:|
| MoEdernColBERT 1.5M | 0.820 [0.800, 0.840] | 0.668 | 5.9 (f32) | 10,576 (fp16) | 0.46 |
| MoEdernColBERT 27M | 0.879 [0.861, 0.895] | 0.724 | 108.5 (f32) | 10,576 (fp16) | 0.96 |
| MiniLM (stock) | 0.823 | | 23.7 | <300 (2-bit) | |
| leaf-ir (stock) | 0.889 | 0.732 | 23.9 | 200 | |
| EG2 4-layer cut, EG2 index | 0.907 | | 25.4 | 192 | 1.00 |
| Ettin-17m (distilled, tuned in-domain) | 0.918 | | 10.2 | <300 (2-bit) | 0.67 |
| EG2 q4 | 0.918 | | ~180 | 192 | |

Posts average 41 tokens. 1.5M encoded them at 1,262 posts/s on 4 CPUs, and exact scoring took
about 5 minutes per model there.
