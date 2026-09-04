# Static (Model2Vec) code embeddings on mini-CTXBench

Run 2026-09-03 in CCotw, 4 vCPU / 15 GB, on the `bekko-embedding-bench`
harness (n=59 scikit-learn issues, gold = the fix PR's file set, recall@5 and
recall@10 over files, `rg` identifier baseline, RRF fusion k=60). The question
came out of evaluating [zvec-grep](https://github.com/zvec-ai/zvec-grep), whose
default local embedder is `minishlab/potion-code-16M-v2`, a Model2Vec static
table: does a static embedder, vanilla or adapted to the corpus, do the job a
transformer encoder does in `xr`, at a fraction of the cost?

**Headline.** No. Vanilla potion-code scores dense r@5 **0.436** where `rg`
scores 0.596 and the bekko-a25m transformer scores 0.651 on the same tree and
queries (p=0.001 against rg, 5 wins to 24; p<0.001 against bekko, 2 wins to
26). None of five cheap adaptations moved it: fine-tuning on 4.2k docstring
pairs, extending the vocabulary with 12k corpus identifiers, both together,
and two distillations from bekko-a25m itself, with and without the same
fine-tune. Fusing a static arm with `rg`
makes `rg` worse, not better. The encode is 6 s for 11,439 chunks against
bekko's 20.6 min.

## Arms

All arms use the `ast` chunking (function/class scoped, 11,439 chunks over 694
files) and the same `rg` ranking. Dense = best chunk per file by cosine.

| arm | what it is | dense r@5 | dense r@10 | RRF r@5 | RRF r@10 | encode |
|---|---|---|---|---|---|---|
| `rg` | identifier grep, files ranked by distinct identifiers matched | 0.596 | 0.682 | | | |
| bekko-a25m | 25M-param ModernBERT ONNX, 384-d, the `xr` encoder | **0.651** | **0.706** | **0.644** | **0.733** | 20.6 min |
| potion-code-16M-v2 | Model2Vec static, 63,457 x 256 fp16, as shipped | 0.436 | 0.524 | 0.537 | 0.663 | 6 s |
| potion-retrieval-32M | Model2Vec static, general English retrieval, 512-d | 0.431 | 0.500 | 0.538 | 0.653 | 8 s |
| potion-code + ft | MNRL on 4,223 docstring->code pairs, 8 epochs | 0.435 | 0.546 | 0.567 | 0.643 | 6 s |
| potion-code + vocab (sum) | + 12,276 corpus words, rows = sum of subword rows | 0.439 | 0.530 | 0.539 | 0.647 | 10 s |
| potion-code + vocab (sum) + ft | the above, then MNRL | 0.418 | 0.563 | 0.573 | 0.663 | 9 s |
| potion-code + vocab (mean) | + 12,276 words, rows = mean of subword rows | 0.334 | 0.367 | 0.491 | 0.568 | 9 s |
| potion-code + vocab (mean) + ft | the above, then MNRL | 0.311 | 0.412 | 0.519 | 0.577 | 9 s |
| distill bekko (own) | teacher pass over 23,615 corpus tokens, PCA 256, SIF | 0.249 | 0.278 | 0.450 | 0.527 | 15 s |
| distill bekko (own) + ft | the above, then MNRL | 0.350 | 0.421 | 0.529 | 0.609 | 14 s |
| distill bekko (model2vec) | `model2vec.distill.distill`, 268,850 rows | 0.052 | 0.059 | 0.273 | 0.334 | 12 s |
| distill bekko (model2vec) + ft | the above, then MNRL | 0.135 | 0.155 | 0.311 | 0.366 | 12 s |

`rg` reproduces the 2026-08-04 numbers exactly (0.596 / 0.682) on today's
scikit-learn main (757f537), and bekko-a25m lands within one issue of its
August score (0.651 vs 0.656), so the tree drift between runs is negligible.

## Paired tests at n=59

Sign test on per-issue recall differences; paired bootstrap 95% CI on the
mean difference. Full table from `scripts/significance.py`.

| comparison (dense r@5) | Δ | 95% CI | w/l | p |
|---|---|---|---|---|
| potion-code vs rg | −0.159 | [−0.258, −0.064] | 5/24 | 0.001 |
| potion-code + ft vs potion-code | −0.001 | [−0.059, +0.052] | 9/6 | 0.607 |
| potion-code + vocab (sum) vs potion-code | +0.002 | [−0.017, +0.025] | 2/2 | 1.000 |
| potion-code + vocab (sum) + ft vs potion-code | −0.019 | [−0.076, +0.035] | 7/6 | 1.000 |
| potion-code + vocab (mean) vs potion-code | −0.102 | [−0.174, −0.034] | 3/13 | 0.021 |
| distill bekko (own) + ft vs potion-code | −0.086 | [−0.176, +0.005] | 6/20 | 0.009 |
| bekko-a25m vs potion-code | **+0.214** | [+0.134, +0.300] | 26/2 | <0.001 |
| bekko-a25m vs rg | +0.055 | [−0.047, +0.156] | 14/10 | 0.541 |
| potion-code RRF vs rg | −0.059 | [−0.131, +0.009] | 7/15 | 0.134 |
| potion-code + vocab (sum) + ft RRF vs rg | −0.023 | [−0.085, +0.035] | 7/11 | 0.481 |

The only comparisons that clear 0.05 are the ones where a static arm loses.
No adaptation is distinguishable from vanilla potion-code except the
mean-initialised vocabulary extension, which is distinguishable because it is
worse.

## Per-adaptation results

**Docstring fine-tune.** Validation MNRL loss fell from 1.64 to 1.14 over 8
epochs (11 s per epoch on CPU), so the table did learn the docstring->code
task. Retrieval did not move: r@5 0.436 -> 0.435. The training pairs are
prose that describes a function, and the test queries are prose that
describes a bug. A static bag of subwords has no mechanism to carry the one
into the other beyond shared vocabulary, and the shared vocabulary was already
in the table.

**Vocabulary extension, mean-initialised.** The intuitive way to add
`predict_proba` as one row is to initialise it as the mean of its pieces. In
a mean-pooled model that halves or fifths the word's contribution to the
pooled vector, because it now counts as one token instead of three or five.
Long, distinctive identifiers are exactly the tokens that get down-weighted.
r@5 0.334, p=0.021 against vanilla. Model2Vec's own recipe avoids this by
re-encoding the whole word through the teacher.

**Vocabulary extension, sum-initialised.** Initialising the new row as the
sum of its pieces preserves each word's pooled contribution, and epoch 0
reproduces vanilla (0.439 vs 0.436; cosine 1.000 per word, 0.94 mean over
whole queries only because the hybrid tokenizer's 512-token cap counts whole
words as one token). Fine-tuning from there gives the whole-word rows freedom
the subword rows do not have. It bought r@10 +0.039 and cost r@5 −0.019,
neither significant.

**Distillation from bekko-a25m, hand-rolled.** 23,615 vocabulary entries
(11,339 bekko subword pieces in corpus use plus the 12,276 whole words), each
encoded alone by the teacher in 40 s, PCA to 256 (92.4% variance kept), SIF
weighting a/(a+p) with a=1e-4 from corpus frequency. r@5 0.249. MNRL on top
recovers to 0.350, still below vanilla potion. Single-token forward passes
through a ModernBERT retrieval model are a poor proxy for how that model
represents tokens in context; the recipe works for potion because the
distilled table is then trained further (tokenlearn, then 1.2M-pair
contrastive stages for the code model), and those stages are where the
quality comes from.

**Distillation from bekko-a25m, model2vec library.** `distill()` over the
teacher's full 256k vocabulary plus our 12k words, 562 s, 268,850 x 256. r@5
0.052, which is close to random over 694 files. The library's Zipf weighting
assumes the vocabulary is frequency-ordered, and a BPE vocabulary appended
with corpus words is not; whatever the cause, the out-of-the-box distillation
of this teacher is unusable without the training stages. Eight epochs of MNRL
lift it to 0.135 (validation loss 5.4 -> 2.6, still far above potion's 1.14).

## The zvec-grep default route

zvec-grep fuses its vector route with BM25 by RRF, and the vector route in its
default configuration is the potion-code table scored above. On this
benchmark that route drags a lexical baseline down (RRF r@5 0.537 vs rg 0.596,
p=0.13). The bekko-a25m route lifts it (0.644, and 0.733 at r@10, the best
cell in the table). zvec-grep's published SWE-QA gains were measured with a
remote Qwen3.7 embedding. Those gains say nothing about the potion route.

The static encoder's advantage is real and narrow: 6 s to index 11k chunks on
first run, no model download beyond 32 MB, and a per-query cost that is a
table lookup. For an in-session tool that has to index before its first
answer, that matters. The route it buys scores below `rg` alone and pulls
`rg` down when fused.

## Caveats

- n=59, identifier-rich issues. The identifier-poor stratum is n=1 of 59 in
  this instance set, and it is the stratum where a corpus vocabulary would
  have its claim. This experiment cannot see it.
- 4,223 training pairs is small. The HuggingFace static-embedding recipe used
  3.5B pairs and 17.8 GPU-hours to reach NanoBEIR 0.50 from random init. A
  corpus of this size cannot supply that.
- One learning rate (2e-3, AdamW, batch 128, scale 20) for every fine-tune.
  The distilled tables have row norms of 0.75 and 7.9 against potion's 9.0,
  so the effective step differs by arm. Train loss on the distilled arm hit
  0.02 while validation stayed at 1.23: overfit, and a lower rate might do
  better. It would have to close a 0.09 gap to reach vanilla and a 0.30 gap
  to reach the transformer.
- `ast` chunking only. The August run found `flat` chunking slightly better
  for bekko (0.658 vs 0.595 at a8m, not significant); static arms were not
  run on it.

## Files

- `scripts/common.py` — harness glue, `HybridTok` (whole-word vocabulary
  first, subword fallback, honours the tokenizer's 512 cap), `StaticTable`
- `scripts/pairs.py` — docstring->code pair mining, held out by file
- `scripts/mine_vocab.py` — corpus identifiers with document frequency >= 3
- `scripts/train_static.py` — MNRL fine-tune, `--extend-vocab`, `--init-mode sum|mean`
- `scripts/distill_bekko.py` — hand-rolled Model2Vec distillation from the ONNX teacher
- `scripts/distill_m2v.py` — the library's `distill()` from the HF checkpoint
- `scripts/run_bench.py` — scores arms into `results.json`
- `scripts/significance.py` — paired sign test and bootstrap over `results.json`
- `results.json` — every per-issue row for every arm
- `data/vocab_words.json`, `data/vocab_df.json` — the mined vocabulary

Models (`models/`, ~1 GB) and the scikit-learn clone are not committed;
`pairs.py`, `mine_vocab.py` and the training scripts rebuild them in under
ten minutes from the `bekko-embedding-bench` chunks.
