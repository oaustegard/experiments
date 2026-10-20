# RESULTS: a small query encoder in EmbeddingGemma 2's space

Issue: [#102](https://github.com/oaustegard/experiments/issues/102). Run 2026-10-19 to
2026-10-20 in a 4-vCPU CCotw container, with one GPU pass in Oskar's browser (Apple Metal).
What broke is in [`ERRORS.md`](ERRORS.md). This file has the answer, the findings behind it,
the method, and the log; only the first three are maintained.

## Answer

Can documents be indexed with EmbeddingGemma 2 (EG2) and queries embedded by a small model
in the browser? Not by a distilled student. The best distilled 24 MB student agrees with EG2 on
0.482 of the top 10 for real queries <!-- @evidence[[results/final.json#/A3 leaf_cont165k epoch0 int8/real_test/overlap@10/0]] -->
and finds a synthetic query's source post 79.4% of the time <!-- @evidence[[results/final.json#/A3 leaf_cont165k epoch0 int8/synth_test/hit@10/0]] -->
against EG2's own 91.9%. <!-- @evidence[[results/final.json#/EG2 (teacher)/synth_test/hit@10/0]] -->
Three times the in-domain training data moved real-query agreement by 0.001. <!-- @evidence[[results/final.json#paired.data_vs_epochs.real_test.0]] -->

EG2 with its vocabulary cut to this corpus's 32,000 commonest tokens
agrees on 0.932 of the top 10 <!-- @evidence[[results/final.json#/A4 pruned EG2 K=32000/real_test/overlap@10/0]] -->
with a hit@10 change of -0.002, <!-- @evidence[[results/final.json#/A4 pruned EG2 K=32000/synth_test/hit@10_minus_eg2/0]] -->
but downloads 102.9 MB against 179.9 MB, <!-- @evidence[[results/sizes.json#/A4 pruned EG2 K=32000/mb]] -->
since the transformer is the other half, and runs at EG2's speed.

What would change this: a student initialised from EG2's own layers, the
[published recipe](https://arxiv.org/abs/2306.11550) that keeps 92.5% of task quality.

## Findings

1. **Distilled students plateau near half of EG2's top 10.** leaf-mt (23M) after 6 epochs
   on 54K queries: 0.462 real, int8; <!-- @evidence[[results/final.json#/A3 mdbr-leaf-mt epoch5 int8/real_test/overlap@10/0]] -->
   4 more epochs on the same data: 0.481; <!-- @evidence[[results/final.json#/A3 leaf_cont54k epoch3 int8/real_test/overlap@10/0]] -->
   one epoch over 165K instead: 0.482. <!-- @evidence[[results/final.json#/A3 leaf_cont165k epoch0 int8/real_test/overlap@10/0]] -->
   Paired difference of the last two on real queries 0.001 [−0.004, +0.006]; <!-- @evidence[[results/final.json#paired.data_vs_epochs.real_test.0]] -->
   on synthetic queries +0.010 [+0.006, +0.014]. <!-- @evidence[[results/final.json#paired.data_vs_epochs.synth_test.0]] -->
   (Round 2, Round 4)
2. **At task level the student costs 12.5 points of hit@10.** Its hit@10 change against
   EG2 on held-out synthetic queries is -0.125; <!-- @evidence[[results/final.json#/A3 leaf_cont165k epoch0 int8/synth_test/hit@10_minus_eg2/0]] -->
   the linear map's is -0.208. <!-- @evidence[[results/final.json#/A1 leaf-mt ridge/synth_test/hit@10_minus_eg2/0]] -->
   (`run/final_eval.py`)
3. **Vocabulary pruning keeps EG2 exactly where the tokenization survives.** At K=32,000,
   queries tokenized identically by the pruned and full vocabularies return EG2's top 10
   exactly (cosine ≥ 0.9999998); every loss comes from words the cut splits differently.
   Top-10 agreement by K: 0.770, 0.851, 0.932, 0.970 for 8K, 16K, 32K, 64K. <!-- @evidence[[results/final.json#/A4 pruned EG2 K=64000/real_test/overlap@10/0]] -->
   (Round 3)
4. **Top-10 agreement is far stricter than task quality.** Pruned to 8K tokens, EG2 agrees
   on 0.770 of the top 10 with a hit@10 change of only -0.009; <!-- @evidence[[results/final.json#/A4 pruned EG2 K=8000/synth_test/hit@10_minus_eg2/0]] -->
   the students lose 0.12 or more at every agreement level they reach. (`run/final_eval.py`)
5. **The vocabulary is half of EG2's q4 file, so pruning caps out near 1.7x smaller.**
   `embed_tokens` is 86 MB of 174 MB (262,144 rows); the transformer stays. K=16,000
   downloads 96.5 MB with 0.851 agreement. <!-- @evidence[[results/sizes.json#/A4 pruned EG2 K=16000/mb]] -->
   (Round 3)
6. **A linear map is a weak lower bound, and its error is structured.** Best ridge map
   (leaf-mt, fit on synthetic queries) 0.354 real. <!-- @evidence[[results/final.json#/A1 leaf-mt ridge/real_test/overlap@10/0]] -->
   Isotropic noise at the same cosine to EG2 (about 0.89) keeps 0.70 agreement, so the
   map's error lies along the directions the corpus ranking depends on. (Round 1)
7. **The base model matters more than the data.** At equal training,
   leaf-mt beats MiniLM-L6 by 0.045 on real queries, paired. <!-- @evidence[[results/final.json#paired.leaf_vs_minilm.real_test.0]] --> In-domain synthetic queries beat 30K MS MARCO queries for the linear map even on
   MS MARCO-style test queries. (Round 1, Round 2)

## Method

**Corpus.** The 61,686 Bluesky posts of the `embeddinggemma2-webgpu-tail` search page
(members of Oskar's "Artificial (and real) Intelligence" list), indexed there with EG2 q4
into remex 2-bit codes (`Quantizer(768, bits=2, seed=42, mean=mu)`). The committed codes
are the document side for every arm; no arm re-embeds a document. A batch scorer that
reproduces remex `search_adc` is checked against that page's 8 parity queries (exact top 10).
Post text was re-fetched from Bluesky (`run/hydrate_posts.py`, 0 of 61,686 missing) and is
kept out of git.

**Queries.** Real: 1,000 MS MARCO queries (all 238 matching an AI regex plus 762 matching a
tech/science regex), held out by topic: no query matching either regex is in any training
set. Synthetic: Haiku wrote one search query per post (`run/gen_synth.py`); posts split once
into a 24,000-post training pool and a disjoint 1,500-post test pool (1,498 test queries).
Training: 30,000 random non-tech MS MARCO queries, 23,985 synthetic, and in the GPU round
55,289 more synthetic queries and 55,314 post snippets (first ~200 characters).

**Teacher.** EG2 q4 ONNX, query prompt `task: search result | query: `, max 128 tokens: on
CPU onnxruntime for 55K queries, and on WebGPU in Oskar's browser for the 110,605 GPU-round
texts (`web/`; 256-query parity min cosine 0.999992; <!-- @evidence[[results/gpu_run.json#parity.min_cos]] -->
int8 shards, spot-checked at cosine ≥ 0.99975 against CPU).

**Arms.** A1: ridge from a small model's query vector to EG2's 768-d (MiniLM-L3, MiniLM-L6,
bge-small, leaf-mt; lambda on a 10% split). A3: the small model plus a Linear(d→768) head
initialised from A1, all weights trained on 1 − cos(student, EG2), lr 5e-5 then 3e-5
for continuations, batch 64; shipped as dynamic int8 ONNX. A4: EG2 with `embed_tokens`
sliced to the top-K tokens by frequency over posts and training queries, closed under the
BPE merges that build them, with a pruned `tokenizer.json` (merges producing a cut token
dropped, byte fallback on).

**Metrics.** Top-10 overlap: the share of EG2-query top 10 on the 2-bit codes that the arm's
query also returns, per query, bootstrap 95% CI (2,000 resamples). hit@10: whether a
synthetic test query's source post is in the top 10, with the paired difference to EG2.
Download: ONNX bytes plus gzip tokenizer (students: int8 ONNX plus `tokenizer.json`).

## Log

### Round 1: setup and A1 (2026-10-19)

Located the corpus via `xr` (`embeddinggemma2-webgpu-tail/web/bsky`); embstore held only
Quora EG2 vectors. Hydrated post text, built the MS MARCO sets, embedded teacher targets on
CPU (~40 queries/s), and generated synthetic queries ($1.65, 510 Haiku children).

A1 on real / synthetic test, by training set:

| model | MS MARCO 30K | synthetic 24K | both |
|---|---|---|---|
| MiniLM-L3 | 0.130 / 0.049 | 0.173 / 0.221 | 0.187 / 0.175 |
| MiniLM-L6 | 0.190 / 0.109 | 0.250 / 0.298 | 0.262 / 0.255 |
| bge-small | 0.207 / 0.124 | 0.286 / 0.318 | 0.278 / 0.267 |
| leaf-mt | — | 0.354 / 0.409 | 0.364 / 0.364 |

Calibration on EG2 itself: the document prompt instead of the query prompt keeps 0.745 at
cosine 0.976; lowercasing keeps 0.995; isotropic noise at cosine 0.995 / 0.981 / 0.958 /
0.895 keeps 0.930 / 0.872 / 0.811 / 0.704.

### Round 2: A3 (2026-10-19 to 10-20)

MiniLM-L6, 6 epochs on MS MARCO + synthetic (54K): 0.209 at init, 0.345, 0.384, 0.403,
0.416, 0.425, 0.425 real; int8 0.417 at 23.3 MB. leaf-mt, same: 0.364 at init, then 0.402,
0.430, 0.446, 0.459, 0.467, 0.474; int8 0.462 at 24.2 MB. A ranking-loss variant (KL over
each batch's teacher top-50 plus 512 random posts) was abandoned after one epoch (ERRORS 4).

### Round 3: A4 (2026-10-20)

| K | vocab after closure | ONNX + tok.gz | real | synthetic | queries tokenized intact |
|---|---|---|---|---|---|
| 8,000 | 13,010 | 92.8 + 0.19 MB | 0.770 | 0.794 | 0.439 |
| 16,000 | 23,289 | 96.2 + 0.36 MB | 0.851 | 0.881 | 0.614 |
| 32,000 | 41,741 | 102.2 + 0.70 MB | 0.932 | 0.947 | 0.841 |
| 64,000 | 75,267 | 113.2 + 1.34 MB | 0.970 | 0.983 | 0.943 |

Full EG2: 174.5 MB ONNX + 5.4 MB gzip tokenizer; 77,654 distinct tokens appear in the posts
and training queries.

### Round 4: GPU-built targets (2026-10-20)

Oskar asked for the browser GPU to do the builds. `web/index.html` embedded 110,605 texts on
Apple Metal in 37 minutes (queries ~89/s, snippets ~36/s; CPU here ~40 and ~10/s), parity min
cosine 0.999992. GitHub's web uploader refused the shards; they came in as release
`eg2-targets-20261020` (ERRORS 2). From the 6-epoch leaf-mt checkpoint: one epoch over 165K
(2,572 steps) reached 0.489 real / 0.518 synthetic at fp32; the control, 4 epochs over the
original 54K (3,376 steps), 0.490 / 0.511. Data helped the synthetic queries it resembles and
did nothing for real ones.

### Final table (int8 students, `run/final_eval.py`)

| arm | download | real overlap | synthetic overlap | hit@10 | vs EG2 |
|---|---|---|---|---|---|
| EG2 | 179.9 MB | 1 | 1 | 0.919 | — |
| A1 ridge, leaf-mt | (student) | 0.354 | 0.409 | 0.711 | −0.208 |
| A3 MiniLM-L6 | 23.5 MB | 0.417 | 0.424 | 0.698 | −0.221 |
| A3 leaf-mt, 6 ep | 24.4 MB | 0.462 | 0.484 | 0.774 | −0.145 |
| A3 leaf-mt, +4 ep 54K | 24.4 MB | 0.481 | 0.502 | 0.791 | −0.128 |
| A3 leaf-mt, +1 ep 165K | 24.4 MB | 0.482 | 0.513 | 0.794 | −0.125 |
| A4 K=8K | 93.0 MB | 0.770 | 0.794 | 0.910 | −0.009 |
| A4 K=16K | 96.5 MB | 0.851 | 0.881 | 0.915 | −0.004 |
| A4 K=32K | 102.9 MB | 0.932 | 0.947 | 0.917 | −0.002 |
| A4 K=64K | 114.6 MB | 0.970 | 0.983 | 0.918 | −0.001 |

Gzip of the int8 students is about 18.6 MB. Cost: Haiku $5.25 over 1,617 children; CPU
about 9 hours including the lost run.
