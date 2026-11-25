# MSD catalog tokens for SparseUp v2

Question (Oskar, 2026-11-25): can `Linkup-Platform/linkup-sparseup-embed-v2` be fine-tuned to Meso Scale
Discovery's assay catalog, by giving the catalog's terms their own vocabulary entries, so that it finds the
PubMed abstracts about a catalog analyte better than the stock model?

## Answer

Yes. Adding one vocabulary row per catalog analyte (696 rows, 50,370 → 51,066) and training in two stages
lifts R@10 on the MSD retrieval test from 0.673 to 0.800 (+0.127, 95% CI +0.091 to +0.165) and MRR from
0.929 to 0.969, with general retrieval unchanged within noise (SciFact nDCG@10 0.744 against 0.748,
difference −0.004, CI −0.019 to +0.011). Gene-symbol queries gain most (0.58 → 0.86), catalog names gain
0.09, MeSH-heading queries 0.70 → 0.73 (CI spans 0). The tuned model alone beats the earlier tag-vector
dictionary (0.746) and fuses with it and TF-IDF to 0.822.

What makes it work: every spelling of an analyte (`IL-1β`, `IL1B`, `interleukin-1 beta`) is rewritten to one
new token before encoding, the new rows are trained first on their own, and the full fine-tune is anchored to
the original model's vectors. Without that anchor the model collapses its queries to 2–5 words and gives up
0.06–0.08 on SciFact for 0.01 more MSD R@10.

What would change it: the test labels and training pairs both come from MeSH indexing (different papers), so
this measures learning the catalog, not generalising to analytes outside it.

## Findings

1. **Anchored stage 2 is the candidate**: with anchor weight 4 and 4,000 steps, MSD R@10 0.800, MRR 0.969, SciFact
   0.744 (−0.004, CI spans 0); abstracts keep 235 active terms, queries 13 (`results/colab5.json`, run colab5). Anchor
   weight 2 and 2,500 steps gave 0.790 / 0.738 (`results/colab4.json`): a stronger anchor trained longer is better
   on both sides.
2. **Stage 1 alone (only the 696 new rows train, the model frozen) gets most of the way**: 0.751 (+0.077 [+0.037,
   +0.118]), SciFact −0.011 (CI spans 0). Gene symbols 0.58 → 0.82; MeSH −0.04 (n.s.) (`results/colab2.json`, s1).
3. **Stage 1 saturates**: 6,000 steps instead of 2,000 give 0.758 (`results/kaggle2.json`).
4. **An unanchored full fine-tune collapses the vectors**: with the sparsity penalty, 2.3 terms per query, MSD 0.805,
   SciFact 0.664 (−0.084); without it, 4.9 terms, MSD 0.796, SciFact 0.693 (−0.055) (`results/colab2.json` s2,
   `results/colab3.json` s2). The contrastive task is solvable by the concept token alone, so expansions die.
5. **New rows before training hurt**: initialised as the mean of their name's pieces and not trained, MSD drops to
   0.449 (`results/colab2.json`, s0). The token rewrite only pays once the rows are learned.
6. **Baselines on the same fixture** (re-run, `../sparseup-v2-msd-probe/tag-vector/results/sparseup_test.json`):
   tag vector 0.746, BM25 0.616, TF-IDF 0.646, stock SparseUp 0.673–0.683 (GPU fp16 vs CPU fp32).

| model (MSD test, R@10) | catalog | symbol | MeSH | all | MRR | +tag+TF-IDF | SciFact |
|---|---|---|---|---|---|---|---|
| stock SparseUp | 0.726 | 0.584 | 0.697 | 0.673 | 0.929 | 0.767 | 0.748 |
| new rows, untrained | 0.435 | 0.444 | 0.466 | 0.449 | 0.832 | 0.756 | 0.741 |
| stage 1 (2k steps) | 0.780 | 0.822 | 0.661 | 0.751 | 0.950 | 0.807 | 0.736 |
| stage 1 (6k steps) | 0.795 | 0.846 | 0.646 | 0.758 | 0.931 | 0.809 | 0.739 |
| stage 2, sparsity penalty | 0.832 | 0.883 | 0.711 | 0.805 | 0.936 | 0.778 | 0.664 |
| stage 2, no penalty | 0.823 | 0.877 | 0.699 | 0.796 | 0.941 | 0.816 | 0.693 |
| stage 2, anchored ×2, 2.5k steps | 0.811 | 0.861 | 0.707 | 0.790 | 0.960 | 0.817 | 0.738 |
| **stage 2, anchored ×4, 4k steps** | 0.814 | 0.864 | **0.732** | **0.800** | **0.969** | **0.822** | **0.744** |

## Method

- **Vocabulary.** The 696 concepts of `../sparseup-v2-msd-probe/tag-vector/data/vocab.json` (MSD catalog names
  plus HGNC symbols, names and aliases; 4,055 match keys). `msd_vocab.py` tags text with the same greedy
  longest-match rules as the tag-vector study, and replaces every token run a tagged span covers with that
  concept's single new token, in queries and documents alike. The token embedding table is tied to the output
  head, so each new row is both the input token and an output dimension. Rows start as the mean of the
  catalog name's pieces; the decoder bias likewise.
- **Training data** (`build_corpus.py`, `data/train_pmids.tsv`): PubMed abstracts that name each concept in
  title or abstract (E-utilities, English, 2010 on, up to 80 per concept): 47,213 abstracts over 676 concepts.
  Every evaluation PMID is excluded. Queries: a concept's catalog name or HGNC symbol, its MeSH heading for
  concepts outside the test split, or the abstract's title (a quarter of the time). In-batch negatives, with
  other abstracts carrying the query's concept masked out.
- **Stage 1**: only the new rows and their biases train (gradients on old rows zeroed), AdamW 2e-3,
  2,000 steps of 16 pairs. **Stage 2**: everything trains (2e-5 or 1e-5; new rows 2e-4), the original model
  runs frozen as teacher on the raw text. Losses: cross-entropy over dot-product scores, KL to the teacher's
  in-batch score distribution, and, in the anchored run, 2 × (1 − cosine) between the student's and the
  teacher's query and document vectors over the original 50,370 dimensions, weighted 2 (2,500 steps) or 4
  (4,000 steps, the candidate).
- **Evaluation**: the frozen `evaluate.py` of the tag-vector study (91 test concepts, 260 queries, 1,241
  abstracts; labels from NLM's MeSH indexing), run by `score.py`; SciFact test (300 queries, 5,183 abstracts)
  for general retrieval. Encodes in fp16 on the GPU. 95% bootstrap CIs over queries, paired against the stock
  model encoded the same way.
- **Compute**: Colab T4, L4 and A100 and a Kaggle T4; each job rebuilds its inputs on the box
  (`kaggle_job.py`). Stage 1 takes 25 min on a T4, anchored stage 2 16 min on an A100.

## Log

### 2026-11-25

Built the concept-token rewrite and the stage 0/1/2 trainer; CPU smoke test; first GPU run ran out of memory
encoding 32 × 512-token abstracts (the head materialises 51k logits per token) and now batches by tokens.
Stage 0 lost 0.22; stage 1 gained 0.08 and saturated by 6k steps. Two unanchored stage-2 runs reached 0.80 on
MSD but collapsed the vectors and lost 0.06–0.08 on SciFact; the cosine anchor to the teacher's vectors kept
SciFact level at 0.790 on MSD. Anchor weight 4 with 4,000 steps then reached 0.800 on MSD and 0.744 on SciFact,
the best on both. Its weights (fp16 state dict, 379 MB in eight parts), the stage-1 rows, `vocab.json`, the
rewrite code, a README and SHA256SUMS are on R2: https://pub-38421218e82c4d4ba2bc53974c28b55d.r2.dev/sparseup-v2/msd-vocab-v1/README.md
(bucket austegard-media, public, Apache-2.0 as the base model).
