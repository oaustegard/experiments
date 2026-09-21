# msd-context-classifier — vocabulary adaptation

**Started / finished:** 2026-09-20, 4:25 PM – 10:30 PM Eastern · **Status:** done —
**two epochs of continued pretraining on the site teach a 32M encoder the site's
terms: whole-term recovery of domain vocabulary on held-out pages goes from 0.32
to 0.69, to 0.83 with whole-term masking, to 0.75 with 300 terms added to the
vocabulary. None of it reaches the tasks that matter. Mean-pooled retrieval stays
at zero before and after (gte-small, an embedding-trained model with a worse
tokenizer, scores 0.96 recall@10), and the adapted encoders classify within two
points of the stock one. The vocabulary was never the bottleneck; the training
objective was.**

**Question (Oskar).** "I was looking more for finetuning the vocabulary I guess.
There are loads of terms in the product listing that are unlikely to be
contained in the model corpus." Plan and six pre-registered predictions in
[`PLAN-vocab.md`](PLAN-vocab.md), committed before any adaptation run. Follows
[`RESULTS.md`](RESULTS.md) (the classifier round) on the same corpus.

## Tokenizer audit

`vocab_audit.py`, 887 sampled pages: 1,624 site-specific word types (digits,
hyphens, two capitals or nine-plus letters; absent from the wordfreq top-50k;
count ≥ 3). Pieces per term under each tokenizer, against ordinary words from
the same pages:

| tokenizer | vocab | pieces per domain term | ≥3 pieces | ordinary words |
|---|---|---|---|---|
| ModernBERT / Ettin BPE | 50,368 | 3.43 | 71% | 1.13 |
| gte-small WordPiece (uncased) | 30,522 | 3.96 | 85% | 1.08 |
| BiomedBERT PubMed WordPiece | 30,522 | 2.76 | 58% | 1.05 |

The most frequent terms are analyte names (IFN, IL-1, IL-6, IL-10, MIP-1, MCP-1,
IL-12p70, GM-CSF), product families (U-PLEX, R-PLEX, V-PLEX) and the company's
acronym; an interleukin is `IL` + `-` + digits under every tokenizer. Oskar's
premise holds: the terms are not in any pretraining vocabulary as units.

**Token-level masked-LM loss cannot see this.** With 15% random masking, domain-
term tokens are *easier* than ordinary tokens for every stock model (ettin-150m
CE 1.24 vs 1.60): masking one piece of `IL-6` leaves the other two as the answer
key. The diagnostic below masks every piece of a term at once and scores whether
the model recovers the whole term (`mlm_ppl.py`, one target per forward pass,
60 held-out pages, ~900 domain-term occurrences over 175 types, 900 ordinary
control words).

## Corpus

Full crawl of every labelled URL in the sitemap (5,256; 1,198 fetched earlier at
one request per second, the rest with six workers at half a second each after
Oskar asked for speed, 1.4 pages/s aggregate, 6:14 PM Eastern). 4,760 pages
kept after the 60-word filter, 2.0M words; 3,028 product pages, 1,420
bibliography pages. The 270 dev/test URLs of the sampled corpus are held out of
every adaptation run.

## Arms and recipe

ettin-32m, continued MLM, 15% masking, AdamW lr 5e-5, 256 tokens, batch 16,
2 epochs over 4,490 pages (281 steps/epoch, ~16 min/epoch on 4 vCPU):

| arm | change |
|---|---|
| dapt | plain continued pretraining |
| dapt+term | domain-term occurrences masked whole at p=0.5 on top of the 15% |
| dapt+vocab | 300 most frequent ≥3-piece domain terms added as tokens, embeddings initialised from the mean of their old pieces (decoder tied) |
| dapt+vocab+term | both, run under the plan's "expansion beat plain by more than 3" rule |

## Whole-term recovery

| model | domain-term exact | ordinary-word exact | pieces / term |
|---|---|---|---|
| ettin-32m stock | 0.324 | 0.386 | 3.0 |
| ettin-150m stock | 0.395 | 0.490 | 3.0 |
| ModernBERT-base stock | 0.389 | 0.336 | 3.0 |
| BioClinical-ModernBERT-base stock | 0.408 | 0.430 | 3.0 |
| ettin-32m dapt | 0.688 | 0.837 | 3.0 |
| ettin-32m dapt+term | **0.832** | 0.824 | 3.0 |
| ettin-32m dapt+vocab | 0.752 | 0.858 | 1.7 |
| ettin-32m dapt+vocab+term | 0.824 | 0.854 | 1.7 |

Plain continued pretraining adds 36 points on domain terms. Whole-term masking
adds 14 more and is the single best change. Vocabulary expansion adds 6 over
plain and nothing over term masking; its real effect is on length: 300 tokens
cut the mean product page from 697 to 659 tokens (5.4%). BioClinical
ModernBERT, after 53B tokens of PubMed and clinical continued pretraining, is
within two points of plain ModernBERT-base on this catalog's terms.

Ordinary words rose 45 points too (0.39 → 0.84). That is the template: the
held-out product pages share their boilerplate with the training pages, and two
epochs learn the boilerplate before they learn the analytes. Domain-term
recovery on *unseen page templates* would be lower than the table; this corpus
has no such split.

### Per term

Exact recovery per term type on the same 60 pages, stock ettin-32m against the
two term-masked arms (`results/mlm2_*.json`; the three navigation-junk "terms"
of ERRORS.md #7 excluded):

| term | occurrences | stock | dapt+term | dapt+vocab+term |
|---|---|---|---|---|
| R-PLEX | 86 | 0.49 | 0.96 | 0.98 |
| U-PLEX | 79 | 0.34 | 0.95 | 0.99 |
| V-PLEX | 44 | 0.46 | 0.89 | 0.93 |
| MSD | 41 | 0.22 | 0.88 | 0.88 |
| multiplexing | 21 | 0.00 | 0.95 | 0.95 |
| calibrator | 18 | 0.00 | 1.00 | 0.89 |
| single-analyte | 14 | 0.00 | 1.00 | 1.00 |
| singleplex | 13 | 0.08 | 1.00 | 0.92 |
| IL-6 | 10 | 0.60 | 1.00 | 0.91 |
| datasheet | 10 | 0.00 | 1.00 | 1.00 |
| IL-17A | 9 | 0.56 | 1.00 | 0.91 |
| GM-CSF | 8 | 0.62 | 1.00 | 1.00 |

Of 175 term types with three or more occurrences, 24 are never recovered by
the stock model and 8 by the term-masked one; the survivors are rare on the site
(`SULFO-TAG` 3, `JAK/STAT` 3, `BAFF-R/TNFRSF13C` 4, `SARS-CoV-2` 4). The
frequent interleukins were already half-known to the stock model from their
pieces; the words it did not have at all were the company's own (`multiplexing`,
`singleplex`, `single-analyte`, `calibrator` in this sense), and those are the
ones adaptation took from zero to one.

## Downstream: retrieval

147 product questions (one per product, written by a worker from the title
alone, a third carrying the catalog code; 87 target pages the adaptation never
saw). Index = all 3,028 product pages, title + text.

| retriever | recall@1 | recall@10 | MRR | unseen R@10 | code R@10 |
|---|---|---|---|---|---|
| BM25 (`bm25s`) | 0.306 | 0.762 | 0.454 | 0.782 | 0.727 |
| gte-small, mean-pool (embedding-trained, no domain) | **0.551** | **0.959** | **0.711** | 0.954 | 0.939 |
| ettin-32m stock, mean-pool | 0.007 | 0.048 | 0.021 | 0.034 | 0.030 |
| ettin-150m stock, mean-pool | 0.000 | 0.014 | 0.004 | 0.000 | 0.000 |
| ettin-32m dapt | 0.000 | 0.014 | 0.005 | 0.011 | 0.000 |
| ettin-32m dapt+term | 0.000 | 0.007 | 0.004 | 0.011 | 0.000 |
| ettin-32m dapt+vocab | 0.007 | 0.014 | 0.010 | 0.011 | 0.000 |
| ettin-32m dapt+vocab+term | 0.000 | 0.014 | 0.007 | — | — |

A masked-LM encoder's mean-pooled hidden state is not an embedding, before or
after adaptation: every MLM arm is at chance over 3,028 documents. gte-small
fragments the same terms into four pieces and retrieves 96% of targets in the
top ten, because it was trained to put a question and its answer near each
other. BM25 loses to it even on the questions carrying a catalog code (0.73 vs
0.94), which is worth knowing: the codes live in URLs and headers, not reliably
in page text. Caveat on the absolute numbers: the questions were generated from
per-category templates over the product titles, so their overlap with titles is
higher than real traffic would give; the ranking of the arms is not affected.

## Downstream: classification

The adapted encoders as site-section classifiers, the recipe of the previous
round's check arm (class-weighted cross-entropy, six epochs, `train.py`):

| encoder | pages macro-F1 | questions macro-F1 |
|---|---|---|
| ettin-32m stock | 0.828 | 0.139 |
| dapt | 0.809 | 0.146 |
| dapt+term | 0.823 | 0.165 |
| dapt+vocab | 0.815 | 0.133 |

Within two points of stock on both surfaces, in both directions. The page task
was template-limited and the question failure is distribution shift; neither
was a vocabulary problem, and knowing the vocabulary did not change either.

## Predictions scored

| # | prediction | outcome |
|---|---|---|
| V1 (75%) | dapt +≥15 on domain terms, ordinary words move <5 | half — domain +36 **right**; ordinary +45 **wrong** (template effect, above) |
| V2 (55%) | dapt+vocab within ±3 of dapt | **wrong** upward, +6; the rule fired and the combined arm ran: +0 over term masking alone |
| V3 (65%) | dapt+term ≥ +5 over dapt, no cost on ordinary words | **right** — +14, ordinary −1 |
| V4 (80%) | BioClinical-ModernBERT no better than ModernBERT-base here | **right** — 0.41 vs 0.39, a tie |
| V5 (65%) | dapt +≥10 recall@10; still loses to gte-small; BM25 wins on code questions | **wrong** on two of three — no retrieval gain at all (0.048 → 0.014); gte-small wins **right**; BM25 loses on code questions 0.73 vs 0.94 |
| V6 (70%) | 300 terms cut tokens per page ≥8% | **wrong** — 5.4% |

Two right, one half, three wrong. The misses share a shape with the previous
round's: I predicted the adaptation would show up downstream in proportion to
what it learned, and it showed up only on the diagnostic that measures the
learning itself.

## Recommendations

- **The site's terms are learnable in an afternoon.** Two epochs, 32M
  parameters, 4 vCPU, 33 minutes per arm: 0.32 → 0.83 whole-term recovery.
  Whole-term masking of the domain vocabulary is the change to keep; add it to
  any continued-pretraining recipe over catalog text.
- **Vocabulary expansion is a length optimisation.** 300 added tokens: +6 over
  plain adaptation on recovery, +0 over term masking, −5% tokens per page. Worth
  it for throughput on long pages; it adds no knowledge.
- **Neither reaches a task.** The masked-LM objective produces a model that
  predicts the site's words and does not produce an embedding or a better
  classifier. For retrieval over this catalog the next experiment is
  contrastive fine-tuning on (question, page) pairs, starting from gte-small
  and from dapt+term ettin-32m in parallel, and the number to read is
  whether the domain-adapted start wins once both have been trained to embed.
  For classification the previous round's conclusion stands: change the
  training distribution; the encoder is already good enough.
- **BioClinical ModernBERT is not a shortcut.** Biomedical literature is not
  catalog register; 53B tokens of it left the model where ModernBERT-base was
  on this site's terms.

## Files

`PLAN-vocab.md` · `vocab_audit.py` → `results/vocab_audit.json` ·
`mlm_ppl.py` → `results/mlm1_*.json` (one target per pass; `mlm_*.json` are the
earlier eight-per-pass runs, ERRORS.md #5) · `dapt.py` · `run_dapt_arms.sh`,
`run_dapt_extra.sh` → `results/dapt_arms.log` · `retrieval_eval.py` →
`results/retr_*.json` · `results/tokens_per_page.json` ·
`results/ft_dapt_*_cw6.json` · `data/full/fetch_parallel.py` (gitignored dir).
Adapted checkpoints under `models/` on this container only (gitignored).

## Not done

- No unseen-template split; the ordinary-word rise says the template is a large
  part of the held-out gain and the table cannot separate it from vocabulary.
- One learning rate, two epochs, one seed, 32M only. ettin-150m adaptation was
  gated on V2 and V5 and neither justified the five-fold cost.
- No contrastive arm. That is the follow-up the retrieval table asks for.
- Questions are template-generated; real inbound questions would replace them
  before any absolute retrieval number is quoted.
