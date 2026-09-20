# msd-context-classifier — plan and pre-registered predictions

Written 2026-09-20 before any arm ran on real data. Follows
`encoder-platform-survey/RESULTS.md` (same day), which recommended fixed-schema
classifiers fine-tuned on labelled data, on Ettin encoders, int8 ONNX on CPU.

**Question (Oskar).** Fine-tune a classifier for a niche bioscience context,
using the public content of mesoscale.com (Meso Scale Diagnostics: immunoassay
kits, instruments, software, services) as the example domain.

## Task

Seven-way classification into the site's own information architecture, with the
label taken from the URL path (`data/taxonomy.json`, derived by a worker from the
sitemap; no model labelled anything):

| label | source paths | sampled pages |
|---|---|---|
| products | /en/products/ (individual kit and antibody pages) | 300 of 3,155 |
| references | /en/references/ (publication bibliography, 2000–2026) | 300 of 1,503 |
| support | /en/support/{faqs, product_information, manuals, ordering} | 217 |
| services | /en/products_and_services/services/ | 172 |
| platform_products | software, instrumentation, plates, reagents, assay-development tools | 94 |
| assay_kits | /en/products_and_services/assay_kits/ (kit families) | 67 |
| science_resources | /en/applications/, technical literature, webinars, posters | 48 |

Two test surfaces:

1. **Pages** — held-out pages, stratified 70/15/15 by URL hash. This is the easy
   surface: pages in one section share templates and boilerplate.
2. **Queries** — ~175 model-written user questions (support inbox / chat / search
   box register), 25 per label, written by a worker from the label descriptions
   alone, never from page text. The deployed use is classifying questions, not
   pages, so this is the number that matters. It is synthetic; a real inbound
   sample would replace it.

## Arms

| arm | model | params | what it tests |
|---|---|---|---|
| gliclass | knowledgator/gliclass-modern-base-v2.0, zero-shot | 151M | the no-training floor |
| probe | thenlper/gte-small frozen mean-pool + logistic regression | 33M | the cheap floor (`nl2sh-dense` rule: linear probe before pricing a fine-tune) |
| ft-32m | jhu-clsp/ettin-encoder-32m fine-tuned | 32M | the router-size candidate |
| ft-68m | jhu-clsp/ettin-encoder-68m | 68M | the likely knee |
| ft-150m | jhu-clsp/ettin-encoder-150m | 150M | ModernBERT-base class |
| ft-mb | answerdotai/ModernBERT-base | 149M | the reference every clone uses |

Fine-tune recipe, fixed before running: mean-pool + linear head, AdamW lr 5e-5,
wd 0.01, 6% warmup then linear decay, batch 16, max 256 tokens of
`title + text`, 3 epochs, best epoch by dev macro-F1, seed 20260920, CPU.
Temperature fitted on dev; coverage threshold chosen on dev at a 5% error budget
and applied to test. Metrics: accuracy, macro-F1, ECE raw and after temperature,
coverage at 5% error, per-label F1, ms/example at batch 1 (torch fp32; the ONNX
int8 numbers are in the survey).

## Predictions (confidence)

- **P1 (60%)** On pages, the frozen gte-small probe lands within 5 macro-F1
  points of the best fine-tuned arm. Section templates make page classification
  near-trivial for any encoder.
- **P2 (85%)** Every fine-tuned arm beats GLiClass zero-shot by more than 15
  macro-F1 points on pages. Site-IA labels are not natural-language categories.
- **P3 (65%)** On queries every arm drops at least 15 macro-F1 points from its
  page score, and the fine-tuned arms drop more than the probe: fine-tuning on
  pages learns template and boilerplate that questions do not carry.
- **P4 (55%)** ettin-32m is within 3 macro-F1 of ettin-150m on pages, and the
  gap is at least 5 points on queries. The small model has the templates; the
  larger one has more of the vocabulary.
- **P5 (70%)** Raw ECE of every fine-tuned arm is above 0.08 on queries;
  temperature scaling at least halves it.
- **P6 (60%)** Coverage at a 5% error budget on queries is below 50% for every
  arm, i.e. the page-trained classifier cannot run unsupervised on questions.
- **P7 (75%)** `products` vs `assay_kits` and `platform_products` vs `support`
  are the confused pairs on queries: the site's IA splits by page type, users
  ask by topic.

## Kill / decision rules

- If P1 holds and P3 holds, the recommendation is to label questions, not pages:
  the page corpus is the wrong training distribution and the cheap probe is
  enough for what pages can teach.
- If the fine-tuned arms hold their page score on queries (P3 fails), the
  page corpus is a usable free training set and the next step is the ONNX int8
  export of the smallest arm within 2 points of the best.
- Differences under 3 macro-F1 points on a 180-row test set are not
  meaningful and are reported as ties.

## Not in scope this round

- Out-of-domain rejection (is this text about MSD at all). Needs a negative
  corpus; follow-up.
- Real inbound questions. Synthetic queries only; stated wherever a query
  number appears.
- ONNX export and int8 latency of the winner. The survey's ladder covers the
  backbone latency; the head adds nothing measurable.
