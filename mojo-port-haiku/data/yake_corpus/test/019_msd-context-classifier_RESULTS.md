# msd-context-classifier

**Started / finished:** 2026-09-20 · **Status:** done — **a page-trained
site-section classifier does not transfer to questions. Every fine-tuned encoder
and the frozen probe score 0.91–0.96 accuracy on held-out pages and 0.21–0.35 on
question-shaped inputs, calling 65–76% of questions "support" because the FAQ
section is the only question-shaped text on the site. 140 model-written questions
train a frozen-embedding probe to 0.78 macro-F1 on questions; 617 pages train it
to 0.25, and adding the pages to the questions costs 12 points. Label the input
distribution you will classify, not the one that is free.**

**Question (Oskar).** Fine-tune a classifier for a niche bioscience context, with
the public content of mesoscale.com (Meso Scale Diagnostics: immunoassay kits,
instruments, software, services) as the example domain. Plan and seven
pre-registered predictions in [`PLAN.md`](PLAN.md), committed before any arm ran.

## Corpus

A Sonnet worker read the sitemap (14,595 URLs, one flat file; robots.txt
disallows `/LP/` and `/Bin/` and carries `crawl-delay:1000`, which I read as
milliseconds and fetched at one request per second), derived seven labels from
the URL paths, and sampled 1,198 pages stratified at 300 per label. This session
fetched them as a tracked background job (1:22–2:22 PM Eastern), extracted main
text with trafilatura, and dropped 101 non-HTML or redirected fetches and 210
pages under 60 words.

| label | source paths | sampled | kept | median words |
|---|---|---|---|---|
| products | /en/products/ — one kit, antibody or calibrator per page | 300 of 3,155 | 288 | 299 |
| references | /en/references/ — bibliography entries by year | 300 of 1,503 | 287 | 389 |
| support | /en/support/{faqs, product_information, manuals, ordering} | 217 | 103 | 79 |
| services | /en/products_and_services/services/ | 172 | 107 | 99 |
| assay_kits | /en/products_and_services/assay_kits/ — kit families | 67 | 48 | 200 |
| science_resources | /en/applications/, technical literature, webinars, posters | 48 | 34 | 139 |
| platform_products | instruments, plates, reagents, software, quote requests | 94 | 20 | 192 |

887 pages, split 617 / 132 / 138 train / dev / test by URL hash, stratified.
`platform_products` lost 74 of 94 pages to the length filter (instrument pages
are image and spec-table heavy), so its page-test cell is 3 rows; the question
set carries that label with 25 rows. Corpus and page text stay out of git
(`data/` is ignored); the URL inventory, taxonomy and sampling seed are
reproducible from `data/fetch_and_extract.py`.

**Question set.** A second Sonnet worker, given only one-line label descriptions
and never the pages, wrote 175 unambiguous questions (25 per label; support-email,
chat, search-box and forum registers, a dozen with realistic typos) plus 21
flagged ambiguous between two labels. Only the 175 count below. They are
synthetic: a sample of real inbound questions would replace them.

## Results

Six pre-registered arms plus one check arm. Pages = 138 held-out pages.
Questions = the 175 unambiguous questions, scored by the page-trained model.
ECE is top-label, 15 bins; the arrow is raw → after a temperature fitted on the
page dev split. ms is torch fp32 at batch 1 on 4 vCPU (int8 ONNX figures for
these backbones are in `encoder-platform-survey`).

| arm | params | pages acc | pages macro-F1 | pages ECE | questions acc | questions macro-F1 | questions ECE | ms | train |
|---|---|---|---|---|---|---|---|---|---|
| GLiClass zero-shot | 151M | 0.167 | 0.159 | 0.83 → 0.53 | 0.217 | 0.166 | 0.78 → 0.48 | 388 | 0 |
| gte-small probe (frozen + LR) | 33M | **0.957** | **0.936** | 0.10 → 0.07 | **0.354** | **0.250** | 0.36 → 0.41 | 254 | 4 min |
| ft ettin-32m | 32M | 0.913 | 0.744 | 0.05 → 0.05 | 0.211 | 0.122 | 0.60 → 0.60 | 38 | 3 min |
| ft ettin-68m | 68M | 0.935 | 0.751 | 0.05 → 0.04 | 0.246 | 0.169 | 0.46 → 0.36 | 104 | 8 min |
| ft ettin-150m | 150M | 0.913 | 0.747 | 0.07 → 0.05 | 0.217 | 0.145 | 0.39 → 0.29 | 193 | 17 min |
| ft ModernBERT-base | 149M | 0.935 | 0.754 | 0.03 → 0.04 | 0.240 | 0.168 | 0.24 → 0.21 | 186 | 17 min |
| ft ettin-32m, class-weighted, 6 epochs (check) | 32M | 0.942 | 0.828 | 0.04 → 0.04 | 0.223 | 0.139 | 0.65 → 0.51 | 38 | 5 min |

Chance on seven balanced question labels is 0.143 accuracy.

**On pages, the frozen probe wins by 18 macro-F1 points** and the four fine-tuned
encoders tie each other within one point. The gap is the rare classes: every
fine-tuned arm scores 0.00 on `platform_products` (3 test rows) and 0.62–0.71 on
`assay_kits` (8 rows), while the probe scores 0.80 and 0.88. Three epochs of
unweighted cross-entropy over 617 rows with 288 `products` and 20
`platform_products` never learns the small classes; the logistic regression on
frozen features does. The check arm (inverse-frequency class weights, six
epochs) recovers 8 of the 18 points on pages (0.828; `assay_kits` 0.84, `platform_products` 0.40) and leaves 11 to the probe, so the imbalance recipe is part of the gap and the frozen-features-plus-linear method is the rest at this data size. On questions the check arm is unchanged: 0.139 macro-F1, 142 of 175 questions sent to `support`.

**On questions, every page-trained arm lands within 13 points of chance**, and
the failure has one shape. The fine-tuned models predict `support` for 114–133 of
175 questions (65–76%); ettin-32m's confusion table is `assay_kits → support` 25
of 25, `products → support` 22 of 25, `platform_products → support` 21 of 25.
The `support` section is the site's FAQ, the only place in the corpus where text
is a question, so the page-trained model learned that a question is a support
page. Confidence does not flag it: ettin-32m's mean confidence on its wrong
answers is 0.79, and the temperature fitted on page dev moves question ECE from
0.60 to 0.60. The probe falls less far (0.35 accuracy) and spreads its errors,
but its 0.25 macro-F1 is still 69 points under its page score.

### Follow-up: training data for the question task

Frozen gte-small embeddings, logistic regression with balanced class weights,
5-fold stratified cross-validation over the 175 questions (`probe_queries.py`,
3:23 PM Eastern):

| training data | questions macro-F1 (mean ± sd over folds) | acc |
|---|---|---|
| none — cosine to the embedded label description | 0.51 ± 0.10 | 0.51 |
| 617 pages | 0.25 ± 0.04 | 0.35 |
| 140 questions (the other four folds) | **0.78 ± 0.08** | 0.79 |
| 617 pages + 140 questions | 0.66 ± 0.05 | 0.67 |

Per-label F1 with 140 questions: references 0.90, assay_kits 0.88, products 0.86,
science_resources 0.79, services 0.79, support 0.65, platform_products 0.63.
Zero training with a one-sentence description per label (0.51) beats 617
labelled pages (0.25) and beats GLiClass zero-shot (0.17) with the same
descriptions. Adding the pages to the questions costs 12 points: the page
distribution pulls the decision boundary the wrong way for questions.

## Predictions scored

| # | prediction | outcome |
|---|---|---|
| P1 (60%) | probe within 5 macro-F1 of the best fine-tune on pages | **right, and understated** — the probe is 18 points above |
| P2 (85%) | every fine-tune beats GLiClass by >15 on pages | right — 0.74–0.75 vs 0.16 |
| P3 (65%) | every arm drops ≥15 on questions; fine-tunes drop more than the probe | first half right — drops of 58–69 points; second half **wrong** — the probe dropped furthest in points (0.94 → 0.25) because it had furthest to fall; all arms land within 13 points of chance |
| P4 (55%) | ettin-32m within 3 of ettin-150m on pages; gap ≥5 on questions | first half right (0.744 vs 0.747); second half **wrong** — 0.122 vs 0.145, both at chance |
| P5 (70%) | fine-tune raw ECE >0.08 on questions; temperature halves it | first half right (0.24–0.60); second half **wrong** — a temperature fitted on page dev moved question ECE by 0.00–0.11, never half. Calibration fitted on one distribution does not transfer to another |
| P6 (60%) | coverage at a 5% error budget on questions <50% for every arm | **wrong as stated, and the metric was the error**: the dev-chosen threshold covers 21–90% of questions with a realized error of 0.59–0.77. A threshold chosen on the training distribution is void on another; see ERRORS.md |
| P7 (75%) | confused pairs are products/assay_kits and platform/support | **wrong** — the failure is one sink (`support`), not pairwise |

Four of seven fully or half wrong, all in the same direction: I predicted graded
degradation on questions and got a drop to chance with a single mechanism. The
decision rule in PLAN.md fires on P1 and P3 both holding: **label questions, not
pages.**

## Recommendations

- **Training data is questions.** Two hundred model-written questions were
  enough for 0.78 macro-F1 with a frozen 33 MB encoder and no fine-tuning. Real
  inbound questions (support inbox, site search logs) are the next corpus, and
  the cheapest labeller is the label-description cosine at 0.51 followed by a
  human pass on the disagreements.
- **The probe is the deployable artifact at this data size.** gte-small int8 is
  33 MB and 0.455 acc@1 on WANDS in `hypothetical-classification`; here its
  frozen features beat four fine-tuned encoders on pages and on questions.
  Re-run `train.py --class-weight` once the question corpus reaches the low
  thousands; that is where fine-tuning has room to beat frozen features.
- **The page corpus is the retrieval corpus.** A routed question lands in it;
  it is the wrong training set for the router.
- **Do not fit calibration or an abstention threshold on pages** for a question
  classifier. Every calibration number here that was fitted on page dev was
  wrong on questions by an order of magnitude.

## Files

`PLAN.md` predictions · `train.py` (ft and probe arms, metrics) ·
`zeroshot_gliclass.py` · `probe_queries.py` (follow-up) · `run_arms.sh` driver ·
`results/*.json` per arm, `results/*_query_preds.jsonl` per-question predictions,
`results/run_arms.log` · `data/` (gitignored): `fetch_and_extract.py`,
`extract_corpus.py`, `taxonomy.json`, `sampled_urls.json`, `queries.jsonl`,
`label_descriptions.json`. Models: `models/final_ft_ettin_{32m,68m,150m}.pt`
(gitignored, on this container only).

## Not done

- No real questions; the 175 are Sonnet-written from label descriptions and the
  0.78 is a ceiling on how separable those descriptions are, not a measurement
  of inbound traffic.
- No out-of-domain rejection arm.
- No ONNX export of any arm; the survey's ladder covers the backbones.
- The 21 ambiguous questions were written and not scored.
- Fine-tune recipe was run once per backbone at one learning rate; the check arm
  varies class weighting and epochs only.
