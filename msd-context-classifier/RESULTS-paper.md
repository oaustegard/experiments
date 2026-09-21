# msd-context-classifier — paper filter

**Started / finished:** 2026-09-20 10:35 PM – 2026-09-21 1:55 AM Eastern ·
**Status:** done — **the filter works from abstracts and the vocabulary
adaptation does not change it. Every fine-tuned encoder scores AUC 0.98 and
catches 91–95% of the bibliography papers whose abstracts never name the
platform, at 8–14% false positives on topic-matched papers; at 95% precision
that recall is 70–80%. Stock ettin-32m and the two adapted ettin-32m checkpoints
are within noise of each other on every number, and ettin-150m is no better.
Only 11% of MSD's own bibliography names the platform in title or abstract, so
the abstract-level signal is what the study measured and how, which a stock
encoder already reads.**

**Question (Oskar).** "One of our challenges is: does this paper relate to MSD?
That's one I was hoping would be aided by the vocabulary expansion." Plan and
six predictions in [`PLAN-paper.md`](PLAN-paper.md), written before the corpus
existed. Third round on this corpus after [`RESULTS.md`](RESULTS.md) and
[`RESULTS-vocab.md`](RESULTS-vocab.md).

## Corpus

| set | rows | source | with a platform cue |
|---|---|---|---|
| positives (`msd`) | 1,004 | MSD's bibliography pages (1,420 titles) resolved by title to PubMed, 885 resolved to one PMID, 1,006 with an abstract of ≥ 50 words | **113 (11%)** |
| hard negatives | 1,489 | 20 PubMed queries on the positives' subjects (serum cytokines, neuroinflammation, multiplex immunoassay, oncology biomarkers, immunogenicity, metabolic-syndrome biomarkers, …), same years, NOT any platform term | 0 by construction |
| easy negatives | 977 | 10 queries in unrelated fields, same NOT clause | 0 |

A Sonnet worker built the title list and the scripts; this session ran the
NCBI chain as a tracked job (3 requests/s, 11:34 PM – 12:08 AM). Cue = any of
meso scale, MSD (whole word), V/U/S/R-PLEX, QuickPlex, SECTOR imager,
SULFO-TAG, MULTI-ARRAY, electrochemiluminescen. Split 70/15/15 by PMID hash,
stratified: test = 523 rows, 152 positives, **134 of them cue-free**.

**The 11% is the finding that reframes the task.** Nine of ten papers that used
the platform say so only in their methods section. An abstract-level filter
cannot be a vocabulary matcher for this domain; it has to recognise the kind of
study MSD's customers publish.

## Arms

Same recipe as the earlier rounds (`train.py`: mean-pool + linear head,
class-weighted cross-entropy, lr 5e-5, batch 16, 256 tokens of title +
abstract, best epoch by dev macro-F1; six epochs for the 32M arms, four for
150M). Thresholds: 0.5, and the highest threshold giving 95% precision **on the
test set itself** (dev probabilities are not dumped; this makes the P95 column
optimistic by a few points for every arm equally and noisy at 152 positives).

## Results

| arm | AUC | macro-F1 @0.5 | recall @0.5 | cue-free recall @0.5 | hard-neg FPR @0.5 | recall @P95 | cue-free recall @P95 | train |
|---|---|---|---|---|---|---|---|---|
| cue regex | — | — | 0.118 | 0.000 | 0.000 | 0.118 (P = 1.00) | 0.000 | 0 |
| gte-small probe (frozen + LR) | 0.952 | 0.870 | 0.809 | 0.791 | 0.126 | 0.441 | 0.425 | 18 min |
| ft ettin-32m stock | 0.984 | **0.929** | 0.914 | 0.910 | **0.079** | 0.750 | 0.761 | 19 min |
| ft ettin-32m dapt+term | 0.983 | 0.915 | 0.934 | 0.933 | 0.126 | **0.796** | **0.799** | 19 min |
| ft ettin-32m dapt+vocab+term | **0.984** | **0.931** | 0.928 | 0.925 | 0.084 | 0.724 | 0.746 | 19 min |
| ft ettin-150m stock | 0.983 | 0.914 | **0.954** | **0.955** | 0.140 | 0.697 | 0.694 | 83 min |

Easy-negative false-positive rate is 0.6–1.3% for every fine-tuned arm and 0
for the probe; the hard negatives are where the errors live.

**Reading the table.** The four fine-tuned arms share one AUC to the third
decimal; what differs between them is where the 0.5 threshold happens to sit
on the same curve (the 150M arm trades 6 points of hard-negative precision for
4 of recall). Cue-free recall at 0.5 spans 0.91–0.96 across the four; at the
95%-precision point it spans 0.69–0.80 with the term-masked adapted arm on top
by 4 points over stock, inside the 5-point tie band the plan set for 134 rows.
The frozen probe is the only arm clearly behind, and the regex is not a
competitor: it finds the 11%.

**Why the adaptation cannot show here.** The adapted encoders learned the site's
terms (V-PLEX, SULFO-TAG, the analyte panel names) to 0.83 whole-term
recovery. The abstracts contain those terms in 11% of positives, and those
11% the regex already catches. The remaining 89% are separated from the hard
negatives by study design and phrasing ("plasma levels of IL-6, IL-8 and TNF-α
were quantified by multiplex immunoassay in n = …"), which the stock encoder
reads as well as the adapted one. The vocabulary lives in the methods
sections that PubMed does not serve.

**Label noise in the hard negatives, in the direction that flatters no arm.**
MSD's bibliography is curated and incomplete, and the NOT clause only removes
papers whose abstract names the platform, so some "hard negatives" used MSD
without saying so. Part of the 8–14% hard-negative false-positive rate may be
correct answers. Full text would settle it; abstracts cannot.

## Predictions scored

| # | prediction | outcome |
|---|---|---|
| W1 (80%) | 30–60% of positives carry a cue; regex precision ≥ 0.97 at that recall | **wrong** on the rate (11%); regex precision 1.00 at recall 0.12 |
| W2 (65%) | dapt+term beats stock on cue-free recall at P95 by ≥ 10 | **wrong** — +4, a tie at n = 134 |
| W3 (60%) | dapt+vocab+term within 3 of dapt+term | **wrong** at 3 (−5), a tie at the plan's 5 |
| W4 (55%) | gte-small probe beats stock fine-tune on macro-F1, loses on cue-free recall | first half **wrong** (0.870 vs 0.929); second half right (0.43 vs 0.80) |
| W5 (50%) | ettin-150m matches or beats adapted 32m on cue-free recall | **wrong** at P95 (0.69 vs 0.80); at 0.5 it is 2 points ahead at 6 points worse precision — same curve |
| W6 (70%) | every encoder's hard-negative precision < 0.95 at 0.5 | **right** — 0.82–0.89 |

One right of six. The misses are one miss: I modelled the task as
string-detection with a long tail, and it is topic recognition with a short
head. Fourteen predictions over three rounds on this corpus are now wrong or
half-wrong in that same direction.

## Recommendations

- Run the regex on every abstract, then the model on the rest. The regex is
  exact on the 11% and free. The fine-tuned ettin-32m (stock; the adaptation earns nothing here) runs on the
  rest at 0.5 for a 91% catch rate with 8% false positives on look-alike papers,
  or at a stricter threshold for 95% precision and 75% recall. 38 ms per
  abstract on 4 vCPU (survey ladder: 12 ms int8).
- **Real triage data next.** The 1,004 positives are MSD's curated bibliography
  and the negatives are query-matched, so the numbers are a ceiling on how
  separable the two lists are, not a measurement of the inbound stream. Two
  hundred actually triaged papers with the human's decision would recalibrate
  every threshold above.
- **Full text if recall on cue-free papers must go above 0.9.** Methods sections
  carry the platform names; PubMed Central's open-access subset would give the
  regex the 89% and the vocabulary adaptation something to read.

## Later checks (2026-09-21)

Predictions X1–X4 in the PLAN-paper.md addendum, plus two checks Oskar and a
teammate asked for. Every number below is SPECTER2 vectors from Semantic
Scholar's API (768-d, title + abstract, 3,443 of 3,470 PMIDs returned) unless
it says ettin.

**1. The AUC depends on how the negatives were drawn.** Same 1,004 bibliography positives,
three negative sets:

| negatives | fine-tuned ettin-32m AUC | SPECTER2 + LR AUC |
|---|---|---|
| my 20 topic queries (the first result above) | 0.984 | 0.966 |
| PubMed's own "similar articles" of each positive, top 5, 1,500 sampled | 0.815 | 0.794 |

Against the papers PubMed itself rates closest to MSD's, recall at 0.5 is 0.61
at 36% false positives; at 95% precision recall is 0.07. X1 (AUC ≥ 0.95)
**wrong**. Most of the first result was the query set. Label noise in the
neighbour set, measured against PMC full text (below): 0.5% are MSD papers
(1.7% of the query negatives), which moves the AUC by well under a point.

**2. Semantic Scholar's SPECTER2 as the detector.** Logistic regression on the
hosted vectors, original split: AUC 0.966, cue-free recall 0.87, two points
under the fine-tune with no encoder of our own. X2 **right**. Cosine to the
centroid of the positives: AUC 0.56; X3 **wrong**. MSD's bibliography is not
one cluster in that space. 10-NN vote against the labelled set: 0.90.
Semantic Scholar's recommendation endpoint, seeded with 100 positives and 50
negatives, returned 499 papers all from 2026 and none of the 150 held-out
positives: it searches recent papers only and cannot be the sweep.

**3. Centring and 1-bit codes (Oskar's remex question).** Subtracting the
training mean before cosine, the "one bit from the centre" step, takes the
centroid detector from 0.56 to 0.91 and the k-NN vote from 0.90 to 0.93;
SPECTER2 is anisotropic and the common direction swamps raw cosine. Logistic
regression on 1-bit sign codes: centred 0.924, centred + random rotation
0.942, raw 0.910, against 0.966 in float. Hard-negative FPR at 98% recall on
the original split: ettin fine-tune 0.18, LR float 0.34, LR 1-bit rotated
0.48, centred centroid 0.59 (`paper_s2_greedy.py`).

**4. PMC full text as the labelled set.** `esearch db=pmc` for "meso scale
discovery" OR "meso scale diagnostics": 9,999 hits returned (the endpoint's
cap; the true count is ~14,000 against 531 PubMed abstracts), 9,927 PMIDs,
9,818 with an abstract; 5% of those abstracts name the platform. Only 25 of
the 1,004 bibliography papers are in it, and 99% of it is 2019 or later: the
curated bibliography and the uncurated user population are two different
populations. The bibliography-trained SPECTER2 detector recalls **39%** of the
9,729 PMC papers outside the bibliography at its 0.5 point, 58% at a threshold
passing 22% of look-alikes.

**5. The modern population, properly sampled.** 9,754 PMC positives against
9,916 of their own PubMed similar-articles (top 2 each, 116 cue-carrying
neighbours set aside), split 70/15/15 by PMID hash, SPECTER2 vectors:

| classifier | test AUC | recall at 5% neighbour-FPR | neighbour-FPR at 95% recall |
|---|---|---|---|
| logistic regression | **0.745** | 0.20 | 0.73 |
| RBF-kernel SVM | 0.714 | 0.16 | 0.78 |
| MLP, 256 hidden | 0.712 | 0.17 | 0.77 |
| 10-NN vote, centred cosine | 0.578 | 0.09 | 0.93 |
| logistic regression, train ≤ 2023 → test 2024+ | 0.687 | — | — |

The teammate's objection that a single hyperplane cannot cover a multi-cluster
positive class is right in principle and does not bind here: every nonlinear
classifier on the same vectors is worse, so the limit is the information in
the vectors, not the shape of the boundary. The PMC-trained detector recalls
37% of the old bibliography at its 0.5 point, the mirror of check 4.

**6. Citation-graph features (Oskar: "try the citation-graph features
from Semantic Scholar next").** Same population and split as check 5.
Semantic Scholar's batch endpoint returned graph fields for all 19,670 papers;
its reference lists are non-empty for 78% (publisher elision, C4 said ≥ 40%
missing), PubMed elink's for 81%. Incoming citations are present for 92% (Semantic
Scholar) and 87% (PubMed). Every
"known MSD" set (papers, authors, the 100-paper canon most cited by
positives) is built from training positives only. Logistic regression on
test, AUC and recall at 5% neighbour-FPR (`paper_graph_probe.py`):

| features | test AUC | recall at 5% FPR |
|---|---|---|
| SPECTER2 alone (check 5) | 0.745 | 0.20 |
| all 18 graph scalars | 0.727 | 0.17 |
| bag of PubMed references (82k columns) | 0.739 | 0.22 |
| bag of Semantic Scholar references | 0.706 | 0.19 |
| bag of citers (PubMed / Semantic Scholar) | 0.539 / 0.586 | 0.14 / 0.15 |
| bag of authors | 0.575 | 0.16 |
| venue | 0.709 | 0.22 |
| all bags, no embedding | **0.774** | **0.26** |
| SPECTER2 + scalars | 0.749 | 0.20 |
| SPECTER2 + scalars + all bags | 0.757 | 0.21 |
| SPECTER2 + scalars, train ≤ 2023 → test 2024+ | 0.672 | 0.11 |

The single scalars are the surprise. Every "cites known MSD papers" feature
scores *below* 0.5: references to training positives 0.35, citers that are
training positives 0.23, references to the canon 0.37. The neighbours cite
known MSD papers more than the positives do. This is the sampling: a
neighbour is PubMed's similar-article of a positive, so it sits next to the
positives in time and topic and often cites them, while the positives' own
reference lists point at older work that the 99%-post-2019 positive set
does not contain (ERRORS.md #16). Citation counts run the same way
(0.31: positives are newer and less cited). The one feature that behaves
as predicted, author overlap, is weak: 0.61 alone, 0.58 as bag. Labs that
published on MSD do publish again, but their neighbours share authors
nearly as often. Reference bags beat the embedding by three points and
every combination with the embedding lands between the two, a
regularisation mismatch (one C over a standardised dense block and a binary
sparse block) I did not tune away. The temporal holdout drops the combined
model 7.7 points, more than SPECTER2 alone dropped in check 5 (5.8): graph
features age worse here than topic features.

C1 (a bag alone ≥ 0.80) **wrong**, best 0.739. C2 (author overlap ≥ 0.75
and strongest) **wrong**, 0.615 and fourth. C3 (combined ≥ 0.85) **wrong**,
0.757. C4 (≥ 40% elided; PubMed covers more) **wrong** on the number, 22%,
right on the direction by two points. C5 (temporal within 5 points) **wrong**,
7.7. Five for five in the same direction as the previous nineteen: I keep
expecting a new signal to be worth more than it is against negatives drawn
from the positives' own neighbourhood. Graph features are worth about three
AUC points over SPECTER2 on this task and cost two API pulls per paper.

**What the sequence says.** 0.98 was the bibliography against my queries; 0.82
the bibliography against its neighbours; 0.75 the uncurated population against
its neighbours, 0.77 with its citation graph; 0.69 across a year boundary. A title-and-abstract embedding
trained on citation structure carries some of the "used MSD" signal, and at
the base rate of a literature sweep, one in a few hundred at best, an AUC of
0.75 gives a candidate list that is nearly all false positives at any usable
recall. The abstract-level rung can rank a topic-narrowed pool; it cannot be
the sweep.

**What can.** PMC's full-text index is reachable at no cost and already returns
~14,000 MSD papers with precision the abstract rung cannot approach; Europe
PMC's full-text search (not reachable from this container) covers a broader
open-access set. The pipeline that follows from this round: full-text search
first, wherever full text exists; the SPECTER2 rung only over the non-open
remainder, with a threshold set on the PMC-labelled positives and their
neighbours (`paper_pmc_probe.py`), and a full-text check on whatever it
passes. Semantic Scholar's SPECTER2 is the cheapest usable feature for that
remainder, and nothing in this round found a better one.

## Files

`PLAN-paper.md` · `paper_corpus.py` (assembly + cue regex + regex baseline
→ `results/cue_regex.json`) · `run_paper_arms.sh` → `results/paper_arms.log`
· `results/paper_*.json`, `results/paper_*_test_preds.jsonl` ·
`paper_score.py` → `results/paper_summary.json` · `paper_s2_probe.py`,
`paper_s2_greedy.py`, `paper_pmc_probe.py`, `paper_graph_probe.py` →
`results/paper_{s2,pmc,graph}_probe.json` · `data/papers/` (gitignored):
`titles.jsonl`, `resolved.jsonl`, `positives.jsonl`, `hard.jsonl`,
`easy.jsonl`, `neighbors*.jsonl`, `pmc_positives.jsonl`, `s2_specter2.jsonl`,
`s2_graph.jsonl`, `pubmed_links.jsonl`, the fetch scripts and `neg_queries.py`.

## Not done

- P95 thresholds chosen on test, not dev (`train.py` dumps test predictions
  only); stated above.
- Label noise measured against PMC full text after the first result: 1.7% of
  the query negatives, 0.5% of the neighbour negatives.
- One seed, one learning rate; ties are ties.
- No full-text classifier arm; full-text *search* (PMC) was measured as the labelled-set source and is the recommended first rung.
