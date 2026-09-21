# "Does this paper relate to MSD?" — plan and pre-registered predictions

Written 2026-09-20, 10:35 PM Eastern, before the corpus existed. Oskar: "One of
our challenges is: does this paper relate to MSD? That's one I was hoping would
be aided by the vocabulary expansion." This is the vocabulary-bound task the
previous round lacked (RESULTS-vocab.md, memory 4216b6ad): the label turns on
whether a paper used the company's platform, and the platform is named by
exactly the terms the adaptation learned (V-PLEX, U-PLEX, SULFO-TAG, MSD,
electrochemiluminescence).

## Data

- **Positives**: MSD's own bibliography, the 1,420 `references` pages of the
  crawl, resolved by title to PubMed abstracts (a worker; expect 60–80% to
  resolve with an abstract).
- **Hard negatives** (~1,500): PubMed papers on the same subjects (cytokine
  panels, plasma biomarkers, serology, bioanalysis) whose title and abstract
  do not mention the platform. Some will have used MSD in their methods without
  saying so in the abstract; that is label noise in the direction that lowers
  every arm's measured precision equally.
- **Easy negatives** (~600): unrelated biomedical fields.
- **Cue flag**: whether title+abstract contains a literal platform string. The
  cue-free positives are the subset where a classifier must know the domain
  rather than match a string; that subset is the experiment.

Split 70/15/15 by PMID hash, stratified by label. Input = title + abstract,
256 tokens.

## Arms

| arm | what it tests |
|---|---|
| cue regex | the string-match baseline every classifier must beat on cue-free positives, and the precision ceiling on cue positives |
| gte-small probe | frozen embedding-trained features + LR |
| ft ettin-32m stock | fine-tuned, no domain knowledge |
| ft ettin-32m dapt+term | fine-tuned from the term-masked adapted checkpoint |
| ft ettin-32m dapt+vocab+term | fine-tuned from the expanded-vocabulary checkpoint |
| ft ettin-150m stock | the size control: is 150M of general pretraining worth more than 32M + domain adaptation |

Recipe as before: mean-pool + linear head, class-weighted cross-entropy, lr
5e-5, batch 16, 6 epochs, best epoch by dev macro-F1, threshold 0.5 and a
dev-chosen threshold at 95% precision. Metrics on test: AUC, precision, recall,
macro-F1; **recall on cue-free positives at the 95%-precision threshold**;
false-positive rate on hard negatives.

## Predictions (confidence)

- **W1 (80%)** Between 30% and 60% of resolved positives carry a literal cue in
  title+abstract. The regex scores ≥0.97 precision and recall equal to that
  cue rate; recall 0 on cue-free positives by construction.
- **W2 (65%)** The adapted ettin-32m (dapt+term) beats stock ettin-32m on
  cue-free-positive recall at 95% precision by ≥10 points. This is the
  prediction the whole round exists to test.
- **W3 (60%)** dapt+vocab+term is within 3 points of dapt+term (expansion adds
  nothing over term masking, as in the previous round).
- **W4 (55%)** gte-small probe beats stock ettin-32m fine-tuned on overall
  macro-F1, and loses to the adapted fine-tune on cue-free recall.
- **W5 (50%)** ettin-150m stock matches or beats adapted ettin-32m on cue-free
  recall: general pretraining at 5x the size carries as much of the
  immunoassay literature as two epochs of the site.
- **W6 (70%)** Every encoder arm's precision on hard negatives is under 0.95 at
  the 0.5 threshold; the topic overlap is the hard part, and cue-free
  positives are indistinguishable from hard negatives on a first read.

## Decision rule

- W2 holds → the vocabulary adaptation paid for this task; ship the adapted
  encoder behind the regex (regex first, model on the remainder), and the next
  step is real MSD triage data.
- W2 fails and W5 holds → size beats adaptation; use ettin-150m or
  ModernBERT-base fine-tuned, forget the adaptation.
- W2 fails and W5 fails → nothing beyond the regex works on cue-free
  abstracts at this data size; the filter is regex + full-text methods
  section, not an encoder over abstracts.
- Differences under 5 points on the cue-free subset (likely 200–400 rows) are
  ties.

## Addendum, 2026-09-21 2:15 AM Eastern — two checks, predicted before running

Oskar: "Yes run it but also: Semantic Scholar already publishes specter2
encodings of titles and abstracts. Does this suggest we may be able to use
those to detect other MSD papers?"

**Check 1 — neighbour negatives.** For every positive, PubMed's own "similar
articles" (elink `pubmed_pubmed`, top 5, positives excluded, any neighbour
carrying a platform cue set aside as a probable unlisted MSD paper), sampled to
1,500, replacing the query-built hard negatives. Same positives, same easy
negatives, same recipe, stock ettin-32m only (the arms tied).

**Check 2 — SPECTER2 probe.** Semantic Scholar's precomputed `specter_v2`
vectors for every PMID in the corpus, logistic regression on the frozen 768-d
vectors, same split; plus cosine to the centroid of the training positives as
the zero-parameter "more like these" baseline. If this works, the detector runs
over embeddings Semantic Scholar already hosts for the whole literature, with
no encoder on our side.

- **X1 (60%)** Stock ettin-32m AUC on positives vs neighbour negatives stays
  ≥ 0.95. The topic signal is mostly real; the query set contributed under 3
  AUC points.
- **X2 (55%)** SPECTER2 + logistic regression reaches AUC within 0.02 of the
  fine-tuned encoder on the original split (≥ 0.96). A citation-trained
  paper embedding already places MSD-customer studies in one region.
- **X3 (65%)** Centroid cosine alone reaches AUC ≥ 0.90 on the original
  split: the positives are one cluster, and nearest-neighbour search over
  Semantic Scholar's embeddings is a usable "find other MSD papers" tool
  before any training.
- **X4 (70%)** The neighbour set contains ≥ 3% papers with a platform cue,
  i.e. MSD papers absent from the curated bibliography, which bounds the
  label noise in every hard-negative number above from below.
