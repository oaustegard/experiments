# Binary tag vector over MSD catalog terms: results

Verified: 2026-11-25, test split run once after the dev freeze (hashes in `results/frozen_sha256.txt`).
Gate: `python3 evaluate.py test` from this directory with `data/vocab.json` and `data/fixture.json` rebuilt as below.

## Answer

A binary vector with one dimension per MSD catalog analyte beats BM25 and word+char TF-IDF at
finding PubMed abstracts about that analyte when the query is the catalog name or the gene
symbol. Test R@10 for catalog-name queries is 0.81 against 0.72 for TF-IDF and 0.68 for BM25.
For gene-symbol queries it is 0.85 against 0.59 and 0.51. For MeSH-heading queries it is no
better than the lexical arms (0.59 against 0.63 and 0.62), because the tagger recognizes only
71% of those headings. Its MRR is lower there too.
RRF fusion of tag and TF-IDF did not beat tag alone on R@10.

The tag vector's advantage is that the analyte's surface forms (`IL-1β`, `IL1B`, `interleukin-1 beta`)
collapse to one dimension. Each of its three query forms lands within 0.05 of the others where the
tagger covers all three; BM25's forms spread by 0.22.

sparseup v2 was not run. Its weights sit on hf.co and the int8 pack on an R2 host, and both are refused
at this container's egress. The Hugging Face connector reads text files only, and there is no torch here.

## Test split (91 concepts, 260 queries, 1,144 abstracts)

R@10 = hits in top 10 / min(relevant, 10). Mean with 95% bootstrap CI (2,000 resamples over queries).

| query form | n | tag coverage | tag_jaccard | bm25 | tfidf_wc | rrf(tag, tfidf) |
|---|---|---|---|---|---|---|
| catalog name | 91 | 1.00 | **0.807** [0.746, 0.863] | 0.684 [0.617, 0.747] | 0.723 [0.664, 0.786] | 0.805 [0.750, 0.858] |
| HGNC symbol | 78 | 0.96 | **0.851** [0.795, 0.902] | 0.505 [0.427, 0.582] | 0.586 [0.512, 0.655] | 0.731 [0.668, 0.792] |
| MeSH heading | 91 | 0.71 | 0.586 [0.495, 0.675] | 0.633 [0.564, 0.698] | 0.621 [0.557, 0.683] | 0.630 [0.560, 0.702] |
| all | 260 | 0.89 | **0.743** [0.698, 0.784] | 0.612 [0.573, 0.653] | 0.647 [0.608, 0.683] | 0.721 [0.684, 0.759] |

MRR, same layout:

| query form | tag_jaccard | bm25 | tfidf_wc | rrf(tag, tfidf) |
|---|---|---|---|---|
| catalog name | 0.933 | 0.939 | 0.948 | 0.954 |
| HGNC symbol | 0.955 | 0.875 | 0.882 | 0.937 |
| MeSH heading | 0.688 | 0.886 | 0.857 | 0.782 |
| all | 0.854 | 0.901 | 0.896 | 0.889 |

Paired differences (R@10, 95% CI):

- tag − bm25: catalog +0.123 [0.055, 0.190]; symbol +0.346 [0.265, 0.427]; MeSH −0.046 [−0.132, 0.039]
- tag − tfidf: catalog +0.084 [0.016, 0.149]; symbol +0.264 [0.193, 0.341]; MeSH −0.035 [−0.121, 0.056]
- rrf(tag, tfidf) − tfidf: all +0.075 [0.043, 0.106]
- MRR, tag − bm25 on MeSH: −0.197 [−0.303, −0.096]

The MRR gap comes from ties. A binary vector gives every abstract tagged only with the query's
analyte the same score, so the first relevant hit is often not at rank 1 even when most of the top 10
are relevant. The lexical arms order by term frequency and rarely tie.

Other arms (in `results/summary_test.json`): tag_dot trails tag_jaccard everywhere (all-forms R@10 0.721);
rrf(tag, bm25) 0.607; rrf(bm25, tfidf) 0.626 R@10 and the best MRR at 0.916.

Dev split (39 concepts, 115 queries), after the one dev-side tagger change: all-forms R@10 tag 0.806,
bm25 0.557, tfidf 0.655, rrf(tag, tfidf) 0.764. TF-IDF beat BM25 on dev, so it was the lexical arm
fixed for fusion.

## Predictions, scored

Written before any run (`PREDICTIONS.md`, commit 11794a7).

| | predicted | test |
|---|---|---|
| catalog: tag / bm25 / tfidf / rrf | 0.70 / 0.60 / 0.65 / 0.72 | 0.81 / 0.68 / 0.72 / 0.81 |
| symbol | 0.70 / 0.45 / 0.50 / 0.72 | 0.85 / 0.51 / 0.59 / 0.73 |
| MeSH | 0.55 / 0.60 / 0.62 / 0.68 | 0.59 / 0.63 / 0.62 / 0.63 |
| coverage catalog / symbol / MeSH | 0.97 / 1.00 / 0.70 | 1.00 / 0.96 / 0.71 |

1. Tag beats BM25 on catalog-name and symbol queries by more than the CI: **held** (both CIs exclude 0).
2. Tag loses to the best lexical arm on MeSH queries: **held on MRR only**. On R@10 the difference is
   inside the CI.
3. RRF(tag, best lexical) is the best arm pooled: **wrong**. Tag alone has the higher R@10, and rrf(bm25, tfidf) the higher MRR.
4. Tag's forms within 0.05 where fully covered, BM25's spread over 0.10: **held**. Tag catalog/symbol/MeSH 0.891/0.891/0.849 over the
   57 test concepts with all three forms tagged; BM25 0.698/0.481/0.674 on the same queries.

Same bias as 37dc9ebe: the absolute numbers came in higher than predicted, by 0.05 to 0.15 for the tag arm.

## Method

**Vocabulary.** One dimension per catalog concept. The 773 catalog names in `../data/msd_analytes.txt` collapse to 696
concepts (517 mapped to an HGNC gene). Surface forms come from two sources only: the catalog name itself
(whole, without its parenthetical, and each `/` part) and HGNC (approved symbol and name, alias and
previous symbols and names). No surface form is written by hand. Keys are normalized for case and punctuation, and Greek letters
are written both spelled out and as their Latin initial. A key that names two concepts is dropped (25),
as is an HGNC-sourced key that HGNC also gives to another gene (165), unless the key is the
catalog's own name for exactly one concept. Keys of the other ~190k HGNC genes block a span without
adding a dimension, so `IL-1 receptor antagonist` is not read as IL-1. HGNC file sha256
`46b83736ec351f15e22331575dce2f08416c5233765b26a08c65e6b42e56b651`.

**Tagger.** Greedy longest match over 1–8 word n-grams (split on whitespace, punctuation, hyphen, slash).
Short purely alphabetic keys (≤5 letters) need at least two capitals in the text, so `ACE` the gene is
not `ace`. **Dev change:** that rule now follows the catalog's own spelling. A short key the catalog
writes as a plain word (`Furin`, `Tau`, `Renin`, also `Akt`, `Fas`, `Met`, `Kit`) matches in any case.
This fixed the dev misses on furin and tau, where abstracts write the name in lower case. It was the only change made after looking at dev. Code and data were frozen before the test run.

**Retrieval set.** For each of 220 concepts sampled from the 696 (seed 20261125), I named the MeSH
descriptor for the analyte where one specific heading exists (137). I checked each name against PubMed's
query translation. Each descriptor got a PubMed search `<descriptor>[Majr] AND english[la]`, 10 results, requested from 2015 on (some older papers came back anyway)
(`data/seed_pmids.tsv`; descriptors in `data/mesh_map.json`).
Two concepts were dropped at search time: IDO (the heading translated to tryptophan oxygenase) and
proinsulin (results were insulin papers). Title, abstract and MeSH indexing came from the PubMed
connector. Abstracts under 200 characters were dropped.

**Labels.** An abstract is relevant to a concept when NLM's MeSH indexing of it includes the concept's
descriptor. The indexers never saw this experiment, so the labels are independent of the tagger.
Five concepts whose named descriptor appeared in fewer than half of their own seed abstracts were
dropped rather than relabelled: APP (indexed as Amyloid beta-Peptides), HER2 (Erb-b2 Receptor Tyrosine
Kinases), GAPDH, LDL receptor and PARP. That leaves 130 concepts, split 39 dev and 91 test at the concept level.
Median 9 relevant abstracts per concept. `data/concepts.tsv` lists each concept with its split, descriptor and query forms.

**Queries.** Up to three per concept, from three naming authorities: the MSD catalog name, the HGNC
symbol, and the MeSH heading. There is no symbol query where the concept has no HGNC mapping, or where the
automatic mapping is wrong (AGP→ATP5MK, IgA→CD79A, tPA→HADHA).

**Arms.** tag_jaccard and tag_dot over binary concept sets. BM25 is Okapi (rank_bm25) over lowercased `\w+` tokens.
tfidf_wc is word 1–2-gram plus char_wb 3–5-gram TF-IDF, sublinear, stacked, L2-normalized, cosine.
RRF uses k=60. Ties break by a fixed random permutation (seed 20261125).

## Caveats

- The symbol query and the alias map share a source (HGNC), so the symbol-form win is partly the
  tagger reading back its own dictionary. The catalog-name win has the same issue in weaker form, since catalog
  names are also aliases. The MeSH form is the only query authority the tagger was not built from,
  and there the tag vector does not win.
- The corpus is small (1,144 abstracts) and every abstract was chosen because it is about one of these
  analytes. A real corpus has far more abstracts that mention no catalog analyte (77 here have no tag)
  and more that mention several. Expect lexical arms to fare better on precision there.
- The descriptor mapping is mine, checked against PubMed's translation and the seed abstracts' indexing.
  Muninn `generate` (Gemini) timed out, so no second model did it independently.
- One of the ten helper jobs that copied PubMed records stopped early, so 113 of the 1,329 seed PMIDs
  (low IDs, papers from about 2008 to 2016) never arrived. The surviving concepts still have ≥3 relevant abstracts each.
- The case rule now admits `kit` and `met` in any case because the catalog spells them that way. In
  abstracts, "kit" and "met" also occur as plain English words.
- `/`-splitting the catalog names turns influenza strain names into keys like `massachusetts`; 30 abstracts carry
  that spurious tag. No flu concept is in the fixture, so it only adds noise to Jaccard denominators.

## Files and rebuild

```
python3 build_vocab.py hgnc_complete_set.txt        # HGNC complete set, sha256 in Method
python3 build_fixture.py <PubMed metadata files>    # get_article_metadata JSON or "pmid ||| title ||| mesh;... ||| abstract" lines
python3 evaluate.py dev && python3 evaluate.py test
```

`data/vocab.json` (4.9 MB) and `data/fixture.json` (abstract text) are not committed. The fixture is pinned
by `data/seed_pmids.tsv`, `data/mesh_map.json`, and the sha256 of its sorted docs,
`e8a30eee0bf1167c1b800959159647e00da81b902709b2e9e7e74a4411c04d43`. `data/concepts.tsv` lists the 130
labelled concepts with split, descriptor and query forms. `results/summary_test.json` holds every arm
as [R@10, lo, hi, MRR, lo, hi]. Titles, abstracts and MeSH indexing are from PubMed (NLM).
