# Predictions, written 2026-11-25 before the fixture existed

Test split, R@10 (hits in top 10 / min(relevant, 10)) per query form. Written before any
retrieval run, dev or test.

| query form | tag_jaccard | bm25 | tfidf_wc | rrf(tag, best lexical) | tag coverage |
|---|---|---|---|---|---|
| catalog name (`TNF-α`, `GRO-α`, `HAVCR2/TIM-3`) | 0.70 | 0.60 | 0.65 | 0.72 | 0.97 |
| HGNC symbol (`TNF`, `CXCL1`, `HAVCR2`) | 0.70 | 0.45 | 0.50 | 0.72 | 1.00 |
| MeSH heading (`Chemokine CXCL1`) | 0.55 | 0.60 | 0.62 | 0.68 | 0.70 |

Claims, each falsifiable on the test split:
1. Tag-binary beats BM25 on catalog-name and symbol queries by more than its CI.
2. Tag-binary loses to the best lexical arm on MeSH-heading queries.
3. RRF(tag, best lexical) is the best arm over all forms pooled.
4. Tag-binary's three query forms score within 0.05 of each other where coverage is complete
   (surface variants collapse to one dimension); BM25's spread across forms exceeds 0.10.

Prior from 37dc9ebe (Muninn tags, 2026-09-22): 6 of 8 predictions wrong, all too low.
