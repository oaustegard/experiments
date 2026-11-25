# Predictions for the sparseup v2 arm (written before any encode, 2026-11-25)

Arm: `Linkup-Platform/linkup-sparseup-embed-v2`, run for real (PyTorch fp32, and the int8 weights of the
browser pack in `../../sparseup-v2/pack/`). Queries encoded with the "[Q]" prompt, abstracts (title + abstract)
with "[D]", max 512 tokens, dot-product scoring. Same 91 test concepts, 260 queries and split as RESULTS.md, on a
fixture rebuilt from E-utilities that has all 1,329 seed PMIDs (1,241 abstracts after the 200-character filter,
against the published run's 1,144). The baselines were re-run on the same fixture.

Test R@10, predicted:

| query form | sparseup fp32 | rrf(sparseup, tag) | rrf(sparseup, tfidf) |
|---|---|---|---|
| catalog name | 0.78 | 0.84 | 0.80 |
| HGNC symbol | 0.62 | 0.82 | 0.68 |
| MeSH heading | 0.70 | 0.68 | 0.70 |
| all | 0.70 | 0.78 | 0.73 |

1. sparseup beats BM25 and TF-IDF on MeSH-heading queries (expansion should bridge heading wording to
   abstract wording) by more than the CI.
2. sparseup trails the tag vector on symbol queries by more than the CI (the tokenizer probe found symbols
   split into short shared pieces).
3. rrf(sparseup, tag) is the best arm pooled on R@10.
4. int8 is within 0.01 of fp32 on every form.
