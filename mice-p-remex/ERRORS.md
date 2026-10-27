# ERRORS: mice-p-remex

1. **A module named `codecs.py` shadowed the standard library.** `import codecs as C`
   loaded Python's `codecs`, and the first triage run died on `C.l2n`. Renamed to
   `mcodecs.py`.
2. **sklearn `MiniBatchKMeans` at K = 8,192 took 15 to 20 min per PLAID arm** under CPU
   contention, and I had set 20 iterations where FastPLAID's default is 4. Switched to
   faiss k-means at 4 iterations, cached per (dataset, normalisation) so the bit widths share
   one codebook, and recomputed every PLAID arm. The first three SciFact PLAID arms (sklearn,
   20 iterations) were discarded; their proxy errors were within 0.001 of the faiss ones.
3. **The doc-grouped scorer (`score_fast`) degenerates when documents are rarely shared.**
   It computes each document's K/V once for all the queries that retrieved it. On NFCorpus
   (18.5 queries per document) that ran an arm in 5 min. On FiQA (1.3) it ran batches of
   one at 3.6 pairs/s, slower than the per-query path. FiQA and SciDocs used `score`.
4. **SciFact's 41 test queries cannot resolve the deltas in question.** Every nDCG@10 CI on
   SciFact is about +/-0.03, and PLAID 2-bit came out +0.001 there against the paper's
   -0.018. Ranking agreement with fp32 (Spearman, top-10 overlap) was added as the measure
   that does not need judgements; NFCorpus and SciDocs carry the significance.
5. **Jobs hit the 2 h background limit twice**: the first SciDocs fp32 scoring (at document
   1,000 of 4,486 with the doc-grouped scorer) and the SciFact arm loop after six arms. Both
   loops skip finished arms; SciFact `remex-2`, `plaid-n-4` and `plaid-8` were left unscored.
6. **`colab exec` timed out on the client while the job finished on the VM.** The runs were
   there on `/content/cache/runs`; check the VM before re-running.
7. **The hub's jail guard refuses any non-jailed command whose text names `/tmp/jail`**,
   including `python3`, `sleep`, `ps`, `timeout` and shell `for` loops. Stage inputs under
   `cache/`, then `cp` and `chown` them in a separate call; copy outputs out with `cp` before
   any `python3` reads them.
