# Pure RM3 / plain BM25 on muninn — the platform-less site-search floor

**Question.** Site search has no inference platform (can't host a model) and no
agent (raw user query, no Claude to expand it). So the embedding path (Jina→remax,
needs a platform) and the skill path (lexical+agent-expansion 1.00, needs an
agent) are both out. What does model-free, agent-free retrieval actually deliver?
Test pure RM3 (corpus-driven pseudo-relevance feedback) vs plain BM25.

**Method.** Reconstruct the 73-post corpus from `muninn.kb`, build a lexical
index (`build_lexkb.py`) at whole-doc and 500-char granularity, run `search.py`
on the 5 phase-0 gold queries — raw query as the only term (the site-search case)
— with and without `--rm3`. Score R@5/@10 (distinct-post). Everything runs in
the Worker at deploy time; no model, no API, no agent.

## Results

| method (no agent, no inference, no per-query cost) | R@5 | R@10 |
|---|---|---|
| **BM25 raw, whole-doc** | **0.833** | **1.000** |
| BM25 + RM3, whole-doc | 0.833 | 0.900 |
| BM25 raw, 500-char | 0.800 | 0.933 |
| BM25 + RM3, 500-char | 0.800 | 0.933 |

Baselines (same gold/corpus):

| | R@5 | R@10 | available to site search? |
|---|---|---|---|
| lexical + agent expansion | 1.00 | 1.00 | ❌ needs a Claude agent |
| Jina-q4 → remax d512/k4 | 0.833 | 1.00 | ❌ needs an inference platform |
| Jina-fp32 full-float | 0.90 | 1.00 | ❌ needs a platform |

## Findings

1. **RM3 does not help** — identical R@5 to raw BM25, and it *hurt* R@10 on
   whole-doc (1.000→0.900). On a small corpus with in-vocab queries, pseudo-
   relevance feedback injects more noise than signal. Don't ship it.
2. **Plain BM25 (whole-doc) ties the embedding stack** — 0.833/1.00, equal to
   Jina-q4→remax and just under Jina-float's 0.90, at **zero inference platform,
   zero agent, zero per-query cost**, running entirely in the Worker over a small
   KV index. The only thing that beats it is the agent path site search can't use.
3. Whole-doc > 500-char here (0.833/1.00 vs 0.800/0.933) — fewer, larger units
   help BM25 on this small corpus.

## The honest caveat (the residual Gemini actually buys)

These are 5 **in-vocabulary acceptance** queries. Phase-0's "paraphrase frontier"
finding holds: the case where embeddings genuinely win is a **vocabulary-divergent**
query — the searcher's words don't appear in the relevant post. BM25 (and RM3)
cannot bridge that lexical gap; an embedder (or agent expansion) can. So BM25's
0.833 is the *in-vocab* number; real-world paraphrase/concept queries will be
weaker. That gap is precisely what a hosted embedder (Gemini) buys — and **no
platform-free option closes it** (RM3 was the statistical hope; it didn't).

## Bottom line / recommendation

For platform-less, agent-less muninn site search, **plain BM25 in the Worker is
the pragmatic answer**: 0.833/1.00 on in-vocab queries, matching the embedding
stack, needing no model / API / agent. Drop Gemini *if* muninn searches are mostly
keyword-ish (in-vocab). Keep Gemini only if searches are paraphrase-heavy — that's
the one thing BM25 can't do and no platform-free path recovers. RM3 is not worth
shipping either way.

Decision input needed: the actual query mix (keyword vs concept) muninn sees.

## Reproduce
```bash
python bench.py   # reconstructs corpus from muninn.kb, builds index, BM25 ± RM3
```
