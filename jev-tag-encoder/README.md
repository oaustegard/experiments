# Jev as a 256-tag document encoder

Question: can TypeSafe's Jev, asked 256 `noul` questions ("The document is about <tag>.") in one call, produce a useful named, calibrated 256-dim document vector?

Origin: hev's Jev reranker post (github.com/hev/reranker). Same Noul-per-item shape, pointed at a fixed taxonomy instead of candidate documents.

## Probe results (claude.ai, 2026-09-22, via Muninn gateway op=jev, jev-1.13.0)

- 256 nouls fit in one call: ~3,900 input / 4,868 output tokens per doc (~19 output tokens per answer).
- Taxonomy: `tags.txt`, 16 lines x 16 tags; question id `tNNN` = flattened index.
- Doc A (Pogacar Alpe d'Huez news, W/kg, doping debate, 120 g/h fueling) >=0.5: cycling .99, specific named person .98, Europe .87, news report .87, physical measurements .80, controversy .76, statistics and data .58.
- Doc B (how-to fuel a 4 h ride, 60-90 g/h, zone 2) >=0.5: tutorial .98, cycling .97, fitness training .96, nutrition .95.
- Background tags 0.00-0.03. Shared bits 1, Hamming 9, cosine 0.41.
- Misses: phrasing "is about" gates on primary topic (doping -> pharmacology .14; fueling in A -> nutrition .23). "a question being asked" is noisy (.49, .37 with no question present) - reword or drop.

## Task for CCotw

Run it at corpus scale from code (the claude.ai session could only paste questions by hand).

1. Jev client: call Jev directly (TypeSafe API key; check claude-workspace env / proxy.env for a TYPESAFE/JEV key; if absent, stop and report). Encode with one call per doc, all 256 nouls. Cache vectors to parquet keyed by doc id. Record latency p50/p95 and tokens.
2. Taxonomy fit check: 200 docs from a mixed corpus (e.g. 20 Newsgroups or AG News). Report per-tag fire rate at >=0.5, dead tags (never fire), always-on tags, and the tag correlation matrix top pairs.
3. Classification eval with labels: a multi-label set whose labels map to tags (arXiv categories subset, or EUR-Lex subset). Compare:
   - Jev zero-shot (tag prob >= 0.5, and best-threshold upper bound)
   - Dense embedding (gemini-embedding-2 or a local ST model) + per-label logistic probes trained on 50/200/500 labels
   Metric: micro/macro F1 with bootstrap CIs. This is the head-to-head the idea depends on.
4. Retrieval eval: one BEIR set (SciFact or FiQA). Doc side = Jev vectors; query side = same 256 questions over the query text ("The query asks about <tag>" variant too). Scores: dot of probs, Bernoulli log-likelihood, Hamming on 0.5 bits. Report nDCG@10 standalone and as a third RRF leg with BM25 + dense. Expect standalone to lose; the question is whether the leg adds anything.
5. Phrasing ablation on 50 docs: "is about" vs "mentions" vs "is substantially about". Report fire-rate shift and effect on step 3.
6. Determinism: re-encode 30 docs, report mean/max abs delta and bit flips.

Write RESULTS.md with numbers + CIs; no judge-produced rates. Commit to main in this folder.
