# Phase 0 — does agent-expansion + BM25 actually match embeddings?

The load-bearing study for the embedding-free KB (`creating-kb`). Run on the real
muninn.austegard.com corpus (73 posts, the same corpus behind the published
embedding `muninn.kb`). Three questions:

1. **Chunk size** — does lexical recall survive bigger chunks? (the null hypothesis)
2. **Head-to-head** — lexical (agent-expansion + BM25) vs the embedding *ceiling*.
3. **Paraphrase frontier** — where lexical leans entirely on expansion: what do
   embeddings still buy?

## Method

- **Corpus**: post text reconstructed from the embedding KB's own chunks
  (`reconstruct_corpus.py`), so the text is byte-identical to what embeddings
  indexed — chunk size is the only variable.
- **Gold**: per #76 query, the on-topic posts judged from each post's
  `description` meta (not body term-frequency), so gold doesn't pre-favour
  lexical matching.
- **Metric**: chunk hits → distinct posts; `rank` of first gold post, `R@5`,
  `R@10` over distinct posts. Gold is post-level.
- **Lexical**: `creating-kb` builder + `search.py`, agent-crafted `core`/`expand`
  per query (the consuming agent's job), raw query carried at floor weight.
- **Embedding**: full-float Jina v5-nano (768-d, cosine) — the *ceiling*,
  strictly stronger than the deployed 1-bit `muninn.kb` codes, so this is a
  conservative test for lexical. (Avoids the `remax` quantizer entirely.)

## 1. Chunk-size sweep (`sweep.py`) — null hypothesis confirmed

Gold-post retrieval at four chunk sizes, agent expansion, 5 acceptance queries:

| target_chars | chunks | avgdl (tok) | mean R@5 | mean R@10 |
|---:|---:|---:|---:|---:|
| 500       | 1237 | 69   | 0.87 | 1.00 |
| 1500      | 432  | 198  | 1.00 | 1.00 |
| 4000      | 174  | 490  | 1.00 | 1.00 |
| whole-doc | 73   | 1169 | 1.00 | 1.00 |

First gold post is **rank 1 in every configuration**. Chunk count falls **17×**
(1237 → 73) with **no recall loss** — R@5 actually *improves* with bigger chunks.

**Caveat (honest):** part of the R@5 gain is mechanical — at whole-doc, 1 chunk =
1 post, so top-5 chunks = 5 distinct posts; at 500-char, several top chunks share
a post, leaving fewer distinct posts in the top-5. The load-bearing finding is
the absence of recall *loss* at 17× fewer chunks. Lexical BM25 tolerates — even
prefers — big chunks, exactly because there is no vector centroid to dilute.
**Whole-document chunking is the right default for lexical KBs.**

## 2. Head-to-head vs the embedding ceiling (`stage_b.py`)

Lexical whole-doc vs full-float Jina, 5 acceptance queries (rank / R@5 / R@10):

| query | LEXICAL | EMBEDDING |
|---|---|---|
| Q1 centered-simhash | 1 / 1.00 / 1.00 | 1 / 1.00 / 1.00 |
| Q2 memory-storage | 1 / 1.00 / 1.00 | 5 / 0.50 / 1.00 |
| Q3 failure-modes | 1 / 1.00 / 1.00 | 1 / 1.00 / 1.00 |
| Q4 compiled-transformer-mojo | 1 / 1.00 / 1.00 | 1 / 1.00 / 1.00 |
| Q5 atproto-bluesky | 1 / 1.00 / 1.00 | 1 / 1.00 / 1.00 |
| **MEAN** | **1.00 / 1.00** | **0.90 / 1.00** |

On queries carrying domain vocabulary, **agent-expanded lexical matches — slightly
edges — the embedding ceiling.** Both reach R@10 = 1.00; lexical's R@5 is higher
(embedding ranked one memory-storage gold post 5th). These queries are favourable
to lexical, so read this as *"lexical does not lose,"* not *"lexical wins broadly."*

## 3. Paraphrase frontier (`paraphrase_probe.py`) — the honest limit

Lay-phrased queries with little/no corpus vocabulary, targeting known gold.
`LEX-raw` = no expansion; `LEX+exp` = agent expansion; values are rank / R@10:

| paraphrase query | LEX-raw | LEX+exp | EMBED |
|---|---|---|---|
| P1 remember-across-sessions | 4 / 0.50 | **1 / 1.00** | 3 / 1.00 |
| P2 demo-to-production | 1 / 1.00 | 1 / 1.00 | 1 / 1.00 |
| P3 weights-as-program | 2 / 1.00 | 2 / 1.00 | 4 / 1.00 |
| P4 decentralized-twitter | 1 / 1.00 | 1 / 1.00 | 1 / 1.00 |
| P5 forgetting-old-info | 1 / 0.50 | 1 / 0.50 | 3 / **1.00** |

- **Expansion is load-bearing and it works** — P1: raw 0.50 → expanded 1.00. The
  agent closed a gap embeddings would otherwise win.
- **Lexical+expansion ties or beats embedding on 4 of 5** paraphrase queries
  (P1/P3 lexical ranks the gold *higher* than embedding; P2/P4 tie).
- **P5 is what embeddings actually buy.** Even with expansion, lexical missed a
  conceptually-related but vocabulary-divergent gold post
  (`from-selective-consolidation…`) that the embedding caught. Expansion quality
  is the ceiling for lexical, and it does not always anticipate every relevant
  document's vocabulary. This is the residual embeddings cover.

## Verdict

For this corpus and query mix, **agent-expansion + BM25 is competitive with dense
embeddings** — at parity on in-vocabulary queries and recovering most of the gap
on paraphrase, while deleting the embedding model, the 440–847 MB asset, the
build-time embedding pass, and the runtime query-embedding call. The price is a
**recall residual on conceptually-related, vocabulary-divergent documents** that
expansion fails to anticipate (1 of 5 paraphrase queries here). That residual is
the precise, bounded answer to *"what did embeddings buy?"* — and it shrinks
further against the deployed 1-bit `muninn.kb` (this test used the stronger
full-float ceiling).

Practical implication: the embedding-free `.skill` is a sound default for
agent-driven retrieval; the RM3 fallback and, if needed, a small hybrid
re-rank are the levers for the paraphrase residual.

## Caveats / scope

- 5 acceptance + 5 paraphrase queries, post-level gold — small, coarse. Directional, not a benchmark.
- Gold is judgment-based (from descriptions). Reasonable, not adjudicated by a third party.
- Expansions were authored by the same agent evaluating — representative of real use, but not blind.
- Embedding = full-float ceiling; the shipped 1-bit codes would score somewhat lower.

## Repro

```bash
python3 reconstruct_corpus.py     # rebuild corpus/ from muninn.kb chunks
python3 sweep.py                  # Stage A: chunk-size sweep (no model)
python3 stage_b.py                # Stage B: head-to-head (needs Jina ONNX)
python3 paraphrase_probe.py       # paraphrase frontier (needs Jina ONNX)
```

Stage B / paraphrase need the Jina v5-nano ONNX (~847 MB, cached at
`~/.cache/remax_kb/jina-v5-nano/model.onnx`) + `.spokes/jina-v5-nano-mirror`
(tokenizer) + `onnxruntime tokenizers bm25s`. `corpus/` and the cached corpus
matrix are gitignored (regenerable).

## Addendum — recall unit ≠ reasoning unit (the noise objection)

Phase 0 measured retrieval *recall*, not the quality of what's handed to the
agent to reason on. Whole-doc hits are big (the muninn posts average ~1169
tokens; one is 17.6 KB) and query terms are sparse in them (~3–4% of tokens) —
so a raw whole-doc payload is mostly noise around the answer. Crucially, the
signal is *distributed*, not in one window: the densest 120-token window of a hit
holds only 29–42% of its query-term matches, so returning a single window would
drop most of the relevant material.

Fix (shipped in `creating-kb` v0.2.0): **decouple the retrieval unit from the
reasoning unit.** The searcher still *ranks* on the whole chunk (recall, Phase 0
results unchanged) but *returns* the query-densest **sentences** — extractive
multi-passage selection scored by query-term weight·idf, sentences from anywhere
in the doc, capped to ~1200 chars (`--snippet`, default on; `--snippet 0` for the
full chunk). Cross-runtime byte-identical (scores rounded; `test_parity.py`
covers the snippet path on a long doc).

Effect on the real corpus (whole-doc index): payload drops to 6–25% of the
document while the answer-bearing lines survive — e.g. "what does Muninn use for
memory storage" returns a 1.1 KB passage from a 17.6 KB post containing
`Turso/libSQL — 2,638 memories`. So whole-document chunking keeps its recall/cost
win *and* gives the agent focused context. Default chunking is now `--target-chars
0` (whole document).
