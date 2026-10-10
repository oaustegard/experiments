# Jina→remax as a Gemini replacement for muninn search (+ CF-native alternatives)

## Headline: our special case — Jina-q4 → remax — works

Jina-q4 → remax 1-bit on the muninn corpus (1238 chunks/73 posts), 5-query
phase-0 gold, vs the full-float ceiling and lexical:

| pipeline | R@5 | R@10 | bytes/doc |
|---|---|---|---|
| Jina-q4 full-float cosine | **0.900** | 1.000 | — |
| Jina-q4 → remax d=256 k=8 (shipped default) | 0.667 | 0.933 | 256 |
| **Jina-q4 → remax d=512 k=4** | **0.833** | **1.000** | 256 |
| Jina-q4 → remax d=768 k=2 | 0.800 | 1.000 | 192 |
| lexical (BM25 + agent-expand) | 1.00 | 1.00 | — |
| Jina-fp32 full-float | 0.90 | 1.00 | — |

1. **q4 is free on muninn** — full-float Jina-q4 = 0.900/1.000, identical to fp32.
   The deployable quantized runtime loses nothing.
2. **Jina→remax is a credible Gemini replacement** — at d=512/k=4 (256 B/doc),
   0.833 R@5 / 1.00 R@10, near the float ceiling. The *shipped* d=256/k=8 default
   is the weak config (0.667) — dims beat stacks here too (kb-k-sweep), so the fix
   is just the pack config, not the embedder.
3. Lexical still edges R@5 (1.00), but the embedding path holds R@10 perfectly.

(Gemini→remax, the production incumbent, couldn't be measured live — no AI-gateway
token this session. Jina-fp32 0.90 float is the embedding reference.)

## Practicality: indexing is free, only the query embed is hard

muninn search has two embedding moments, and they have *opposite* constraints:

- **Indexing (offline, GitHub Action `build-search-index.yml`):** embed the whole
  corpus → pack the `.kb` → upload to KV. No runtime limit — use Jina fp32/torch
  in CI. `remax_kb.pack` already does this; the manifest records the Jina model +
  the hosted q4 `release_url`. **Trivial.**
- **Query (online, in the Worker):** embed ONE query at request time. The Worker
  **cannot host the model** (~10 MB bundle / 128 MB memory / WASM-only; 170 MB q4
  won't fit), and Workers AI has no Jina. Viable paths, cheapest first:
  1. **CF Container running q4** (Worker → container via service binding). q4's
     170 MB fits a container; CPU decode ~0.1 s/query. CF-native, replaces the
     paid per-query Gemini call with fixed cheap compute. **Best fit for "our stack."**
  2. **Tiny external endpoint** (Fly.io/VM/HF endpoint) running q4; Worker calls
     it. Same idea, off-CF.
  3. Client-side (transformers.js) — ruled out: 170 MB browser download/user.

  This is exactly where the q4 work pays off: it makes the *online* embedder light
  enough (170 MB, CPU) to self-host cheaply, removing the Gemini per-query billing
  surface (the thing the Worker rate-limits today).

**Net:** Jina→remax is quality-credible (0.833/1.00 at d=512/k=4) and deployable —
index offline with Jina, serve queries from a CF Container/endpoint running q4.
The migration is a corpus re-pack (Jina space, d=512/k=4) + a query-embed service
swap; the KV 1-bit-code mechanism is unchanged.

---

## Sidebar: CF-native (Workers AI) embedders underperform

**Context.** muninn search (Cloudflare Worker `worker/index.js`) embeds each query
via a paid Gemini API call (`gemini-embedding-001`) through the CF AI Gateway,
then hybrid-searches the Gemini-built `.kb` in KV. Question: could a cheaper
Cloudflare-native embedder replace Gemini? Gate: it must not regress retrieval.

**Deployment reality (why not Jina-q4):** a 170 MB ONNX can't run in a Worker
(~10 MB bundle, 128 MB memory, WASM-only); Workers AI is a curated catalog (BGE,
bge-m3, embeddinggemma-300m, Qwen3 — no Jina, no BYO-ONNX). So the only
*zero-hosting* CF-native swap is a Workers-AI model. This bench tests whether one
matches the incumbent's quality.

**Method.** Embed the muninn corpus (1238 chunks / 73 posts) + the 5 acceptance
queries with each candidate; full-float cosine query→chunk, collapse to distinct
posts, R@5/@10 vs the model-independent topical gold from `lexical-kb-phase0`
(14 gold posts, all present). `embed_one.py` (one model per foreground call,
resumable — background runs got reaped and the 847 MB fp32 ONNX got SIGKILLed by
a sandbox arena cap).

## Results

| embedder | dim | R@5 | R@10 | Workers AI? |
|---|---|---|---|---|
| lexical (BM25 + agent-expand)* | — | **1.00** | **1.00** | n/a |
| Jina-v5-nano fp32* | 768 | 0.90 | 1.00 | no |
| bge-large-en-v1.5 | 1024 | 0.733 | 0.833 | ✅ |
| bge-base-en-v1.5 | 768 | 0.667 | 0.833 | ✅ |

\* baselines from `lexical-kb-phase0` (same gold/corpus).

## Findings

1. **The CF-native BGE models underperform the incumbent.** bge-large 0.733,
   bge-base 0.667 R@5 — vs Jina 0.90 and lexical 1.00. Swapping Gemini for a
   Workers-AI BGE model is a **quality regression** on this corpus.
2. **Among embedders, more dims helped** (bge-large 1024 > bge-base 768), but not
   enough to close the gap to Jina.
3. **Lexical still wins outright** (1.00) — consistent with the whole muninn
   embedding-free line. If the motive to drop Gemini is cost, the lexical +
   agent-expansion path beats trading down to a weaker embedder.

## Caveats / untested

- **n=5 queries** — directional, not significant.
- **embeddinggemma-300m** (Google, on Workers AI, 768-d) is **HF-gated** (401, no
  token this session) — as a Gemma-family model it might track Gemini better than
  BGE; the most interesting untested candidate.
- **bge-m3** (strongest BGE, on Workers AI, 1024-d, multilingual, 2.2 GB) untested.
- **No live Gemini** (no gateway token) — Jina-0.90 is the embedding reference;
  the production Gemini number on this exact gold isn't measured here.
- Matched-space requirement: any swap means **rebuilding the whole index** in the
  new model's space, not a query-side change.

## Bottom line

The easy, zero-hosting CF-native swaps (bge-base/large) **don't match** muninn's
current retrieval quality. A migration off Gemini isn't justified by these
candidates. Remaining hopes: embeddinggemma-300m (needs an HF token) and bge-m3.
The standing alternative — and the actual quality champion here — is the
embedding-free lexical path, not a weaker embedder.

## Reproduce
```bash
python embed_one.py BAAI/bge-base-en-v1.5  bge-base
python embed_one.py BAAI/bge-large-en-v1.5 bge-large
# needs .spokes/muninn.austegard.com/knowledge/muninn.kb + lexical-kb-phase0/sweep.py gold
```
