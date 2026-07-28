# LFM2.5-Embedding-350M as a local-CPU embedder for remax_kb

**Question.** remax_kb's two embedders both carry CCotw friction: `JinaONNXEmbedder`
pulls an 847 MB ONNX export, `GeminiEmbedder` needs `$GEMINI_API_KEY` + a network
round-trip per encode. Liquid AI ships `LFM2.5-Embedding-350M`, a 1024-d CLS-pooled
embedder built on the LFM2 backbone. Does it work as a third, fully in-process,
no-key, no-network embedder — and where does its retrieval land against the
`lexical-kb-phase0` baselines (lexical 1.00 / Jina full-float 0.90 mean R@5)?

## What was built

`lfm25_embedder.py` — an `LFM25Embedder` conforming to remax_kb's `Embedder`
protocol (`fingerprint()` + `encode(texts, *, prompt)`), loading the model via
`sentence-transformers` (`trust_remote_code=True`), CLS-pooled, L2-normalized,
`query:` / `document:` prefixes applied through ST's registered prompts.

### One real fix was required

The model's bidirectional remote code (`modeling_lfm2_bidirectional.py`) rebinds
`Lfm2ShortConv.slow_forward` to a handler written before transformers added the
`seq_idx` kwarg (variable-length sequence packing). transformers 5.12.1 passes
`seq_idx` through, so every forward raised
`TypeError: _noncausal_shortconv_forward() got an unexpected keyword argument 'seq_idx'`.

The embedding model *needs* its bidirectional code — CLS pooling under the base
model's causal attention is meaningless (token 0 attends only to itself) — so
bypassing remote code is not an option. Fix is at our layer, not the model's
cached files: `_patch_shortconv_kwargs()` wraps `slow_forward` to drop the
extra kwarg. Sound here because our batches are plain padded sequences;
`seq_idx` only matters for packed variable-length batches.

## Results

### 1. Integration smoke test — `run_remax_kb.py`

Packed remax_kb's `examples/tiny_corpus` (5 Federalist/Gettysburg files) through
the full `pack → SRHT rotate → 1-bit → Hamming-search` pipeline and ran the same
3 queries as `tests/test_retrieval.py`:

```
Q: What is a faction?                              -> federalist_10_factions.txt  PASS
Q: Why must government be controlled by checks…?   -> federalist_51_checks.txt    PASS
Q: What was said at the dedication of the…?        -> gettysburg.txt              PASS
=== 3/3 topical top-3 retrievals correct ===
```

Matches what the Jina torch embedder scores on the same three queries. The
local CPU embedder produces correct retrieval through the deployed 1-bit format.

### 2. Head-to-head on the real muninn corpus — `bench_muninn.py`

Same methodology as `lexical-kb-phase0/stage_b.py`'s embedding ceiling:
full-float cosine over the embedding KB's own 1238 chunks (73 posts), chunk hits
collapsed to distinct posts, R@5/R@10 against the identical topical gold from
`sweep.py` (5 acceptance queries).

| embedder | mean R@5 | mean R@10 |
|---|---|---|
| lexical (agent-expand + BM25) | **1.00** | **1.00** |
| Jina v5-nano full-float (768-d) | 0.90 | 1.00 |
| **LFM2.5-Embedding-350M full-float (1024-d)** | **0.73** | **0.83** |

Per-query (LFM2.5): rank-of-first-gold / R@5 / R@10

| query | rank | R@5 | R@10 |
|---|---|---|---|
| Q1 centered-simhash | 1 | 1.00 | 1.00 |
| Q2 memory-storage | 8 | 0.00 | 0.50 |
| Q3 failure-modes | 1 | 0.67 | 0.67 |
| Q4 compiled-transformer-mojo | 1 | 1.00 | 1.00 |
| Q5 atproto-bluesky | 1 | 1.00 | 1.00 |

**It lands below Jina, not between the baselines** — and it's the *larger* model
(350M / 1024-d vs Jina v5-nano's 768-d) yet weaker on this corpus. The standout
miss is Q2 (memory-storage → the Turso/`introducing-muninn` posts): first gold at
rank 8, so 0.00 R@5. The other four queries are perfect at R@5.

This is a clean apples-to-apples number: both rows are **full-float cosine over
the full embedding dim** (no remax truncation, no 1-bit), same 1238 chunks, same
topical gold, same distinct-post collapse as `stage_b.py`. So the non-Matryoshka
caveat (below) does **not** distort it.

Caveats: n=5 acceptance queries — directional, not significant; Q2 alone swings
the mean ~0.13–0.20. In-vocab queries only (the paraphrase frontier is separate).

**Cost:** embedding the 1238 chunks took **1172 s (~19.5 min, 1.1 chunks/s)** on
4 vCPU fp32 — the first run hit a 15-min `timeout`; embeddings now checkpoint to
`.corpus_vecs.npz`. The slowness is the model, and it's the lever below.

## Front-loading / inline quantization (per-chunk memory)

remax's 1-bit step is **corpus-global by construction** (`pack.py:268-273`):
`mean_full = vectors.mean(0); centered = vectors - mean_full; sign(rotate(...))`.
A chunk's final code can't be computed without the corpus mean, so quantizing
each chunk to 1-bit *independently up front* is impossible for this scheme; it is
inherently two-pass, and an embedder emitting 1-bit for remax to consume is a
non-starter (the rotation needs reals). "Per-chunk memory" splits into three:

- **Float accumulation buffer** `np.empty((N, full_dim), float32)` (`pack.py:256`)
  — O(N·full_dim), the at-scale killer. Levers: (a) **fp16/int8 buffer** —
  `pack.py:265` forces float32, but only the *sign of each rotated coord*
  survives, so fp16 inputs cost ~nothing (one-line pack change); (b) **streaming
  two-pass / Welford mean** — pass 1 mean in O(full_dim), pass 2 quantize inline,
  peak memory → O(codes)+one row (needs cached/mmap'd embeddings or 2× compute);
  (c) **truncate dim up front** — provably identical codes since
  `(v−mean)[:dim] == v[:dim]−mean[:dim]` and the quantizer already runs on the
  dim-truncation; 4× smaller buffer at dim=256. **But** LFM2.5-Embedding is not
  documented as Matryoshka, so its first 256 coords aren't importance-ranked — a
  pre-existing concern for remax_kb truncating any non-MRL model, independent of
  *when* you truncate.
- **Model weights** (919 MB fp32 — the dominant cost in-container and the cause of
  the 1.1 chunks/s). Lever: **int8 dynamic quant** (`torch.ao.quantization`) or
  GGUF Q4/Q8 — ~3–4× less model RAM, 2–4× faster CPU inference; outputs stay
  float so remax is untouched downstream. Highest-value lever *for this
  environment*.
- **Stored codes** — already 1-bit; nothing to do.

## How to reproduce

```bash
pip install transformers sentence-transformers bm25s
# remax + remax_kb on PYTHONPATH (src-layout); muninn.kb fetched to
# .spokes/muninn.austegard.com/knowledge/muninn.kb
python3 run_remax_kb.py     # tiny-corpus 1-bit pipeline smoke test
python3 bench_muninn.py     # full-float R@5/R@10 vs lexical & Jina baselines
```

## Practical upshot

A ~350M embedder runs in-process on container CPU with no API key and no network
after the one-time model pull — filling the gap between remax_kb's heavy-download
Jina path and its API-bound Gemini path, as a drop-in on the same `Embedder`
protocol. **But the convenience isn't free:** on this corpus it retrieves worse
than the smaller Jina v5-nano (0.73 vs 0.90 R@5) *and* embeds slowly (1.1
chunks/s fp32). So it's a reasonable fit where in-process / no-key / no-network
is the hard constraint and retrieval is tolerant (rough triage, offline builds),
not where retrieval quality is the priority — there, Jina remains the pick.
Next lever if pursued: int8 model quant to close the speed gap and shrink RAM
(see above); whether quality survives quant is an open question worth a re-run.
