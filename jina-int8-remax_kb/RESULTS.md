# int8-quantized Jina v5-nano for remax_kb

**Question.** The local-CPU bake-off (`lfm25-embedder-remax_kb`) showed LFM2.5-350M
is weak (0.73 R@5) and slow. Jina v5-nano is the better embedder (0.90 R@5) but
its cost is the **847 MB ONNX download**. So quantize the *good* model instead:
does int8 preserve Jina's retrieval while cutting size + CPU latency?

## What was done

- Downloaded the fp32 retrieval ONNX export (847 MB, SHA-verified against
  `JinaONNXEmbedder.release_sha256`).
- `quantize.py`: `quant_pre_process` (shape inference, 74.6 s) →
  `quantize_dynamic(weight_type=QInt8)` (12.4 s). Dynamic weight-only int8, no
  calibration data.
- `bench.py`: fp32 vs int8 on the real muninn corpus, **same methodology as
  `lexical-kb-phase0/stage_b.py`** — full-float cosine over the embedding KB's
  own 1238 chunks (73 posts), chunk hits → distinct posts, R@5/R@10 vs the
  topical gold from `sweep.py`. The int8 model still emits fp32 vectors; only its
  weights are quantized. `JinaONNXEmbedder` already accepts `model_path` /
  `tokenizer_path`, so no subclass was needed — just point it at the int8 file.

## Results

| variant | size | embed (1238 ch) | rate | R@5 | R@10 |
|---|---|---|---|---|---|
| Jina fp32 | 847 MB | 149 s | 8.3 ch/s | **0.90** | 1.00 |
| **Jina int8** | **212 MB** | **74 s** | **16.7 ch/s** | **0.83** | **1.00** |

Per-query R@5 (fp32 → int8): Q1 1.00→1.00, Q2 0.50→0.50, Q3 1.00→**0.67**,
Q4 1.00→1.00, Q5 1.00→1.00.

Reference baselines (same harness): lexical (BM25 + agent-expand) 1.00/1.00 ·
LFM2.5-Embedding-350M 0.73/0.83.

### Reading

1. **fp32 reproduces the prior 0.90/1.00 exactly** — the harness is consistent,
   the int8 delta is real signal not noise.
2. **int8 is 4.0× smaller and 2.0× faster, with R@10 unchanged at 1.00.** The
   R@5 drop (0.90→0.83) is *one query* — Q3's gold slips from rank ≤5 to the
   6–10 band. Perfect R@10 means int8 perturbs only fine-grained ordering, not
   gross recall.
3. **int8 Jina dominates the LFM2.5 local option on every axis**: R@5 0.83 vs
   0.73, size 212 MB vs 919 MB (fp32 RAM), rate 16.7 vs 1.1 ch/s. For a
   small/fast *local* embedder, int8 Jina is strictly the better path; LFM2.5's
   only edge (in-process, no separate download) is outweighed.

### Caveats

- n=5 acceptance queries — the R@5 delta is a single gold post crossing the
  rank-5 boundary. Directional, not significant.
- Dynamic QInt8 weight-only. Untried levers that might recover the R@5: per-
  channel quant (`per_channel=True`), `reduce_range=True`, QUInt8 activations, or
  static QDQ with calibration. Worth a sweep if the 0.07 matters.
- Speedup is on a 4-vCPU box without VNNI; AVX-512-VNNI hardware widens the int8
  gain.
- This is the full-float cosine *ceiling*. The deployed remax_kb path 1-bits
  these vectors — re-running through `pack()` would give the on-disk-format
  number (a possible follow-up; the ceiling is the right baseline comparison).

## Bigger corpus + sub-int8 sweep — "is int8 the floor if we 1-bit anyway?"

The muninn probe is 5 queries. To test on real qrels and answer whether
embedder precision below int8 matters (given remax 1-bits the vectors), ran
**BEIR NFCorpus** (corpus-first subsample: 600 docs, 120 test queries — NFCorpus
is dense, ~38 rel/query) measuring recall **two ways per variant**: full-float
cosine and the deployed remax 1-bit path (center→truncate dim=256→StackedSignBit
k=8→Hamming). `bench_nfcorpus.py`. (Resumable memmap checkpointing — background
jobs are reaped on idle in this session, so caches must survive mid-variant.)

Variants (sizes after the embedding-table workaround, below):

| variant | size | cos R@10 | cos R@100 | 1-bit R@10 | 1-bit R@100 | per-doc cos vs fp32 |
|---|---|---|---|---|---|---|
| fp32 | 847 MB | 0.242 | 0.534 | 0.222 | 0.477 | 1.000 |
| int8 (per-tensor dynamic) | 212 MB | 0.115 | 0.372 | 0.077 | 0.291 | **0.445** |
| **q4** (4-bit blk + int8 embed) | **170 MB** | **0.241** | **0.536** | **0.208** | **0.486** | **0.975** |
| q2 (2-bit blk + int8 embed) | 141 MB | 0.083 | 0.305 | 0.048 | 0.270 | 0.730 |

**Findings:**

1. **int8 is not the floor — it's the worst non-trivial rung.** q4 (blockwise
   4-bit, 170 MB) *matches fp32* on both cosine and 1-bit recall; per-tensor
   dynamic int8 (212 MB) *collapses* — its embeddings are only **0.445 cosine to
   fp32**. Granularity (32-element blocks) beats bit count: 4-bit-blockwise >
   8-bit-per-tensor, smaller *and* far more faithful.
2. **The int8 collapse is domain-dependent, not seq-length.** int8 held up on
   muninn (tech text, 0.83 R@5) but craters on NFCorpus (medical abstracts).
   Probe at max_length 256 vs 512: cos(fp32,int8) = 0.436 vs 0.407 — bad at both,
   so length is ruled out. This is the activation-outlier fragility of per-tensor
   int8: different vocabularies trigger outliers it can't represent. q4 (blockwise)
   is robust across both domains (0.975). **So prefer blockwise 4-bit outright —
   per-tensor int8 is an unreliable gamble for this model.**
3. **q2 is a step too far** (0.730 fidelity, recall halved). The practical
   quantization floor for this model sits at 4-bit.

### The embedding-table size workaround

`MatMulNBits` only quantizes MatMul nodes, leaving EuroBERT's large multilingual
token-embedding table (a `Gather`, ~400 MB fp32) untouched — so naive int4 was
**465 MB, bigger than int8's 212 MB**. Fix: low-bit the MatMuls, then int8
`quantize_dynamic` to mop up the embedding Gather. Result: **q4 = 170 MB, q2 =
141 MB**, both < int8, both verified running on CPU. Matmul-only sizes barely
differ (465 vs 437) — the int8 *embedding* now dominates, so it, not MatMul
bit-width, is the size floor. (3-bit is unsupported: ORT asserts bits ∈ {2,4,8}.)

## Reproduce

```bash
pip install onnxruntime onnx onnx_ir transformers
# fetch model.onnx (847MB, asset 417841350) + model/tokenizer.json from
# oaustegard/jina-v5-nano-mirror (both gitignored — regenerable)
python3 quantize.py          # -> model.int8.onnx (212 MB, whole-graph dynamic)
python3 quantize_lowbit.py   # -> model.q4.onnx (170), model.q2.onnx (141)
python3 bench.py             # muninn: fp32 vs int8 R@5/R@10
# BEIR NFCorpus auto-downloaded by bench_nfcorpus.py into data/
python3 bench_nfcorpus.py    # fp32/int8/q4/q2 — cosine + 1-bit recall
```

## Upshot

int8 dynamic quantization looked great on the muninn probe (212 MB, 16.7 ch/s,
−0.07 R@5) — but the bigger NFCorpus test overturns it: **per-tensor dynamic int8
is domain-fragile** (0.445 cosine to fp32 on medical text), while **blockwise
4-bit (q4, 170 MB) matches fp32 across domains** (0.975) and is *smaller*. So the
recommended quantized Jina is **q4, not int8** — and the answer to "is int8 the
floor before we 1-bit?" is no: 4-bit blockwise is both smaller and more faithful,
the practical floor is 4-bit (q2 halves recall), and the int8 *embedding table* —
not MatMul bit-width — is now the dominant size cost. Standing retrieval-quality
recommendation remains lexical + agent expansion; q4 Jina is the real-vector
fallback when you want small/fast and *reliable* across domains.
