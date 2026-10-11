# embeddinggemma2-remex-quant: remex, remax and Matryoshka on EmbeddingGemma 2 text vectors

**Started:** 2026-10-10 · **Status:** done (SciFact, NFCorpus) · **Runtime:** encode 57 + 46 min, bench under 30 s per corpus, 4 vCPU

## Answer

Does Google ship vector quantization for EmbeddingGemma 2, and if not, how far can the
text-only model's 768-d vectors be compressed? Google ships none: the card, docs and
launch post offer Matryoshka truncation to 512/256/128 dims ("up to a 6x reduction")
and weight quantization only. On SciFact and NFCorpus (fp32 nDCG@10 0.8505 / 0.3688),
the vendor's 6× point (fp32 d=128, 512 B) loses −0.0558 / −0.0652, while remex
centered 4-bit on all 768 dims (384 B, 8×) is +0.0015 / +0.0012 and centered 2-bit at
d=512 (128 B, 24×) is −0.0067 / −0.0085. Truncating 768→512 is free on this model, so
the two widths tie at matched bytes; below 512, spending bytes on bits beats keeping
dimensions. The vectors share a large common direction (‖mean‖ 0.845 / 0.866), so
centering matters at 1 bit (+0.0185 / +0.0169) and not at 2+; at 1 bit,
sentence-transformers' plain binary with a float query ties centered remex.

What would change it: a multimodal corpus (the card says 128d hurts multimodal quality
more), a larger or non-English corpus, or a second rotation seed per cell moving the
1-bit gaps, which sit inside ±0.02.

## Findings


1. **The vendor's 6× point loses 0.056–0.065 nDCG@10; remex at 8× loses nothing.**
   fp32 d=128 (512 B) is −0.0558 / −0.0652. remex centered 4-bit at the full 768 dims
   (384 B) is +0.0015 / +0.0012, and beats fp32 d=128 by +0.0573 [+0.0379, +0.0783] /
   +0.0663 [+0.0507, +0.0833]. At 128 B (24×), remex centered 2-bit d=512 still beats the
   vendor's 512 B point by +0.0492 [+0.0309, +0.0687] / +0.0567 [+0.0407, +0.0737].
   (`pairs.txt`, Round 1 table)

2. **Truncating 768 to 512 dims is free on this model, and truncating further is not.**
   fp32 d=512 is +0.0002 / −0.0009 with top-10 overlap 0.89 / 0.87. So at matched bytes
   d=512 and d=768 trade evenly: at 192 B the best 2-bit d=768 arm minus 3-bit d=512 is
   −0.0014 [−0.0186, +0.0150] SciFact and −0.0032 [−0.0137, +0.0059] NFCorpus
   (`report_*.txt`), and the Pareto frontier alternates between the two widths. At d=256 and below, quantizing a
   wider vector wins: 2-bit d=512 over 4-bit d=256 at 128 B is +0.0039 [−0.0088, +0.0175]
   / +0.0120 [+0.0026, +0.0219], and d=128 reaches the frontier only below 64 B, where
   little else fits.

3. **Centering is the remex setting that matters at 1 bit, and buys nothing at 2 bits
   and up.** At 96 B centered remex 1-bit beats plain remex 1-bit by +0.0185
   [+0.0005, +0.0375] / +0.0169 [+0.0033, +0.0307]. At 2 bits the two are within 0.001
   on both corpora. The plain 1-bit loss here (−0.030 / −0.038) is a fifth of what it
   was on the mxbai-edge and pplx tokens (−0.16 to −0.21), at a lower predictor (0.85
   vs 0.94), so the predictor still ranks the damage.

4. **At 32× sentence-transformers' binary with a float query ties centered remex.**
   ST binary asym against remex centered 1-bit: −0.0105 [−0.0281, +0.0071] SciFact,
   −0.0009 [−0.0126, +0.0107] NFCorpus. ST binary keeps the coordinates in the model's
   own basis and needs no stored mean. Both beat plain remex 1-bit and remax k=1 (remax:
   −0.0332 [−0.0547, −0.0125] / −0.0179 [−0.0319, −0.0045] under centered remex 1-bit).
   remax here is uncentered; METHODS.md has the rule that a centered remax index needs a
   centered query too, which this grid does not test. A binarised query (sym) costs
   another 0.024–0.026 for ST binary and 0.035–0.038 for remax k=1.

5. **remex 1.2.0's centered nested-precision fix holds on a second model.** Centered
   8-bit codes read at 2 bits score within 0.005 of a direct centered 2-bit encode
   (+0.0032 [−0.0073, +0.0138] / −0.0046 [−0.0114, +0.0018]). Uncentered nesting still
   costs at 2 bits on NFCorpus: −0.0141 [−0.0235, −0.0052] against a direct 2-bit encode.
   An 8-bit Lloyd-Max code's top bits are not the 2-bit Lloyd-Max code.

6. **The seed floor is small.** Haar seeds 0 and 1 land 0.0018 apart at 2 bits and
   0.0000 at 1 bit on SciFact, 0.0023 and 0.0065 on NFCorpus. The mxbai-edge swing of
   0.108 between seeds came from a 0.94 predictor; at 0.85 it does not appear.

7. **int8 and fp16 are free.** The global-scale int8 point estimate sits above fp32 on
   both corpora (+0.0036 [+0.0001, +0.0081] SciFact). The SciFact interval clears zero
   by 0.0001 and NFCorpus's does not, so read it as noise.

**Recipe for this model:** remex centered 4-bit at d=768 (384 B) is lossless; 2-bit at
d=512 or d=768 (128–192 B) costs under 0.01; below 128 B the choice is a 1-bit code at
d=768 (centered remex or ST binary with a float query, both about −0.01 to −0.02). Use
the vendor's truncation only to go from 768 to 512, and quantize after it.

## Method

- **Model:** `google/embeddinggemma-2` through sentence-transformers 6.1.0 (transformers
  5.19.0, torch 2.14.1 CPU), loaded text-only the way the card says:
  `config_kwargs={"vision_config": None, "audio_config": None}` (271.0M parameters).
  float32 compute; the card warns float16 inference overflows.
- **Prompts:** the card's. Documents `title: {title} | text: {text}` (`title: none` when
  empty); queries use the `Retrieval-query` prompt, `task: search result | query: `.
  `max_seq_length` 8192, so no document is truncated.
- **Corpora:** BEIR SciFact (5,183 docs, 300 test queries) and NFCorpus (3,633 docs,
  323 test queries with a relevant document), from `BeIR/*` on the Hub. NFCorpus qrels
  are graded and nDCG uses the grades.
- **Scoring:** brute-force dot product over every document; no candidate stage.
- **Harness:** `bench.py` reuses `mxbai-edge-remex-quant`'s metrics and ranking and
  `neomme-remex-quant`'s paired bootstrap. New here: every codec crossed with every
  Matryoshka width.
- **Codecs:**
  - Matryoshka truncation to 512/256/128, renormalised, at fp32 (the vendor lever).
  - fp16; int8 with one global scale; sentence-transformers `quantize_embeddings` int8
    (per-dimension ranges calibrated on the corpus) and binary (sign of each
    coordinate, no rotation), scored with a float query (asym) or a binarised one (sym).
  - remex 1.2.0 `Quantizer` (randomized Hadamard rotation, renorm on) at 8/4/3/2/1 bits
    for each width, plain and centered (`mean=corpus_mean(X)`, one mean per width).
    remex has no 5–7-bit widths. Nested precision (encode once at 8 bits, read at
    4/2/1), plain and centered. Haar rotation at seeds 0 and 1 for the 2- and 1-bit
    seed floor.
  - remax 0.3.0 `StackedSignBitQuantizer`, k=1 asym and sym, k=2 asym, uncentered.
- **Bytes:** per document as stored. Every vector is unit-norm before and after
  Matryoshka renormalisation, so remex's 4 B per-vector norm is not counted. Centered
  rows also exclude the one stored mean (3 KB for the whole corpus).
- **Statistics:** nDCG@10 against qrels, R@100. Per-query ΔnDCG@10 against fp32 d=768,
  5,000-resample paired bootstrap, 95% CI. `overlap10` in the JSON is top-10 agreement
  with the fp32 d=768 ranking.

## Log

### Round 1 — 2026-10-10: prior-art check, encode, full grid

**Asked:** Google released EmbeddingGemma 2 on 2026-10-06: one 768-d embedding space for text,
images, video and audio, 740M parameters with every encoder loaded and 270M text-only.
Oskar asked whether Google offers vector quantization for it, and if not, to give the
text-only model the remex treatment.

Google ships no quantized vectors for it, and publishes no quantized-vector results. The
[model card](https://huggingface.co/google/embeddinggemma-2),
[the docs](https://ai.google.dev/gemma/docs/embeddinggemma/model_card_2) and the
[launch post](https://blog.google/innovation-and-ai/technology/developers-tools/embeddinggemma-2/)
offer one size lever, Matryoshka truncation to 512, 256 or 128 dims ("up to a 6x
reduction in vector storage costs"). The only quantization they mention is of the
weights, for on-device RAM ("~191MB active RAM for text-only weights"). The Hub lists 59
community quantizations; all are weight formats. No technical report is linked. So the
comparison is the vendor's truncation against remex, remax and sentence-transformers'
own `quantize_embeddings`, at matched bytes per document.

**Result table.** fp32 d=768: SciFact nDCG@10 **0.8505** (R@100 0.980), NFCorpus **0.3688** (R@100
0.365), 3,072 B per document.

The vectors sit in a cone. The corpus mean of the unit vectors has norm 0.845 on
SciFact and 0.866 on NFCorpus; random document pairs have mean cosine 0.715 and 0.750.
remex warns about this on every uncentered encode ("These N vectors share a large
common direction"). The same predictor read 0.94 on mxbai-edge tokens and 0.945 on pplx
tokens, where plain remex collapsed at 1 bit.

Each cell is nDCG@10 · Δ vs fp32 d=768 [95% CI].

| arm | B/doc | × smaller | SciFact | NFCorpus |
|---|---:|---:|---|---|
| fp32 d=768 | 3072 | 1× | 0.8505 (ref) | 0.3688 (ref) |
| fp16 d=768 | 1536 | 2× | 0.8506 · +0.0001 [+0.0000, +0.0004] | 0.3679 · -0.0008 [-0.0031, +0.0008] |
| ST int8 d=768 | 768 | 4× | 0.8500 · -0.0004 [-0.0013, +0.0000] | 0.3693 · +0.0005 [-0.0005, +0.0015] |
| int8 global d=768 | 768 | 4× | 0.8541 · +0.0036 [+0.0001, +0.0081] | 0.3706 · +0.0018 [-0.0007, +0.0050] |
| remex 8-bit d=768 | 768 | 4× | 0.8494 · -0.0011 [-0.0037, +0.0003] | 0.3688 · +0.0001 [-0.0029, +0.0035] |
| fp32 d=512 | 2048 | 2× | 0.8507 · +0.0002 [-0.0082, +0.0091] | 0.3679 · -0.0009 [-0.0059, +0.0039] |
| fp32 d=256 | 1024 | 3× | 0.8375 · -0.0130 [-0.0244, -0.0016] | 0.3491 · -0.0197 [-0.0292, -0.0110] |
| fp32 d=128 | 512 | 6× | 0.7946 · -0.0558 [-0.0761, -0.0371] | 0.3036 · -0.0652 [-0.0824, -0.0493] |
| remex centered 4-bit d=768 | 384 | 8× | 0.8520 · +0.0015 [-0.0028, +0.0061] | 0.3699 · +0.0012 [-0.0022, +0.0044] |
| remex centered 4-bit d=512 | 256 | 12× | 0.8488 · -0.0017 [-0.0095, +0.0061] | 0.3676 · -0.0011 [-0.0066, +0.0040] |
| remex centered 2-bit d=768 | 192 | 16× | 0.8478 · -0.0026 [-0.0130, +0.0078] | 0.3626 · -0.0062 [-0.0130, +0.0000] |
| remex 2-bit d=768 | 192 | 16× | 0.8486 · -0.0018 [-0.0124, +0.0087] | 0.3616 · -0.0072 [-0.0151, +0.0005] |
| remex 3-bit d=512 | 192 | 16× | 0.8500 · -0.0005 [-0.0121, +0.0113] | 0.3658 · -0.0030 [-0.0103, +0.0046] |
| remex centered 2-bit d=512 | 128 | 24× | 0.8438 · -0.0067 [-0.0160, +0.0030] | 0.3603 · -0.0085 [-0.0162, -0.0010] |
| remex centered 1-bit d=768 | 96 | 32× | 0.8388 · -0.0116 [-0.0264, +0.0040] | 0.3477 · -0.0211 [-0.0309, -0.0117] |
| remex centered 3-bit d=256 | 96 | 32× | 0.8304 · -0.0200 [-0.0331, -0.0074] | 0.3485 · -0.0203 [-0.0308, -0.0105] |
| ST binary d=768 (asym) | 96 | 32× | 0.8283 · -0.0222 [-0.0359, -0.0091] | 0.3467 · -0.0220 [-0.0318, -0.0128] |
| ST binary d=768 (sym) | 96 | 32× | 0.8048 · -0.0457 [-0.0639, -0.0288] | 0.3209 · -0.0479 [-0.0615, -0.0362] |
| remex 1-bit d=768 | 96 | 32× | 0.8203 · -0.0301 [-0.0474, -0.0130] | 0.3308 · -0.0380 [-0.0512, -0.0251] |
| remax k=1 d=768 (asym) | 96 | 32× | 0.8056 · -0.0449 [-0.0645, -0.0253] | 0.3298 · -0.0390 [-0.0526, -0.0262] |
| remex centered 1-bit d=512 | 64 | 48× | 0.8265 · -0.0240 [-0.0419, -0.0071] | 0.3421 · -0.0267 [-0.0367, -0.0171] |
| remex centered 2-bit d=256 | 64 | 48× | 0.8201 · -0.0303 [-0.0473, -0.0148] | 0.3366 · -0.0322 [-0.0440, -0.0213] |
| remex centered 2-bit d=128 | 32 | 96× | 0.7807 · -0.0697 [-0.0932, -0.0477] | 0.2918 · -0.0770 [-0.0960, -0.0587] |
| remex centered 1-bit d=256 | 32 | 96× | 0.7713 · -0.0791 [-0.1040, -0.0553] | 0.3097 · -0.0590 [-0.0758, -0.0437] |

The full grid (63 arms per corpus, including every bits × width cell, nested reads, the
seed floor and remax at every width) is in `results_<corpus>.json`;
`report_<corpus>.txt` lists the Pareto frontier and `pairs.txt` the paired comparisons
quoted in Findings.

**Caveats at the time:**

- **SciFact fp32 0.8505 is high.** It is the highest fp32 SciFact nDCG@10 measured in
  this repo (pplx-embed-v2-late 0.6B: 0.7802; NeoMME late: 0.7198), from a 270M model.
  The card publishes MTEB aggregates only, so this is not checked against Google's own
  number. The harness is the one that reproduced mxbai-edge's and bge-small's card
  values. SciFact's training split shares its corpus with the test split, and a model
  trained on MTEB-adjacent data can have seen it; the deltas are relative to the same
  vectors, so they hold either way.
- Text only. The model's multimodal space, where the card says 128d "degrades
  multimodal quality substantially", is untested.
- Two English corpora, both small (3.6k–5.2k documents). The centroid-free codecs
  (remex, ST) do not change with corpus size; the one stored mean is 3 KB regardless.
- One rotation seed per remex/remax cell outside the seed-floor rows.
- Bytes stored were measured; query latency was not.

**Files:**

- `encode.py` — encodes both corpora with the card's settings, resumable per 128-doc
  chunk. Writes `.work/<corpus>/` (gitignored).
- `bench.py` — the grid; writes `results_<corpus>.json` and per-query
  `results_<corpus>_perquery.npz`.
- `report.py` — Pareto frontier and matched-byte comparisons (`report_<corpus>.txt`).
- `pairs.py` — the named paired comparisons quoted in Findings (`pairs.txt`).
- `chart.py` — the inline-SVG Pareto charts for the blog post (`chart_<corpus>.svg`).
- `recheck.py` — checks every number in this file against the artifacts.
