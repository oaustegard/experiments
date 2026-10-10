# pplx-embed-v2-quant — remex, remax and PLAID on pplx-embed-v2-late-0.6b's token vectors

**Started:** 2026-10-10 · **Status:** done (SciFact) · **Runtime:** encode 2 h 18 min, bench 48 min, follow-ups 30 min, 4 vCPU

## Question

Perplexity released `pplx-embed-v2-late` on 2026-10-07 in two sizes, 0.6B and 9B, with
one shared embedding space, so a 9B-built index can be queried with the 0.6B model.
Both are ColBERT-style late-interaction models: one 128-d unit-norm vector per token,
scored with MaxSim.

Did Perplexity publish quantized-vector or Matryoshka results? No. The
[model card](https://huggingface.co/perplexity-ai/pplx-embed-v2-late-0.6b) and the
[launch post](https://www.perplexity.ai/hub/blog/multimodal-embeddings-beyond-a-single-vector)
report float MaxSim only. Neither mentions int8, binary, residual compression, token
pooling or truncated dimensions, and neither claims Matryoshka training. The post
says a technical report will follow "later this year". So: how far can remex and
remax compress these tokens, against PLAID's residual codec and against truncation?

## Setup

- **Model:** `perplexity-ai/pplx-embed-v2-late-0.6b` through sentence-transformers
  6.1.0 `MultiVectorEncoder` (transformers 5.19.0, torch 2.14.1 CPU), with the card's
  own settings: prompts `[Q] ` / `[D] `, query_length 1024, document_length 4096, no
  query expansion, document-side punctuation skiplist. Tokens are unit-norm to 2e-7.
  The 9B model was not run (CPU only), so cross-model indexing is untested here.
- **Corpus:** BEIR SciFact, 5,183 documents (300.9 tokens per doc, 1,559,411 in all)
  and 300 test queries (20.7 tokens per query).
- **Scoring:** brute-force MaxSim over every document, no candidate stage.
- **Harness:** `bench.py` reuses `mxbai-edge-remex-quant`'s metrics, bootstrap,
  PLAID codec and scan. Changes: d = 128, no dense comparator, k-means fit once
  for all PLAID widths, and new arms for truncation and centered remax.
- **Codecs:**
  - fp16; int8 with one global scale.
  - remex 1.1.0 `Quantizer` (randomized Hadamard rotation, renorm on). Encoded once
    at 8 bits and read at 4/3/2/1 bits by the Matryoshka bit nesting. Direct
    encodes at 2 and 1 bit, seeds 0 and 1. Centered mode (`mean=corpus_mean(T)`)
    at 4/2/1 bits, seeds 0 and 1.
  - remax 0.3.0 `StackedSignBitQuantizer` k = 1/2/4 with a float query (asym) and
    with a binarized query (sym), plain and centered.
  - PLAID-style residual compression as ColBERTv2 specifies it: K = 16,384 k-means
    centroids, a uint16 centroid id per token, and per-dimension residual buckets
    at 4/2/1 bits, plus centroid-only.
  - Matryoshka check: prefix truncation to 64 and 32 dims, re-normalised on both
    sides, alone and with remex at matched bytes.
- **Bytes:** per document as stored. Tokens are unit-norm, so remex's per-vector
  length is not counted. In centered mode the stored length is a function of the
  code and ‖x‖ = 1, so it can be recomputed at decode time. PLAID rows exclude its
  4.2 MB centroid table (809 B per doc amortised over this corpus).
- **Statistics:** nDCG@10 against qrels. Per-query ΔnDCG@10 against fp32 with a
  5,000-resample paired bootstrap, 95% CI.

## Results

fp32 MaxSim: nDCG@10 **0.7802**, R@100 0.9833, 154 KB per document.

| arm | KB/doc | × smaller | nDCG@10 | Δ vs fp32 [95% CI] |
|---|---:|---:|---:|---|
| fp16 | 77.0 | 2 | 0.7812 | +0.0010 [−0.0001, +0.0030] |
| int8 (global scale) | 38.5 | 4 | 0.7862 | +0.0060 [+0.0016, +0.0114] |
| remex 8-bit | 38.5 | 4 | 0.7807 | +0.0005 [−0.0010, +0.0025] |
| remex 4-bit (nested from 8) | 19.3 | 8 | 0.7784 | −0.0018 [−0.0093, +0.0048] |
| remex 3-bit (nested from 8) | 14.4 | 10.7 | 0.7702 | −0.0100 [−0.0222, +0.0012] |
| remex 2-bit, direct s0 / s1 | 9.6 | 16 | 0.7605 / 0.7579 | −0.0196 / −0.0223 |
| remex 2-bit (nested from 8) | 9.6 | 16 | 0.7314 | −0.0488 [−0.0712, −0.0290] |
| remex 1-bit s0 / s1 | 4.8 | 32 | 0.5744 / 0.6354 | −0.2057 / −0.1447 |
| **remex 4-bit centered** | 19.3 | 8 | 0.7832 | +0.0031 [−0.0013, +0.0078] |
| **remex 2-bit centered s0 / s1** | 9.6 | 16 | 0.7858 / 0.7802 | +0.0056 [−0.0027, +0.0152] / +0.0000 [−0.0090, +0.0096] |
| **remex 1-bit centered s0 / s1** | 4.8 | 32 | 0.7650 / 0.7610 | −0.0151 [−0.0288, −0.0014] / −0.0192 [−0.0330, −0.0057] |
| remax k=1 asym | 4.8 | 32 | 0.6457 | −0.1344 [−0.1663, −0.1039] |
| remax k=2 asym | 9.6 | 16 | 0.7272 | −0.0529 [−0.0751, −0.0324] |
| remax k=4 asym | 19.3 | 8 | 0.7632 | −0.0169 [−0.0321, −0.0025] |
| remax k=1 sym | 4.8 | 32 | 0.5612 | −0.2190 [−0.2567, −0.1818] |
| remax k=1 asym centered s0 / s1 | 4.8 | 32 | 0.7641 / 0.7314 | −0.0161 / −0.0488 |
| remax k=2 asym centered | 9.6 | 16 | 0.7766 | −0.0035 [−0.0178, +0.0107] |
| remax k=4 asym centered | 19.3 | 8 | 0.7797 | −0.0004 [−0.0122, +0.0111] |
| remax k=1 sym centered s0 / s1 | 4.8 | 32 | 0.7662 / 0.7201 | −0.0139 / −0.0601 |
| PLAID 4-bit | 19.9 | 7.8 | 0.7802 | +0.0000 [−0.0046, +0.0046] |
| PLAID 2-bit | 10.2 | 15 | 0.7780 | −0.0022 [−0.0097, +0.0054] |
| PLAID 1-bit | 5.4 | 28 | 0.7586 | −0.0215 [−0.0353, −0.0080] |
| PLAID centroid-only | 0.6 | 256 | 0.7241 | −0.0560 [−0.0813, −0.0332] |
| truncate to 64d, fp32 | 77.0 | 2 | 0.7652 | −0.0150 [−0.0292, −0.0027] |
| truncate to 32d, fp32 | 38.5 | 4 | 0.6774 | −0.1027 [−0.1338, −0.0732] |
| truncate 64d + remex 4-bit | 9.6 | 16 | 0.7534 | −0.0267 [−0.0425, −0.0130] |
| truncate 32d + remex 8-bit | 9.6 | 16 | 0.6762 | −0.1039 [−0.1349, −0.0744] |
| truncate 64d + remex 2-bit | 4.8 | 32 | 0.6525 | −0.1277 [−0.1593, −0.0976] |
| truncate 32d + remex 4-bit | 4.8 | 32 | 0.6353 | −0.1449 [−0.1790, −0.1131] |

The centered remax rows with a float query center the query too (`extra.py`).
`bench.py`'s first version of those two arms scored 0.06 and is superseded
(ERRORS.md #1).

## Findings

1. **The tokens sit in a narrow cone, so plain low-bit codes break.** The corpus mean
   has 0.945 of the average token's length, and two random tokens have cosine 0.892.
   This matches mxbai-edge-colbert (0.94 / 0.88), not NeoMME, whose 128-d tokens lost
   only 0.013 at 1 bit uncentered. Plain remex 1-bit costs −0.206 and −0.145 under
   two seeds, a 0.061 seed swing. remax k=1 with a float query costs −0.134 and with
   a binarized query −0.219. remex 1.1.0 raised its `AnisotropyWarning` on this
   corpus before any retrieval ran.

2. **Centered remex 2-bit is 16× smaller than fp32 with no measurable loss.**
   - 2-bit: +0.006 and +0.000 across two seeds, CIs spanning zero. 9.6 KB per doc
     against 154 KB.
   - 4-bit: +0.003.
   - 1-bit (32×): −0.015 and −0.019, with a 0.004 seed spread. Centering is worth
     +0.191 [+0.154, +0.229] at 1 bit over the uncentered code.
   - Uncentered remex is fine down to 4 bits (−0.002) and loses 0.020 at 2 bits.

3. **Centered remex ties PLAID at every width, at fewer bytes and with no centroid
   table.** Paired against PLAID at the same residual width:
   - 4-bit: +0.003 [−0.003, +0.010].
   - 2-bit: +0.008 [−0.002, +0.019].
   - 1-bit: +0.006 [−0.010, +0.023].

   remex saves PLAID's 2-byte centroid id per token (602 B per doc) and the 4.2 MB
   K = 16,384 table. On mxbai-edge, PLAID led at 1 bit by +0.029; here it does not.
   PLAID's centroid id alone (0.6 KB per doc) still scores 0.724, a useful floor for
   a candidate stage.

4. **remax needs centering, and at 1 bit it is seed-unstable even centered.** With
   the query centered too, remax k=2 (−0.004) and k=4 (−0.000) match fp32. At k=1,
   two seeds give −0.016 and −0.049 with a float query, and −0.014 and −0.060 with a
   binarized one. Centered remex 1-bit gives −0.015 and −0.019.
   - The likely cause is the score. A remax code has no residual length, so the
     score (q − μ)·ŝ drops μ·r, which varies per token on unit-norm input.
   - remex instead reconstructs μ + m·û with m solved for ‖x‖ = 1.
   - At seed 0 the two 1-bit codes tie (remax − remex −0.001 [−0.021, +0.019]); the
     seed spread separates them.

5. **The 128-d token space is not Matryoshka-nested, and quantizing beats truncating.**
   - Truncating to 64 dims costs −0.015 and to 32 dims −0.103, so the leading dims
     carry no concentrated signal.
   - At 9.6 KB per doc, centered remex 2-bit at 128 dims beats truncate-to-64 + 4-bit
     by +0.032 [+0.017, +0.050] and truncate-to-32 + 8-bit by more.
   - At 4.8 KB per doc, centered 1-bit (0.765) beats both truncated arms (0.653,
     0.635).

   This is the `quantize wide rather than truncate narrow` result again, on a model
   that never claimed truncation.

6. **remex's Matryoshka bit nesting costs 0.03 at 2 bits, and is broken with centering.**
   - Uncentered, reading 2 bits off the 8-bit code scores 0.731 against 0.761 for a
     direct 2-bit encode (+0.029 [+0.010, +0.050] for direct). The top bits of an
     8-bit Lloyd-Max code give a different partition from the 2-bit Lloyd-Max
     optimum. At 1 bit the two coincide, since both are the sign.
   - Centered, nesting is much worse: 2-bit 0.737 and 1-bit **0.527**, against
     0.786 and 0.765 direct.
   - The cause is in `remex/core.py` `_centred_lengths`. The stored length m solves
     ‖μ + m·û‖ = ‖x‖ for the full-precision direction û. `decode(precision=p)`
     reuses that m with the p-bit direction, so reconstructions land at the wrong
     length. With ‖μ‖ = 0.945 the solve is very sensitive to μ·û.
   - **The fix costs no bytes and closes the gap completely** (`renest.py`).
     ‖x‖ is recoverable from the full-precision decode (max error 1.2e-7 here, and
     it is 1 by construction for unit-norm tokens). Re-solving m for the p-bit
     direction brings the median length error from 0.0109 (1-bit) and 0.0071
     (2-bit) to zero. Nested decode then matches a direct centered encode at
     every width:

     | width | nested, stored m | nested, m re-solved | direct centered |
     |---|---:|---:|---:|
     | 8-bit (the encode itself) | 0.7811 | 0.7811 | — |
     | 4-bit | 0.7759 | 0.7814 | 0.7832 |
     | 3-bit | 0.7532 | 0.7780 | — |
     | 2-bit | 0.7373 | 0.7800 | 0.7858 / 0.7802 (s0 / s1) |
     | 1-bit | 0.5270 | 0.7650 | 0.7650 |

     A median length error of about 1% is enough to cost 0.24 nDCG@10 because
     MaxSim's per-document max compares tokens whose scores sit within a percent of
     each other. This is the same mechanism as the renorm bug fixed in remex#82.
   - **Fixed in remex `8e9134f`** (`Quantizer._centred_lengths_at`). Re-run on these
     tokens through the library: the 1-bit read scores 0.7650 and the 2-bit read 0.7800,
     matching `renest.py`.
   - With the fix, centered nesting is free, where uncentered nesting still costs
     0.03 at 2 bits. So one centered 8-bit encode can serve every width from 1 to 8
     bits.

## Recommendation for pplx-embed-v2-late tokens

Use remex centered at 2 bits: 16× smaller and indistinguishable from fp32 here. Use
centered 1-bit if 32× matters and −0.015 to −0.019 nDCG@10 is acceptable. Store one
512-byte corpus mean. With remex after `8e9134f` (unreleased; 1.1.0 and earlier
have the bug in finding 6), one centered 8-bit encode serves every width. On 1.1.0,
encode directly at the width you will serve. Avoid plain (uncentered) codes below 4 bits,
remax k=1, and truncation.

## Limits

- One corpus (SciFact, 300 queries) and text only. The model's main target is
  visual documents.
- The 0.6B model only. Whether a 9B-built index compresses the same way, and
  whether 0.6B queries still match quantized 9B tokens, is untested. Both models
  output 128-d normalised tokens from one space, so the cone structure probably
  carries over, but that is unmeasured.
- Two seeds per low-bit arm. remex's per-query bootstrap CI does not cover seed
  variance, which is larger than the CI for uncentered 1-bit and centered remax 1-bit.
- int8 with one global scale came out +0.006 with a CI excluding zero. Read that as
  ranking noise from a lossy codec, not a gain.

## Files

`encode.py` (resumable, chunk-checkpointed encode), `bench.py` (main arms),
`extra.py` (corrected centered remax, seed floor, centered nesting, paired
comparisons), `renest.py` (nested centered decode with the length re-solved),
`recheck.py` (prose against artifacts), `results_*.json`, per-query
`results_scifact_perquery.npz`, logs `*_scifact.log`. Encodings live in `.work/`
(gitignored, 800 MB).
