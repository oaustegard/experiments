# RESULTS: spending the token-table bytes on more tokens instead of fewer bits

Thread "Vocabulary embedding quantization" in the Embedded Embedder project, 2026-10-23. All on a
4-vCPU CCotw container, about 13 hours wall clock, most of it EG2 teacher passes and two
1-epoch fine-tunes. No API spend.

## Answer

Can the k = 4 EG2 query cut's token table beat vocabulary trimming at the same byte budget? Yes,
when quantizing better is used to buy a larger vocabulary, not to keep all 262,144 rows.
A 67,347-token vocabulary, with rows at 3 bits for the 19,451 most frequent tokens and 2 bits
for the rest, makes a weight file the same size as today's 41,741-token 3-bit build
(24.41 MB <!-- @evidence[[results/packed_v64k_rhq.json#weights_mb]] -->
against 24.43 MB). <!-- @evidence[[results/packed_trim_t3w.json#weights_mb]] -->
After the same caption fine-tune, held-out text-to-image hit@10 is 0.732 <!-- @evidence[[results/packed_v64k_rhq.json#img_test_hit10]] -->
against 0.718. <!-- @evidence[[results/packed_trim_t3w.json#img_test_hit10]] -->
Top-10 overlap with EG2 rises by 0.022, <!-- @evidence[[results/r6_pairs.json#packed_v64k_rhq_vs_trim_t3w.img_test_overlap.0]] -->
and Bluesky source-post hit@10 changes by −0.006 [−0.011, −0.001]. <!-- @evidence[[results/r6_pairs.json#packed_v64k_rhq_vs_trim_t3w.synth_hit10.0]] -->
The tokenizer.json adds 0.49 MB gzipped.

The same codec on today's vocabulary saves 1.9 MB at unchanged quality.
Quantizing the whole 262K vocabulary loses, because a rare row coded in 8 to 16 bytes is
worse than splitting the word into well-stored pieces.

What would change this is a compact tokenizer format, which would turn the +0.49 MB into
about −0.4 MB, or a longer caption fine-tune.

## Findings

1. **The cut's trained table is not EG2's table any more.** Ten epochs of distillation moved the
   kept rows to cosine 0.48 from EG2's own rows, <!-- @evidence[[results/r1_fullvocab.json#trained_vs_eg2_row_cos.mean]] -->
   at about four times their norm. EG2's rows for the other 220K tokens do not fit the cut's
   layers. With EG2's rows everywhere, Bluesky hit@10 collapses to 0.553. <!-- @evidence[[results/r1_fullvocab.json#full_eg2rows.synth_hit10]] -->
   Adding the untrained rows to the trained ones barely moves anything. A
   larger vocabulary needs new rows learned for the cut. (Round 1)
2. **Rows for unseen tokens can be learned from one-token EG2 queries.** Rows are initialized by a
   ridge fit from [EG2 row, EG2's vector for that token alone], then distilled with the layers
   frozen. With all 262K rows in float, text-to-image hit@10 rises from 0.563 <!-- @evidence[[results/tok_rows_ridge_e3.json#bench.cut_true.img_hit10]] -->
   to 0.611. <!-- @evidence[[results/tok_rows_ridge_e3.json#bench.full_learned.img_hit10]] -->
   As a control, 2,000 trained rows were relearned the same way. Swapping them in costs
   0.004 real-query overlap (0.632 to 0.628). <!-- @evidence[[results/tok_rows_ridge_e3.json#bench.cut_hold_learned.real_overlap]] -->
   (Round 3)
3. **A rotation plus Lloyd-Max levels plus one scale per row (remex / rhq) does at 3 bits what
   int3 in groups of 32 does, in 3/4 of the bytes.** On the cut's table, real overlap is 0.627 at 8.1 MB <!-- @evidence[[results/r4_remex_cut.json#remex3.real_overlap]] -->
   against 0.628 at 10.7 MB. <!-- @evidence[[results/r2_cut_codecs.json#int3_g32.real_overlap]] -->
   The fp16 scale and zero point per 32 values cost int-g32 a full bit per value. The rows are
   close to isotropic (PCA coding gain 0.6 dB), so no rotation or codebook gets them much below
   about 3 bits without loss. (Rounds 2, 4)
4. **All 262K rows at the trimmed budget lose.** With the kept rows at 3 bits, the other 220K
   get 8 to 16 bytes each in product-quantization codebooks, which are then fine-tuned through
   the frozen cut. Real overlap drops to 0.613, <!-- @evidence[[results/cbft_kr3_s16_u8.json#after.real_overlap]] -->
   below the trimmed 0.627. A query whose rare word now maps to one coarse row does worse than
   the same query split into well-stored pieces. Per byte, PQ beats int-g32 at 1 to 2 bits, but
   not by enough. (Rounds 3, 5)
5. **Vocabulary size against bits has a sweet spot at a 10 MB table.** Before the caption
   fine-tune, 67K rows at mixed 3/2 bits lift text-to-image hit@10 to 0.606 <!-- @evidence[[results/r5_vocab_size.json#v64k_mix.img_hit10]] -->
   from 0.561, <!-- @evidence[[results/r5_vocab_size.json#v42k_r3.img_hit10]] -->
   with real overlap 0.623. <!-- @evidence[[results/r5_vocab_size.json#v64k_mix.real_overlap]] -->
   All 2-bit at 67K or 82K rows, and all 3-bit at 54K rows, do worse on one benchmark or the
   other. (Round 5)
6. **The caption fine-tune absorbs about two thirds of the vocabulary gain. The rest holds on the
   shipped files.** Both vocabularies get the same 4,214 steps of run C's recipe. Paired on the
   1,035 held-out images, hit@10 changes by +0.0145 [+0.0019, +0.0271] and overlap by
   +0.022 [+0.016, +0.029]. <!-- @evidence[[results/r6_pairs.json#packed_v64k_rhq_vs_trim_t3w.img_test_overlap.0]] -->
   Bluesky real overlap is flat, and Bluesky hit@10 changes by −0.006 [−0.011, −0.001]. The
   trimmed baseline reproduces the shipped run C (0.718 against 0.712). (Round 6)
7. **The WGSL engine decodes rhq rows in its embed kernel.** It runs a 512-point Walsh-Hadamard
   transform in workgroup memory, with fixed signs, Lloyd-Max levels and an fp16 scale. The 3-bit
   rows come first in the renumbered vocabulary. Headless SwiftShader parity against the NumPy
   reference is cosine ≥ 0.9999999, single and batched, on both new builds. (Round 7)

## Method

Artifacts: R2 `models/eg2-vocab-codebook/` holds the two fine-tuned checkpoints (`vc_trim_final.pt`, `vc_v64k_final.pt`) and `eg2-vocab-codebook-builds.tgz`, the engine builds `v64k_rhq/` and `trim_rhq3/` (weights, manifest, renumbered tokenizer).

**Fixture.** The same benchmarks as eg2-layer-student. Bluesky ASYM: real_test overlap@10 (1,000
MS MARCO tech queries) and synth_test source-post hit@10 (1,498), both against the 61,686-post
EG2 index. Text to image: alt-text queries over the 16,878-image EG2 index on the site. "test"
is the 1,035 queries whose images training never sees. Rounds 1–5 start from the 10-epoch
`mac_k4_v32k` checkpoint (release `eg2-layer-mac-20261020`). Round 6 fine-tunes from it.

**Learned rows.** `run/token_teacher.py` computes EG2 q4's vector for every token as a one-token
query: the query prompt's ids, with token i in place of the trailing `▁`. The three multimodal
placeholder ids get no target. `run/tok_rows.py` does the ridge initialization and the
one-token distillation (sparse Adam, rows only).

**Codecs.** `run/tcodecs.py` has int-g32 (the shipped layout), product quantization (faiss,
frequency-weighted training) and residual PQ. remex 2.1 has haar rotation, Lloyd-Max levels and
an fp16 norm. `run/rhq.py` is the shipping codec: z = H(s ⊙ x), with H the orthonormal
Walsh-Hadamard transform and s xorshift signs, Lloyd-Max levels for N(0, 1), and a least-squares
fp16 scale per row. Its reconstruction matches remex haar at 3 and 2 bits (cos 0.9827 / 0.9397).

**Vocabularies.** The kept 41,741 ids plus the most frequent other ids in training-side text
(posts, training queries, train-split alt and post text, harvested alt text), closed under BPE
by eg2-query-student's `a4_prune`. 3-bit rows go to the most frequent tokens. Added tokens are
always 3-bit.

**Fine-tune (Round 6).** `eg2-layer-student/run/l1_distill.py`: lr 1e-3, 1 epoch, batch 64,
seed 5, msmarco_train + synth_train + gpu_more cut to 30,000, train alt ×4, train posts, and
210,132 harvested alt texts, re-harvested because run C's set was not kept. 269,594 texts.
Everything is trainable, and the table is float during training. Shipping files:
`export.py` q4 layers, then `t3w` for the trimmed arm or `run/pack_engine.py` rhq for the new
one. They are scored through the engine's own NumPy forward pass (`run/eval_packed.py`), which
reproduces the trimmed `t3w` file's onnxruntime numbers exactly.

**Bytes.** Weights are the engine's data file. The tokenizer is tokenizer.json gzip -9: 0.70 MB
at 41,741 rows, 1.19 MB at 67,347. <!-- @evidence[[results/r5_vocab_size.json#v64k_mix.tokenizer_json_gz_mb]] -->
A vocabulary-list format would be 0.28 MB. Gemma's merges are re-derivable from the vocabulary:
all 513,510 merges after the first 1,396 are exactly every (a, b) with a + b in the vocabulary,
sorted by (id(a + b), −len(a)). That format is not built.

## Log

### Round 1: full vocabulary, no retraining (`run/r1_fullvocab.py`)

| arm | real ov | synth hit@10 | img ov | img hit@10 |
|---|---|---|---|---|
| cut (41,741 rows, float) | 0.632 | 0.907 | 0.384 | 0.563 |
| full tokenizer, EG2 rows outside the cut | 0.620 | 0.907 | 0.387 | 0.575 |
| full tokenizer, EG2 rows everywhere | 0.197 | 0.553 | 0.105 | 0.193 |

Trained rows vs EG2's: cos 0.48 on average. A linear map from EG2 rows predicts them at only
cos 0.51 (`run/map_probe.py`). The drift is token-specific content that training wrote into the
rows.

### Round 2: codecs on the cut's own table (`run/r2_cut_codecs.py`, `run/kept_codecs.py`)

| codec | MB | row cos | real ov | img hit@10 |
|---|---|---|---|---|
| int4 g32 | 13.36 | 0.997 | 0.631 | 0.566 |
| int3 g32 (shipped) | 10.69 | 0.987 | 0.628 | 0.561 |
| int2 g32 | 8.01 | 0.934 | 0.611 | 0.547 |
| int1 g32 | 5.34 | 0.790 | 0.480 | 0.449 |
| PQ 64 bytes/row | 2.93 | 0.798 | 0.566 | 0.500 |
| PQ 128 bytes/row | 5.61 | 0.935 | 0.617 | 0.553 |

The run was stopped at its time cap before the PQ 256 arm. Reconstruction only, bytes per row:
int3 g32 0.987 at 256, remex3 0.983 at 194, PQ 128×10-bit 0.971 at 160 plus a 1 MB codebook,
PQ 256 0.994 at 256, remex2 0.940 at 130.

### Round 3: learned rows, then tiered PQ for all 262K (`run/tok_rows.py`, `run/r3_full_codecs.py`, `run/cbft.py`)

Learned rows (float): img hit@10 0.611, overlap 0.440, real 0.635. Tiered PQ without fine-tuning
at 9.4–10.8 MB: real 0.600–0.611, img overlap 0.39–0.41. Codebook fine-tuning (1,500–2,000
steps, codes fixed, text targets plus the one-token loss):

| tiers | MB | real ov | synth hit@10 | img ov | img hit@10 |
|---|---|---|---|---|---|
| kept PQ128 / seen PQ32 / unseen PQ8 | 9.71 | 0.612 | 0.903 | 0.423 | 0.594 |
| top-8K PQ256 / kept PQ128 / seen PQ16 / unseen PQ8 | 9.81 | 0.615 | 0.899 | 0.423 | 0.598 |
| kept remex3 / seen PQ16 / unseen PQ8 | 10.99 | 0.613 | 0.902 | 0.400 | 0.576 |

The last row is the clearest. The kept rows alone at remex3 score 0.627. Adding the coarse rare
rows takes real overlap down to 0.613.

### Round 4: remex on the cut (`run/r4_remex_cut.py`)

remex3 8.10 MB: real 0.6265, synth hit@10 0.9065, img hit@10 0.561. remex2 5.43 MB: 0.614 /
0.905 / 0.552. These equal int3 g32 at 10.69 MB and int2 g32 at 8.01 MB.

### Round 5: vocabulary size against bits (`run/r5_vocab_size.py`)

| vocab | rows | bits | table MB | tok gz MB | real ov | synth hit@10 | img ov | img hit@10 |
|---|---|---|---|---|---|---|---|---|
| v42k | 41,741 | 3 | 8.10 | 0.70 | 0.627 | 0.907 | 0.379 | 0.561 |
| v52k | 53,960 | 3 | 10.47 | 0.93 | 0.624 | 0.909 | 0.421 | 0.595 |
| v64k | 67,347 | 2 | 8.76 | 1.19 | 0.610 | 0.905 | 0.420 | 0.602 |
| v64k mix | 67,347 | 19,451 at 3, rest 2 | 10.00 | 1.19 | 0.623 | 0.907 | 0.429 | 0.606 |
| v78k | 82,402 | 2 | 10.71 | 1.49 | 0.612 | 0.905 | 0.423 | 0.603 |
| v100k | 104,851 | 2 | 13.63 | 1.92 | 0.613 | 0.905 | 0.425 | 0.605 |

The 78K and 100K mixed arms had no budget left for 3-bit rows, so they ran at all 2 bits.

### Round 6: the caption fine-tune with each vocabulary (`run/r6_finetuned.py`, `run/eval_packed.py`, `run/r6_pairs.py`)

Shipping files, q4 layers:

| build | weights MB | real ov | synth hit@10 | img test ov | img test hit@10 |
|---|---|---|---|---|---|
| trimmed, t3w (today's format) | 24.43 | 0.611 | 0.909 | 0.604 | 0.718 |
| trimmed, rhq3 | 22.51 | 0.616 | 0.905 | 0.603 | 0.720 |
| 67K, rhq 3/2 | 24.41 | 0.613 | 0.903 | 0.626 | 0.732 |

Float layers, same checkpoints: the trimmed float table gets img test hit@10 0.714, the 67K
float table 0.735. Before fine-tuning the vocabulary gain was +0.045 hit@10. After, it is
+0.015.

### Round 7: engine (`eg2-wgsl-engine/web/engine.js` `embed_rhq`, `run/pack_engine.py`, `run/engine_check.py`)

The packer copies every non-table tensor of the `t3w` build byte for byte, appends the rhq table
(3-bit codes, 2-bit codes, fp16 scales, 64 bytes of signs, levels) and renumbers the tokenizer.
Parity PASS on `v64k_rhq` and `trim_rhq3`. There is no onnxruntime-web fallback graph for an
rhq table yet. It would need the Hadamard as a 512×512 constant (0.5 MB in fp16) or an unrolled
butterfly.

### What broke

- An EG2 pass over all tokens crashed at id 258,880. The image, audio and video placeholder ids
  route into the multimodal Gather.
- The first one-token prompt kept the trailing `▁` of "query: " as its own token. That is not
  how a real query tokenizes, so it was discarded after 16K tokens.
- Three CPU jobs on 4 vCPUs (each set to 4 threads) slowed each other several-fold. A caption
  teacher pass died silently in that window, probably from memory pressure.
- `vocab.py` rebuilt from re-hydrated posts gave 41,743 rows, not 41,741. The kept set was taken
  from the shipped tokenizer instead.
- Run C's float checkpoint and its harvested texts were not kept anywhere, so Round 6 rebuilds
  both arms from mac_k4 on a fresh harvest.
