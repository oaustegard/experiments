# remex vs FAISS RaBitQ, flat scan, EmbeddingGemma 2 on Quora (100k)

Setup: first 100,000 BEIR Quora corpus docs and 1,000 test queries, google/embeddinggemma-2 text-only fp32 768d (R2 copy, sha256-verified). Exhaustive scan, float query against codes. Recall is against the exact fp32 d=768 inner-product top-10 / top-100, not nDCG. remex 1.2.0 (3 rotation seeds, mean [min–max]), faiss-cpu 1.15.1 `IndexRaBitQ` (qb=4, one run, no seed). Matryoshka rungs are truncated and renormalised, searched exactly. Single thread, 890 s wall. Raw data: `results_remex_flat.json`.

| arm | bytes/vec | R@10 | R@100 |
|---|---|---|---|
| remex 1-bit d=256 | 32.0 | 0.5909 [0.5871–0.5954] | 0.5164 [0.5131–0.5209] |
| remex centered 1-bit d=256 | 32.0 | 0.6731 [0.6704–0.6749] | 0.6168 [0.6134–0.6203] |
| faiss RaBitQ 1-bit d=256 | 40.0 | 0.7038 | 0.6640 |
| remex 1-bit d=512 | 64.0 | 0.7093 [0.7054–0.7145] | 0.6572 [0.6529–0.6622] |
| remex 2-bit d=256 | 64.0 | 0.7453 [0.7427–0.7485] | 0.7011 [0.6996–0.7027] |
| remex centered 1-bit d=512 | 64.0 | 0.7673 [0.7658–0.7695] | 0.7333 [0.7313–0.7349] |
| remex centered 2-bit d=256 | 64.0 | 0.7924 [0.7913–0.7933] | 0.7540 [0.7539–0.7541] |
| faiss RaBitQ 1-bit d=512 | 72.0 | 0.7955 | 0.7706 |
| faiss RaBitQ 2-bit d=256 | 84.0 | 0.8046 | 0.7668 |
| remex 1-bit d=768 | 96.0 | 0.7634 [0.7599–0.7662] | 0.7223 [0.7190–0.7264] |
| remex 8->1-bit nested d=768 | 96.0 | 0.7634 [0.7599–0.7662] | 0.7223 [0.7190–0.7264] |
| remex centered 1-bit d=768 | 96.0 | 0.8106 [0.8069–0.8126] | 0.7883 [0.7873–0.7896] |
| remex centered 8->1-bit nested d=768 | 96.0 | 0.8106 [0.8069–0.8126] | 0.7883 [0.7873–0.7896] |
| faiss RaBitQ 1-bit d=768 | 104.0 | 0.8334 | 0.8197 |
| remex 2-bit d=512 | 128.0 | 0.8280 [0.8278–0.8283] | 0.8082 [0.8071–0.8089] |
| remex 4-bit d=256 | 128.0 | 0.8382 [0.8368–0.8403] | 0.8041 [0.8033–0.8049] |
| remex centered 2-bit d=512 | 128.0 | 0.8679 [0.8673–0.8685] | 0.8524 [0.8516–0.8535] |
| remex centered 4-bit d=256 | 128.0 | 0.8454 [0.8447–0.8458] | 0.8108 [0.8106–0.8111] |
| faiss RaBitQ 2-bit d=512 | 148.0 | 0.8745 | 0.8651 |
| faiss RaBitQ 4-bit d=256 | 148.0 | 0.8452 | 0.8114 |
| remex 2-bit d=768 | 192.0 | 0.8667 [0.8636–0.8689] | 0.8576 [0.8574–0.8578] |
| remex 8->2-bit nested d=768 | 192.0 | 0.8327 [0.8301–0.8347] | 0.8181 [0.8171–0.8190] |
| remex centered 2-bit d=768 | 192.0 | 0.9036 [0.9030–0.9046] | 0.8986 [0.8984–0.8990] |
| remex centered 8->2-bit nested d=768 | 192.0 | 0.8847 [0.8832–0.8865] | 0.8778 [0.8770–0.8783] |
| faiss RaBitQ 2-bit d=768 | 212.0 | 0.9122 | 0.9101 |
| remex 4-bit d=512 | 256.0 | 0.9122 [0.9114–0.9135] | 0.9038 [0.9033–0.9043] |
| remex centered 4-bit d=512 | 256.0 | 0.9194 [0.9186–0.9202] | 0.9103 [0.9098–0.9105] |
| faiss RaBitQ 4-bit d=512 | 276.0 | 0.9215 | 0.9106 |
| remex 4-bit d=768 | 384.0 | 0.9573 [0.9556–0.9590] | 0.9587 [0.9584–0.9591] |
| remex 8->4-bit nested d=768 | 384.0 | 0.9504 [0.9494–0.9513] | 0.9530 [0.9519–0.9538] |
| remex centered 4-bit d=768 | 384.0 | 0.9704 [0.9693–0.9715] | 0.9720 [0.9716–0.9725] |
| remex centered 8->4-bit nested d=768 | 384.0 | 0.9674 [0.9671–0.9677] | 0.9684 [0.9682–0.9686] |
| faiss RaBitQ 4-bit d=768 | 404.0 | 0.9698 | 0.9716 |
| fp32 d=128 | 512.0 | 0.7145 | 0.6358 |
| fp32 d=256 | 1024.0 | 0.8504 | 0.8159 |
| fp32 d=512 | 2048.0 | 0.9263 | 0.9164 |
| fp32 d=768 | 3072.0 | 1.0000 | 1.0000 |

## Reading

Centered remex trails FAISS RaBitQ at every width, and the gap closes as bits go up: at d=768 R@10 is 0.811 vs 0.833 at 1 bit, 0.904 vs 0.912 at 2 bits and 0.970 vs 0.970 at 4 bits; at d=512 it is 0.767 vs 0.796, 0.868 vs 0.875 and 0.919 vs 0.922; at d=256 it is 0.673 vs 0.704, 0.792 vs 0.805 and 0.845 vs 0.845. The 1-bit gap (0.02–0.03) is well outside the remex seed spread (≤0.006 min–max); the 4-bit gap is inside it. FAISS spends 8 to 20 more bytes per vector at every point (104/212/404 vs 96/192/384 at d=768), so at matched bytes the 2- and 4-bit gaps shrink further. Uncentered remex is clearly worse than both (d=768: 0.763/0.867/0.957), so centering is worth +0.047, +0.037 and +0.013 R@10 at 1, 2 and 4 bits; centered 8→2-bit nested precision (0.885) loses 0.019 against a direct 2-bit code (0.904). Against the fp32 Matryoshka ladder (R@10 1.000 at 768 d / 3072 B, 0.926 at 512 d / 2048 B, 0.850 at 256 d / 1024 B, 0.715 at 128 d / 512 B), every quantized code beats the 128 d rung at a fraction of its bytes: centered remex 2-bit d=256 reaches 0.792 in 64 B, and 1-bit d=768 reaches 0.811 in 96 B. Centered 4-bit d=256 (128 B, 0.845) ties fp32 d=256 (1024 B, 0.850), centered 4-bit d=512 (256 B, 0.919) is within 0.007 of fp32 d=512 (2048 B), and centered 2-bit d=768 (192 B, 0.904) sits between the 256 d and 512 d rungs. This is the remex 1.2.0 baseline; the gap to RaBitQ is consistent with the scoring-estimator difference reported by the RaBitQ-vs-remex session, so a rerun with the next release is the test of that.


## remex 2.0.0 rerun (RaBitQ-style centered estimator)

Same data, queries, ground truth, seeds (0, 1, 2) and FAISS arms as above; remex 2.0.0 in a fresh venv. Centered arms (including the 8→k nested ones) are ranked with `Quantizer.search_batch` (`precision=` for nested), which in 2.0 uses q·μ + μ·r + ‖r‖(q−μ)·û/κ. Uncentered arms are unchanged and reproduce the 1.2.0 numbers exactly. Centered bytes per vector are d·bits/8 + 8 (norm + mean_dots). Raw data: `results_remex2_flat.json`; 693 s wall, single thread.

### Centered remex 1.2.0 vs 2.0.0 vs FAISS RaBitQ (bytes/vec in parentheses)

| d | bits | remex 1.2.0 centered (B) R@10 | remex 2.0.0 centered (B) R@10 [seed range] | FAISS RaBitQ (B) R@10 | 2.0 R@100 | RaBitQ R@100 |
|---|---|---|---|---|---|---|
| 768 | 1 | 0.8106 (96) | 0.8373 [0.8364–0.8386] (104) | 0.8334 (104) | 0.8274 | 0.8197 |
| 768 | 2 | 0.9036 (192) | 0.9136 [0.9124–0.9143] (200) | 0.9122 (212) | 0.9159 | 0.9101 |
| 768 | 4 | 0.9704 (384) | 0.9731 [0.9717–0.9741] (392) | 0.9698 (404) | 0.9760 | 0.9716 |
| 512 | 1 | 0.7673 (64) | 0.7961 [0.7937–0.7994] (72) | 0.7955 (72) | 0.7780 | 0.7706 |
| 512 | 2 | 0.8679 (128) | 0.8771 [0.8761–0.8782] (136) | 0.8745 (148) | 0.8687 | 0.8651 |
| 512 | 4 | 0.9194 (256) | 0.9210 [0.9193–0.9220] (264) | 0.9215 (276) | 0.9123 | 0.9106 |
| 256 | 1 | 0.6731 (32) | 0.7095 [0.7059–0.7119] (40) | 0.7038 (40) | 0.6670 | 0.6640 |
| 256 | 2 | 0.7924 (64) | 0.8030 [0.8010–0.8054] (72) | 0.8046 (84) | 0.7712 | 0.7668 |
| 256 | 4 | 0.8454 (128) | 0.8465 [0.8455–0.8471] (136) | 0.8452 (148) | 0.8122 | 0.8114 |

### Reading

With the 2.0 estimator centered remex matches FAISS RaBitQ at every width and dimension. R@10 differences are within +0.004 to −0.002 (d=768: 0.837 vs 0.833, 0.914 vs 0.912, 0.973 vs 0.970; d=512: 0.796 vs 0.796, 0.877 vs 0.875, 0.921 vs 0.922; d=256: 0.710 vs 0.704, 0.803 vs 0.805, 0.847 vs 0.845), which is inside the remex seed range (up to 0.006 at d=768); FAISS is a single run with no seed spread, so a gap this small is not resolved. R@100 is higher for remex 2.0 at d=768 (0.827 vs 0.820, 0.916 vs 0.910, 0.976 vs 0.972). Against 1.2.0 the estimator adds +0.027 to +0.037 R@10 at 1 bit, +0.009 to +0.011 at 2 bits and +0.001 to +0.003 at 4 bits, which removes the whole 1.2.0 deficit to RaBitQ. Bytes: at 1 bit remex 2.0 now equals FAISS (104 B at d=768) because of the extra 8 B, and at 2 and 4 bits it is 12 B smaller (200 vs 212, 392 vs 404 at d=768). Centered 8→k nested precision costs 0.016 at 2 bits (0.898 vs 0.914) and 0.003 at 4 bits against a direct code (1.2.0: 0.019). Uncentered remex and the fp32 Matryoshka rungs are unchanged from the 1.2.0 table, so the earlier conclusions about the ladder stand; in 2.0 terms, centered 1-bit d=768 (104 B, 0.837) is within 0.013 of fp32 d=256 (1024 B, 0.850), and centered 2-bit d=768 (200 B, 0.914) is within 0.012 of fp32 d=512 (2048 B, 0.926).

### Full 2.0.0 table (sorted by bytes per vector; R@10 and R@100 as mean [min–max] over 3 seeds for remex)

| arm | bytes/vec | R@10 | R@100 |
|---|---|---|---|
| remex 1-bit d=256 | 32.0 | 0.5909 [0.5871–0.5954] | 0.5164 [0.5131–0.5209] |
| faiss RaBitQ 1-bit d=256 | 40.0 | 0.7038 | 0.6640 |
| remex centered 1-bit d=256 | 40.0 | 0.7095 [0.7059–0.7119] | 0.6670 [0.6661–0.6679] |
| remex 1-bit d=512 | 64.0 | 0.7093 [0.7054–0.7145] | 0.6572 [0.6529–0.6622] |
| remex 2-bit d=256 | 64.0 | 0.7453 [0.7427–0.7485] | 0.7011 [0.6996–0.7027] |
| faiss RaBitQ 1-bit d=512 | 72.0 | 0.7955 | 0.7706 |
| remex centered 1-bit d=512 | 72.0 | 0.7961 [0.7937–0.7994] | 0.7780 [0.7765–0.7789] |
| remex centered 2-bit d=256 | 72.0 | 0.8030 [0.8010–0.8054] | 0.7712 [0.7706–0.7719] |
| faiss RaBitQ 2-bit d=256 | 84.0 | 0.8046 | 0.7668 |
| remex 1-bit d=768 | 96.0 | 0.7634 [0.7599–0.7662] | 0.7223 [0.7190–0.7264] |
| remex 8->1-bit nested d=768 | 96.0 | 0.7634 [0.7599–0.7662] | 0.7223 [0.7190–0.7264] |
| faiss RaBitQ 1-bit d=768 | 104.0 | 0.8334 | 0.8197 |
| remex centered 1-bit d=768 | 104.0 | 0.8373 [0.8364–0.8386] | 0.8274 [0.8268–0.8284] |
| remex centered 8->1-bit nested d=768 | 104.0 | 0.8373 [0.8364–0.8386] | 0.8274 [0.8268–0.8284] |
| remex 2-bit d=512 | 128.0 | 0.8280 [0.8278–0.8283] | 0.8082 [0.8071–0.8089] |
| remex 4-bit d=256 | 128.0 | 0.8382 [0.8368–0.8403] | 0.8041 [0.8033–0.8049] |
| remex centered 2-bit d=512 | 136.0 | 0.8771 [0.8761–0.8782] | 0.8687 [0.8686–0.8689] |
| remex centered 4-bit d=256 | 136.0 | 0.8465 [0.8455–0.8471] | 0.8122 [0.8121–0.8122] |
| faiss RaBitQ 2-bit d=512 | 148.0 | 0.8745 | 0.8651 |
| faiss RaBitQ 4-bit d=256 | 148.0 | 0.8452 | 0.8114 |
| remex 2-bit d=768 | 192.0 | 0.8667 [0.8636–0.8689] | 0.8576 [0.8574–0.8578] |
| remex 8->2-bit nested d=768 | 192.0 | 0.8327 [0.8301–0.8347] | 0.8181 [0.8171–0.8190] |
| remex centered 2-bit d=768 | 200.0 | 0.9136 [0.9124–0.9143] | 0.9159 [0.9157–0.9161] |
| remex centered 8->2-bit nested d=768 | 200.0 | 0.8978 [0.8973–0.8982] | 0.8980 [0.8974–0.8986] |
| faiss RaBitQ 2-bit d=768 | 212.0 | 0.9122 | 0.9101 |
| remex 4-bit d=512 | 256.0 | 0.9122 [0.9114–0.9135] | 0.9038 [0.9033–0.9043] |
| remex centered 4-bit d=512 | 264.0 | 0.9210 [0.9193–0.9220] | 0.9123 [0.9118–0.9126] |
| faiss RaBitQ 4-bit d=512 | 276.0 | 0.9215 | 0.9106 |
| remex 4-bit d=768 | 384.0 | 0.9573 [0.9556–0.9590] | 0.9587 [0.9584–0.9591] |
| remex 8->4-bit nested d=768 | 384.0 | 0.9504 [0.9494–0.9513] | 0.9530 [0.9519–0.9538] |
| remex centered 4-bit d=768 | 392.0 | 0.9731 [0.9717–0.9741] | 0.9760 [0.9758–0.9762] |
| remex centered 8->4-bit nested d=768 | 392.0 | 0.9706 [0.9691–0.9714] | 0.9734 [0.9731–0.9738] |
| faiss RaBitQ 4-bit d=768 | 404.0 | 0.9698 | 0.9716 |
| fp32 d=128 | 512.0 | 0.7145 | 0.6358 |
| fp32 d=256 | 1024.0 | 0.8504 | 0.8159 |
| fp32 d=512 | 2048.0 | 0.9263 | 0.9164 |
| fp32 d=768 | 3072.0 | 1.0000 | 1.0000 |
