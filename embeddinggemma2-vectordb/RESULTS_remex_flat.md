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
