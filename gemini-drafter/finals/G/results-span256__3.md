## Answer

Yes. Ettin-17m embeddings can be stored at 256 dimensions without losing float search accuracy. Remex codes on those 256 dimensions also beat the live 768-dimensional index in fewer bytes at 3 or 4 bits per dimension.

Oskar asked on 2026-11-09: "does that mean we could trim these to 256 length vectors without penalty by reversing the widening?" Ettin-17m's final output matrix maps 256 dimensions to 768, and it can be folded into a 256-to-256 matrix. On all 9,500 queries, float search gets 0.918 hit@10 at both 768 and 256 dimensions, and the scores agree within 1.1e-06.

The quantized results depend on bit width. The live 768-dimensional remex index uses 2 bits per dimension (200 bytes) and gets 0.857 recall@10 against float search. At the same 2 bits, 256 dimensions get only 0.763. With more bits per dimension, the 256-dimensional codes do better:

| Bits | Bytes | recall@10 |
|---|---|---|
| 2 (live, 768 dims) | 200 | 0.857 |
| 2 | — | 0.763 |
| 3 | 104 | 0.868 |
| 4 | 136 | 0.927 |
| 8 | 264 | 0.993 |

Remex's rotation can be folded into the same 256-to-256 matrix, so the model outputs rotated coordinates directly. With an int8 head, 4-bit codes get 0.931 recall@10.

The List Scout page runs the folded model. Its classifier answers are identical, and fitting takes 1.2 s instead of 3.6 s. The search indexes on the site are still 768-dimensional.
