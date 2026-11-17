## Answer

Ettin-17m embeddings can be stored at 256 dimensions with no loss in float search. Quantized remex codes on those 256 dimensions beat the live 768-dimensional index at 3 and 4 bits per dimension, in fewer bytes. Oskar asked on 2026-11-09: "does that mean we could trim these to 256 length vectors without penalty by reversing the widening?"

Ettin-17m's final 256-to-768 output matrix folds into a 256-to-256 matrix. On all 9,500 evaluation queries, both versions reach hit@10 of 0.918, and their scores agree within 1.1e-06.

The live index uses 2 bits × 768 dimensions, takes 200 bytes and reaches 0.857 recall@10 against float search. At the same 2 bits, the 256-dimensional codes drop to 0.763. With more bits per dimension they do better:

| bits | bytes | recall@10 |
|---|---|---|
| 3 | 104 | 0.868 |
| 4 | 136 | 0.927 |
| 8 | 264 | 0.993 |

Remex's rotation also folds into the same 256-to-256 matrix, so the model can emit rotated coordinates directly. With an int8 head, this reaches 0.931 recall@10 at 4 bits.

The List Scout page runs the folded model. Its classifier answers are identical, and fitting takes 1.2 s instead of 3.6 s. The search indexes on the site are still 768-dimensional.
