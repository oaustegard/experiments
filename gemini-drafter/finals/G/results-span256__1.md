## Answer

Ettin-17m embeddings can be stored as 256-dimensional vectors with no loss in float search. At 3 or 4 bits per dimension, remex codes on those vectors are smaller than the live index's codes and score higher recall.

Oskar asked on 2026-11-09: "does that mean we could trim these to 256 length vectors without penalty by reversing the widening?" Ettin-17m ends with a matrix that widens 256 dimensions to 768. Folding that matrix into a 256-to-256 matrix leaves float search unchanged. On all 9,500 queries, the 768-dimensional and 256-dimensional vectors both reach hit@10 of 0.918, and their scores agree within 1.1e-06.

The live index stores 2-bit remex codes over 768 dimensions. Each vector takes 200 bytes, and recall@10 against float search is 0.857. At 256 dimensions with the same 2-bit width, recall@10 drops to 0.763. Wider codes beat the live index, and the 3-bit and 4-bit codes also use fewer bytes:

| Bits | Bytes | recall@10 |
|---|---|---|
| 3 | 104 | 0.868 |
| 4 | 136 | 0.927 |
| 8 | 264 | 0.993 |

Remex's rotation folds into the same 256-to-256 matrix, so the model can emit rotated coordinates directly. With an int8 head, 4-bit codes reach 0.931.

The List Scout page now runs the folded model. Its classifier answers are identical, and fitting takes 1.2 s instead of 3.6 s. The search indexes on the site are still 768-dimensional.
