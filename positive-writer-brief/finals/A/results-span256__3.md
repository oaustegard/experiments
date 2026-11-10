## Answer

Oskar asked on 2026-11-09: "does that mean we could trim these to 256 length vectors without penalty by reversing the widening?"

Yes, for float search. Ettin-17m's final output matrix maps 256 dimensions to 768, and it folds into a 256-to-256 matrix. On all 9,500 queries, hit@10 is 0.918 for both the 768-dimensional and the folded 256-dimensional vectors, and their scores agree within 1.1e-06.

Quantized vectors need more bits per dimension at 256 dimensions. At 3 and 4 bits they still take fewer bytes than the live index and score higher. The live index stores 2 bits × 768 dimensions in 200 bytes and reaches recall@10 0.857 against float search. These are the remex codes on the 256 folded dimensions:

| bits | bytes | recall@10 |
|---|---|---|
| 3 | 104 | 0.868 |
| 4 | 136 | 0.927 |
| 8 | 264 | 0.993 |

At the same bit width the 256-dimensional codes lose: 2 bits × 256 dimensions gives 0.763.

remex's rotation folds into the same matrix too, so the model can emit rotated coordinates directly. With an int8 head, the rotated codes reach 0.931 at 4 bits.

The List Scout page runs the folded model. Its classifier answers are identical to the 768-dimensional model's, and fitting takes 1.2 s instead of 3.6 s. The search indexes on the site are still 768-dimensional.
