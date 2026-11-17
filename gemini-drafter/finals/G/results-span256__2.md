## Answer

Yes. Ettin-17m embeddings can be stored at 256 dimensions without any loss in float search. At 3 or 4 bits per dimension, the quantized 256-dimensional codes are smaller than the live 768-dimensional index and have higher recall. Oskar asked the question on 2026-11-09: "does that mean we could trim these to 256 length vectors without penalty by reversing the widening?"

Ettin-17m's final output matrix maps 256 to 768 dimensions, and it can be folded into a 256-to-256 matrix. Across all 9,500 queries, float search scores 0.918 hit@10 with either matrix, and individual scores agree within 1.1e-06.

The live index stores 2 bits × 768 dimensions in 200 bytes and reaches 0.857 recall@10 against float search. Here are the results for remex codes on the 256 folded dimensions:

| Bits | Bytes | Recall@10 |
|---|---|---|
| 2 | — | 0.763 |
| 3 | 104 | 0.868 |
| 4 | 136 | 0.927 |
| 8 | 264 | 0.993 |

At the same 2-bit width the 256-dimensional codes are worse than the live index. Remex's rotation also folds into the same matrix, so the model can emit rotated coordinates directly. With an int8 head, this reaches 0.931 recall@10 at 4 bits.

The List Scout page now runs the folded model. Its classifier answers are identical, and fitting takes 1.2 s instead of 3.6 s. The search indexes on the site are still 768-dimensional.
