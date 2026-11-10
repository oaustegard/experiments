## Answer

Yes. Ettin-17m embeddings can be stored at 256 dimensions without losing float search quality. The model's final output matrix maps 256 dimensions to 768, and it can be folded into a 256-to-256 matrix. On all 9,500 queries, hit@10 is 0.918 for both the 768-dim and the folded 256-dim embeddings, and their scores agree within 1.1e-06.

Oskar asked the question on 2026-11-09: "does that mean we could trim these to 256 length vectors without penalty by reversing the widening?"

Quantized codes on the 256 folded dimensions beat the live index while using fewer bytes. The live index stores remex codes at 2 bits × 768 dims in 200 bytes and reaches recall@10 0.857 against float search.

| remex codes on 256 dims | Bytes | Recall@10 |
|---|---|---|
| 3 bits | 104 | 0.868 |
| 4 bits | 136 | 0.927 |
| 8 bits | 264 | 0.993 |

At the same bit width the 256-dim codes do worse: 2 bits × 256 gives 0.763. The gain comes from spending the saved dimensions on more bits per dimension.

remex's rotation folds into the same matrix too, so the model can output rotated coordinates directly. With an int8 head this reaches 0.931 at 4 bits.

The List Scout page now runs the folded model. Its classifier answers are identical, and fitting takes 1.2 s instead of 3.6 s. The search indexes on the site are still 768-dimensional.
