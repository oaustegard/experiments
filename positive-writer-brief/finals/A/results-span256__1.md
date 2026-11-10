## Answer

Yes. On 2026-11-09 Oskar asked: "does that mean we could trim these to 256 length vectors without penalty by reversing the widening?" Ettin-17m's final output matrix maps 256 dimensions to 768, and it can be folded into a 256-to-256 matrix. Float search returns the same results either way: hit@10 is 0.918 for both on all 9,500 queries, and scores agree within 1.1e-06.

With quantized codes, the 256 folded dimensions beat the live index. The live index stores remex codes at 2 bits × 768 dimensions in 200 bytes and reaches recall@10 0.857 against float search. Remex codes on the 256 folded dimensions score as follows:

| bits | bytes | recall@10 |
|---|---|---|
| 3 | 104 | 0.868 |
| 4 | 136 | 0.927 |
| 8 | 264 | 0.993 |

The 3-bit and 4-bit codes beat the live index in fewer bytes. The 8-bit codes take more bytes than the live index does.

At the same bit width the 256-dimensional codes lose to the live index: 2 bits × 256 gives 0.763, against 0.857 for 2 bits × 768.

Remex's rotation also folds into the same matrix, so the model can emit rotated coordinates directly. With an int8 head, this reaches 0.931 at 4 bits.

The List Scout page now runs the folded model. Its classifier answers are identical, and fitting takes 1.2 s instead of 3.6 s. The search indexes on the site are still 768-dimensional.
