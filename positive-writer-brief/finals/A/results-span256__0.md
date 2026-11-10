## Answer

Yes for float search. At the same bit width, compressed codes lose recall. Oskar asked on 2026-11-09: "does that mean we could trim these to 256 length vectors without penalty by reversing the widening?" Ettin-17m's final output matrix maps 256 dimensions to 768, and it can be folded into a 256-to-256 matrix. Float search on the folded 256-dimensional vectors matches the original. Hit@10 is 0.918 both ways on all 9,500 queries, and scores agree within 1.1e-06.

The live index stores remex codes at 2 bits × 768 dimensions (200 bytes). Its recall@10 against float search is 0.857. Here is how remex codes on the 256 folded dimensions compare:

| bits × dims | bytes | recall@10 |
|---|---|---|
| 2 × 768 (live index) | 200 | 0.857 |
| 2 × 256 | — | 0.763 |
| 3 × 256 | 104 | 0.868 |
| 4 × 256 | 136 | 0.927 |
| 8 × 256 | 264 | 0.993 |

The 3-bit and 4-bit codes beat the live index and use fewer bytes. The 8-bit codes reach 0.993 but take 264 bytes. At 2 bits × 256 the folded codes get 0.763, below the live index.

Remex's rotation also folds into the same matrix, so the model can emit rotated coordinates directly. With an int8 head, this gives 0.931 at 4 bits.

The List Scout page now runs the folded model. Its classifier answers are identical, and fitting takes 1.2 s instead of 3.6 s. The search indexes on the site are still 768-dimensional.
