## Answer

Oskar asked on 2026-11-09: "does that mean we could trim these to 256 length vectors without penalty by reversing the widening?" For float search, yes. Ettin-17m's final output matrix maps 256 dimensions to 768, and it folds into a 256-to-256 matrix. On all 9,500 queries, hit@10 is 0.918 for both the 768-dim vectors and the folded 256-dim vectors, and the two versions' scores agree within 1.1e-06.

Quantized vectors behave differently. At the same bit width the 256-dim codes do worse: 2 bits × 256 dims reaches recall@10 of 0.763, where the live index (2 bits × 768 dims, 200 bytes) reaches 0.857. With more bits per dimension, remex codes on the 256 folded dimensions beat the live index on recall and still use fewer bytes:

| Bits × 256 dims | Bytes | recall@10 vs float |
|---|---|---|
| 3 | 104 | 0.868 |
| 4 | 136 | 0.927 |
| 8 | 264 | 0.993 |

The remex rotation also folds into the same matrix, so the model can output rotated coordinates directly. With an int8 head, that setup reaches 0.931 at 4 bits.

The List Scout page now runs the folded model. Its classifier gives identical answers, and fitting takes 1.2 s instead of 3.6 s. The site's search indexes are still 768-dimensional.
