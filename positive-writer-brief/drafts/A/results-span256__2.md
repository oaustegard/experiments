## Answer

Yes. Oskar asked on 2026-11-09: "does that mean we could trim these to 256 length vectors without penalty by reversing the widening?" Ettin-17m's last layer projects 256 dimensions up to 768. That output matrix folds into a 256-to-256 matrix, and float search on the folded vectors matches the 768-dimensional original. Hit@10 is 0.918 both ways on all 9,500 queries, and scores agree within 1.1e-06.

Quantized storage also gets cheaper, but only at a higher bit width. The live index stores remex codes at 2 bits × 768 dimensions: 200 bytes per vector, with recall@10 of 0.857 against float search. Remex codes on the 256 folded dimensions compare as follows:

| bits × dims | bytes | recall@10 |
|---|---|---|
| 2 × 768 (live index) | 200 | 0.857 |
| 3 × 256 | 104 | 0.868 |
| 4 × 256 | 136 | 0.927 |
| 8 × 256 | 264 | 0.993 |

At the same width the shorter codes lose: 2 bits × 256 gives 0.763.

Remex's rotation folds into the same matrix, so the model can emit rotated coordinates directly. With an int8 head, 4-bit codes reach 0.931.

The List Scout page now runs the folded model. Its classifier gives identical answers, and fitting takes 1.2 s instead of 3.6 s. The search indexes on the site are still 768-dimensional.
