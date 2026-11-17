## Answer

Ettin-17m embeddings can be stored as 256-dimensional vectors without loss in float search, and quantized to smaller, more accurate remex codes by using 3 or 4 bits per dimension.

On 2026-11-09, Oskar asked, "does that mean we could trim these to 256 length vectors without penalty by reversing the widening?" Folding Ettin-17m's final 256-to-768 output matrix into a 256-to-256 matrix leaves float search unchanged. Across all 9,500 queries, both the 768-dimensional and 256-dimensional vectors achieve a hit@10 of 0.918, and their scores agree within 1.1e-06.

For quantized search, the live index uses 2-bit remex codes across 768 dimensions, taking 200 bytes per vector for a recall@10 of 0.857 against float search. At 256 dimensions, matching that 2-bit width drops recall@10 to 0.763, while raising the bit width beats the live index in fewer bytes: 3-bit codes take 104 bytes and reach 0.868, 4-bit codes take 136 bytes and reach 0.927, and 8-bit codes take 264 bytes and reach 0.993. Remex's rotation also folds into that 256-to-256 matrix so the model emits rotated coordinates directly; with an int8 head, 4-bit codes reach 0.931.

The List Scout page now runs the folded 256-dimensional model, producing identical classifier answers and cutting fitting time from 3.6 s to 1.2 s. The search indexes on the site remain 768-dimensional.
