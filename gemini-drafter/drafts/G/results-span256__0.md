## Answer

Ettin-17m embeddings can be stored at 256 dimensions with identical float search accuracy, and quantized remex indexes match or beat the 768-dimensional index in fewer bytes when given at least 3 bits per dimension. On 2026-11-09, Oskar asked, "does that mean we could trim these to 256 length vectors without penalty by reversing the widening?" Folding Ettin-17m's final 256-to-768 output matrix into a 256-to-256 matrix preserves float search across all 9,500 evaluation queries. Both versions achieve a hit@10 of 0.918, with scores agreeing within 1.1e-06.

Quantized remex codes on the 256 folded dimensions outperform the 768-dimensional 2-bit baseline (200 bytes, 0.857 recall@10 against float search) once the bit width increases. At 2 bits, the 256-dimensional codes drop to 0.763 recall@10. At 3 bits, they take 104 bytes and reach 0.868 recall@10; 4 bits take 136 bytes for 0.927; 8 bits take 264 bytes for 0.993. Remex's rotation also folds into that 256-to-256 matrix so the model emits rotated coordinates directly, reaching 0.931 recall@10 at 4 bits with an int8 head.

The List Scout page runs the folded 256-dimensional model today, returning identical classifier answers and cutting fitting time from 3.6 s to 1.2 s. The search indexes on the site remain 768-dimensional.
