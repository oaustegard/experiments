## Answer

Ettin-17m embeddings can be stored at 256 dimensions without losing float search accuracy, and quantized remex codes on those 256 dimensions beat the live 768-dimensional index in fewer bytes.

On 2026-11-09, Oskar asked: "does that mean we could trim these to 256 length vectors without penalty by reversing the widening?" Ettin-17m's final 256-to-768 output matrix folds into a 256-to-256 matrix. Across 9,500 queries, float search gets 0.918 hit@10 at both 768 and 256 dimensions, and scores agree within 1.1e-06.

Quantized search requires trading bit width against dimension count. The live 768-dimensional remex index uses 2 bits per dimension (200 bytes) and reaches 0.857 recall@10 against float search. At 2 bits on 256 dimensions, recall@10 drops to 0.763. Raising the bit width on 256 dimensions surpasses the live index at smaller sizes: 3 bits takes 104 bytes and reaches 0.868 recall@10, 4 bits takes 136 bytes and reaches 0.927, and 8 bits takes 264 bytes and reaches 0.993. Remex's rotation also folds into the 256-to-256 matrix so that the model emits rotated coordinates directly; with an int8 head, 4-bit codes reach 0.931 recall@10.

The List Scout page runs the folded 256-dimensional model. Classifier answers are identical and fitting takes 1.2 s instead of 3.6 s. The search indexes on the site remain 768-dimensional.
