## Answer

Yes. Ettin-17m embeddings can be stored at 256 dimensions without any loss in float search. The model's last layer is a 256-to-768 output matrix, and it folds into a 256-to-256 matrix. On all 9,500 queries, hit@10 is 0.918 with the original matrix and 0.918 with the folded one. Scores from the two agree within 1.1e-06.

Quantized, the 256 folded dimensions need more bits per dimension than the live index uses. The live index stores 2-bit remex codes on 768 dimensions: 200 bytes per vector, with recall@10 of 0.857 against float search. Codes on the folded dimensions give:

| remex codes | bytes | recall@10 |
|---|---|---|
| 2 bits × 256 | — | 0.763 |
| 3 bits × 256 | 104 | 0.868 |
| 4 bits × 256 | 136 | 0.927 |
| 8 bits × 256 | 264 | 0.993 |

At the same bit width the 256-dimension codes do worse (0.763 against 0.857). At 3 and 4 bits they beat the live index's recall in fewer bytes. At 8 bits they beat it too, but they take 264 bytes against its 200.

remex's rotation folds into the same matrix, so the model can output rotated coordinates directly. With an int8 head, 4-bit codes on those coordinates reach 0.931.

The List Scout page runs the folded model. Its classifier answers are identical, and fitting takes 1.2 s where it used to take 3.6 s. The search indexes on the site are still 768-dimensional.
