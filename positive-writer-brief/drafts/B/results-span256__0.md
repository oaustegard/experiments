## Answer

Yes. Ettin-17m embeddings can be stored at 256 dimensions with no loss in float search, and quantized 256-dim codes beat the live 768-dim index at fewer bytes.

Oskar asked on 2026-11-09: "does that mean we could trim these to 256 length vectors without penalty by reversing the widening?"

The model's final output matrix maps 256 dimensions to 768. That matrix folds into a 256-to-256 matrix, so the model emits 256-dim vectors directly. On all 9,500 queries, float search gives hit@10 0.918 at both 768 and 256 dimensions. Scores agree within 1.1e-06.

The live index stores remex codes at 2 bits × 768 dims: 200 bytes per vector, recall@10 0.857 against float search. Codes on the 256 folded dimensions compare as follows:

| Bits × dims | Bytes | Recall@10 |
|---|---|---|
| 2 × 768 (live) | 200 | 0.857 |
| 2 × 256 | — | 0.763 |
| 3 × 256 | 104 | 0.868 |
| 4 × 256 | 136 | 0.927 |
| 8 × 256 | 264 | 0.993 |

At the same bit width the 256-dim codes lose, so a switch to 256 dims needs at least 3 bits per dimension. remex's rotation also folds into the same matrix, so the model can output rotated coordinates directly. With an int8 head this reaches 0.931 at 4 bits.

The List Scout page runs the folded model now. Its classifier answers are identical, and fitting takes 1.2 s instead of 3.6 s. The site's search indexes are still 768-dimensional.
