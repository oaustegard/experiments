## Answer

Ettin-17m embeddings can be stored at 256 dimensions without penalty in float search, and quantized to smaller byte sizes than the 768-dimensional index when given more bits per dimension. On 2026-11-09, Oskar asked, "does that mean we could trim these to 256 length vectors without penalty by reversing the widening?"

Folding Ettin-17m's final 256-to-768 output matrix into a 256-to-256 matrix leaves float search unchanged across all 9,500 queries. Both versions score 0.918 on hit@10, and individual scores agree within 1.1e-06.

Quantizing the 256 folded dimensions with remex beats the live index (2 bits x 768 dimensions, 200 bytes, 0.857 recall@10 against float search) when you raise the bit width: 3 bits takes 104 bytes and reaches 0.868 recall@10, 4 bits takes 136 bytes and reaches 0.927, and 8 bits takes 264 bytes and reaches 0.993. At the same 2-bit width, the 256-dimensional codes drop to 0.763 recall@10. Remex's rotation also folds into the output matrix so that the model emits rotated coordinates directly, which reaches 0.931 recall@10 at 4 bits with an int8 head.

The List Scout page now runs the folded model, producing identical classifier answers while reducing fitting time from 3.6 s to 1.2 s. The search indexes on the site remain 768-dimensional.
