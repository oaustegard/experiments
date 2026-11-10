## Centroid cells hold 90% of true neighbours while scanning 0.56% of the corpus

Used as index cells, 4,096 centroids hold 90% of true neighbours while scanning 0.56% of the corpus. Remex's hash cells hold 70% while scanning 34%. I had expected offset coding to be the win and the index cells to be a side note, and it came out the other way.

Oskar sent the e-means page (arXiv 2611.09501) on 2026-11-10 and said: "I am certain you can find a way to make use of this and build upon our prior work." I assumed e-means would mainly speed up training, so I spent the first hour benchmarking speed. On 100,000 EmbeddingGemma vectors, e-means matches the error of 25 iterations of Lloyd's k-means in 1 to 3 passes, which is 1.6 to 10.2 times sooner on 4 CPUs.

Next I tried coding each vector as an offset from its nearest of the 4,096 centroids before remex's 1-bit quantization. Recall@10 rose from 0.840 to 0.891. At 100,000 vectors, though, the centroid table costs 126 bytes per vector, which is more than adding a second bit per dimension would cost.

The index cells use those same 4,096 centroids, and it does not matter much how they were trained. Lloyd and e-means centroids give the same recall to within 0.005, so the choice between them comes down to training time.
