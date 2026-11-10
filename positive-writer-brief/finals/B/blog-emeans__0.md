## 4,096 centroids used as index cells hold 90% of true neighbours while scanning 0.56% of the corpus

Remex can use 4,096 k-means centroids as index cells. Those cells hold 90% of true neighbours while scanning 0.56% of the corpus. Remex's hash cells hold 70% and scan 34% to get there.

That was not the result I set out to find. Oskar sent the e-means page (arXiv 2611.09501) on 2026-11-10 with a note: "I am certain you can find a way to make use of this and build upon our prior work." I assumed e-means would mainly speed up training, so I spent the first hour benchmarking speed.

On 100,000 EmbeddingGemma vectors, e-means matches the error of 25 iterations of Lloyd's k-means in 1 to 3 passes. On 4 CPUs that is 1.6 to 10.2 times sooner. Lloyd and e-means centroids give the same recall within 0.005, so choosing between them is a question of training time.

Next I tried coding each vector as an offset from its nearest of the 4,096 centroids before remex's 1-bit quantization. Recall@10 rose from 0.840 to 0.891. But at 100,000 vectors the centroid table costs 126 bytes per vector, which is more than adding a second bit per dimension would cost.

I had expected the offset coding to be the win and the index cells to be a side note. It came out the other way.
