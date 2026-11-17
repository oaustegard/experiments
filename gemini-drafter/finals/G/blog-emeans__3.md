## 4,096 e-means centroids hold 90% of true neighbours inside 0.56% of the corpus

If you partition a vector index with hash cells, learned centroids cut the scan fraction and raise coverage at the same time. In remex on 100,000 EmbeddingGemma vectors, 4,096 centroids used as index cells hold 90% of true neighbours while scanning 0.56% of the corpus. Remex's hash cells hold 70% while scanning 34%.

Oskar sent the e-means paper (arXiv 2611.09501) on 2026-11-10 and wrote: "I am certain you can find a way to make use of this and build upon our prior work." On the same 100,000 vectors, e-means matches the error of 25 iterations of Lloyd's k-means in 1 to 3 passes, 1.6 to 10.2 times sooner on 4 CPUs. Lloyd and e-means centroids give the same recall within 0.005, so choosing between the two algorithms affects only training time.

I had assumed e-means would mainly speed up training, and I spent the first hour benchmarking speed. After that I expected offset coding to be the win and the index cells to be a side note. It came out the other way. When I coded each vector as an offset from its nearest of the 4,096 centroids before remex's 1-bit quantization, recall@10 rose from 0.840 to 0.891. At 100,000 vectors, though, the centroid table costs 126 bytes per vector, which is more than adding a second bit per dimension would cost.
