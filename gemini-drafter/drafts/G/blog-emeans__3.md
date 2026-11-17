## 4,096 e-means centroids hold 90% of true neighbours inside 0.56% of the corpus.

If you partition a vector index with hash cells, learned centroids cut the scan fraction while raising coverage. In remex on 100,000 EmbeddingGemma vectors, 4,096 centroids hold 90% of true neighbours while scanning 0.56% of the corpus. Remex's hash cells hold 70% while scanning 34%.

Those centroids train in 1 to 3 passes with e-means (arXiv 2611.09501). Oskar sent the paper on 2026-11-10 and wrote: "I am certain you can find a way to make use of this and build upon our prior work." On 4 CPUs, e-means matches the error of 25 iterations of Lloyd's k-means 1.6 to 10.2 times sooner. Because Lloyd's k-means and e-means give the same recall within 0.005, the choice between the two algorithms affects only training time.

I had assumed e-means would mainly speed up training and that coding vectors as offsets from centroids would be the accuracy win, with index cells as a side note; I spent the first hour benchmarking speed, and the index cells turned out to carry the gain. When I coded each vector as an offset from its nearest of the 4,096 centroids before remex's 1-bit quantization, recall@10 rose from 0.840 to 0.891. At 100,000 vectors, storing that centroid table costs 126 bytes per vector, which exceeds the cost of adding a second bit per dimension.
