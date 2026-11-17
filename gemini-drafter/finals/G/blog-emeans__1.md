## 4,096 e-means centroids hold 90% of true neighbours while scanning 0.56% of the corpus

Used as index cells in remex, 4,096 e-means centroids hold 90% of true neighbours while scanning 0.56% of the corpus. remex's hash cells hold 70% while scanning 34%.

Oskar sent the e-means page (arXiv 2611.09501) on 2026-11-10 and said: "I am certain you can find a way to make use of this and build upon our prior work." I assumed e-means would mainly speed up training, and I spent the first hour benchmarking speed. On 100,000 EmbeddingGemma vectors and 4 CPUs, e-means matches the error of 25 iterations of Lloyd's k-means in 1 to 3 passes. It gets there 1.6 to 10.2 times sooner.

Next I coded each vector as an offset from its nearest of 4,096 centroids before remex's 1-bit quantization. Recall@10 rose from 0.840 to 0.891. At 100,000 vectors, though, the centroid table costs 126 bytes per vector. A second bit per dimension would cost less than that.

Then I used the centroids as index cells, and that test produced the numbers above. I had expected the offset coding to be the win and the index cells to be a side note. It came out the other way. Centroids from Lloyd's k-means and from e-means give the same recall within 0.005, so the choice between them is about training time only.
