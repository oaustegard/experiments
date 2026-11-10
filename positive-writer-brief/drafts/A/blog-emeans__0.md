## The centroids work best as index cells

Used as index cells, 4,096 e-means centroids hold 90% of true neighbours while scanning 0.56% of the corpus. Remex's hash cells hold 70% while scanning 34%. I had expected this result to be a side note.

Oskar sent the e-means page (arXiv 2611.09501) earlier today with one line: "I am certain you can find a way to make use of this and build upon our prior work." I assumed the gain would mostly be in training speed and spent the first hour benchmarking that. On 100,000 EmbeddingGemma vectors, e-means matches the error of 25 iterations of Lloyd's k-means in 1 to 3 passes. On 4 CPUs it gets there 1.6 to 10.2 times sooner.

Next I tried coding each vector as an offset from its nearest of the 4,096 centroids before remex's 1-bit quantization. Recall@10 rose from 0.840 to 0.891, and I expected this to be the main result. At 100,000 vectors, though, the centroid table costs 126 bytes per vector. A second bit per dimension would cost less than that.

Lloyd and e-means centroids give the same recall within 0.005, so the choice between them comes down to training time.
