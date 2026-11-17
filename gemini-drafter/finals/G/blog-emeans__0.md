## Centroid cells from e-means hold 90% of true neighbours while scanning 0.56% of the corpus

Remex's hash cells hold 70% of true neighbours while scanning 34% of the corpus. If you replace those cells with 4,096 e-means centroids, 90% of true neighbours sit inside 0.56% of the corpus. E-means also fits those centroids 1.6 to 10.2 times sooner than 25 iterations of Lloyd's k-means.

Oskar sent the e-means page (arXiv 2611.09501) on 2026-11-10 and said: "I am certain you can find a way to make use of this and build upon our prior work." I assumed e-means would mainly speed up training, so I spent the first hour benchmarking speed. On 100,000 EmbeddingGemma vectors on 4 CPUs, e-means matches the error of 25 Lloyd iterations in 1 to 3 passes.

Next I coded each vector as an offset from its nearest of 4,096 centroids before remex's 1-bit quantization. Recall@10 rose from 0.840 to 0.891. At 100,000 vectors, though, the centroid table costs 126 bytes per vector. A second bit per dimension would cost less than that.

I had expected the offset coding to be the win and the index cells to be a side note. The results went the other way. Lloyd centroids and e-means centroids give the same recall within 0.005, so the choice between the two algorithms affects training time and nothing else.
