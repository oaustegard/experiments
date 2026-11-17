## 4,096 e-means centroids hold 90% of true neighbours while scanning 0.56% of the corpus

A vector search index that partitions vectors with hash cells scans a large share of the corpus to catch true neighbours. In remex, 4,096 e-means centroids used as index cells hold 90% of true neighbours while scanning 0.56% of 100,000 EmbeddingGemma vectors; remex's hash cells hold 70% while scanning 34%.

Oskar sent the e-means paper (arXiv 2611.09501) on 2026-11-10 and wrote: "I am certain you can find a way to make use of this and build upon our prior work." I had assumed that e-means would mainly speed up training, and that coding vectors as offsets would be the win while index cells would be a side note; it came out the other way.

When I coded each vector as an offset from its nearest of 4,096 centroids before remex's 1-bit quantization, recall@10 rose from 0.840 to 0.891. At 100,000 vectors, storing the centroid table costs 126 bytes per vector, which exceeds the cost of adding a second bit per dimension.

Using the centroids as index cells instead gives the 90% neighbour capture at a 0.56% scan rate. Centroids from Lloyd's k-means and centroids from e-means give the same recall within 0.005, so the choice between the two algorithms affects training time alone. On the 100,000 EmbeddingGemma vectors across 4 CPUs, e-means matches the error of 25 iterations of Lloyd's k-means in 1 to 3 passes, finishing 1.6 to 10.2 times sooner.
