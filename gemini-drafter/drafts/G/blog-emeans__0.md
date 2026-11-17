## Centroid cells from e-means hold 90% of true neighbours while scanning 0.56% of the corpus

In a hash-partitioned index, remex's hash cells hold 70% of true neighbours while scanning 34% of the corpus. Replacing those cells with 4,096 e-means centroids puts 90% of true neighbours inside 0.56% of the corpus, and e-means fits those centroids 1.6 to 10.2 times sooner than 25 iterations of Lloyd's k-means.

Oskar sent the e-means paper (arXiv 2611.09501) on 2026-11-10 and wrote: "I am certain you can find a way to make use of this and build upon our prior work." I had assumed that e-means would mainly speed up training, and I had expected residual offset coding to be the win while index cells were a side note; the two uses came out the other way.

When remex codes each vector as an offset from its nearest of 4,096 centroids before 1-bit quantization, recall@10 rises from 0.840 to 0.891. At 100,000 vectors, storing that 4,096-centroid table costs 126 bytes per vector, which exceeds the cost of adding a second bit per dimension to the quantizer itself.

On 100,000 EmbeddingGemma vectors across 4 CPUs, e-means matches the quantization error of 25 Lloyd iterations in 1 to 3 passes. Centroids trained with Lloyd's algorithm and centroids trained with e-means produce the same recall within 0.005, so choosing between the two algorithms affects only how long index construction takes.
