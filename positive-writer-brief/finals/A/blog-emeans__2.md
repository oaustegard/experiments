## The e-means centroids are worth more to remex as index cells than as offsets

Used as index cells, the 4,096 centroids that e-means trains hold 90% of true neighbours while scanning 0.56% of the corpus. remex's hash cells hold 70% while scanning 34%.

Oskar sent the e-means page (arXiv 2611.09501) on 2026-11-10 and wrote: "I am certain you can find a way to make use of this and build upon our prior work." I assumed the gain would mostly be training speed and spent the first hour benchmarking it. On 100,000 EmbeddingGemma vectors, e-means matches the error of 25 iterations of Lloyd's k-means in 1 to 3 passes. On 4 CPUs it gets there 1.6 to 10.2 times sooner.

Next I coded each vector as an offset from its nearest of the 4,096 centroids before remex's 1-bit quantization. Recall@10 rose from 0.840 to 0.891. At 100,000 vectors, though, the centroid table costs 126 bytes per vector, which is more than adding a second bit per dimension would cost.

The index cells came after that. Centroids from Lloyd and from e-means give the same recall within 0.005, so the choice between the two algorithms affects only training time.

I had expected the offset coding to be the main result and the index cells to be a side note, and it came out the other way around.
