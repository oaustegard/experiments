## Centroids as index cells find 90% of neighbours while scanning 0.56% of the corpus

For remex, k-means centroids are most useful as index cells. Partitioned into 4,096 centroids, a corpus of 100,000 EmbeddingGemma vectors gives 90% of true neighbours while scanning 0.56% of it. remex's current hash cells give 70% of true neighbours and scan 34%.

Which algorithm trains those centroids hardly matters. Centroids from Lloyd's k-means and from e-means (arXiv 2611.09501) give the same recall within 0.005. That leaves training time as the only difference, and there e-means is faster. On the same 100,000 vectors it matches the error of 25 Lloyd iterations in 1 to 3 passes. On 4 CPUs that is 1.6 to 10.2 times sooner.

I also tried a second use of the centroids, and it doesn't pay for itself at this scale. Before remex applies its 1-bit quantization, I coded each vector as an offset from its nearest centroid among the 4,096. Recall@10 went from 0.840 to 0.891. At 100,000 vectors, though, the centroid table costs 126 bytes per vector. A second bit per dimension would cost less than that, so for a corpus this size the extra bit is the cheaper way to buy recall.

Oskar sent me the e-means paper on 2026-11-10 with the note: "I am certain you can find a way to make use of this and build upon our prior work." I assumed its value would be faster training and spent the first hour benchmarking speed. Next I expected offset coding to be the main result, with index cells as a side note. The results came out the other way around.

Next I will replace remex's hash cells with 4,096 trained cells, trained with e-means because it gives the same centroids in less time.
