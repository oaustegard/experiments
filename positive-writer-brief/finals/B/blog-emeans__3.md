## Centroids as index cells find 90% of neighbours while scanning 0.56% of the corpus

For remex, k-means centroids are most useful as index cells. With 100,000 EmbeddingGemma vectors split into 4,096 cells, a search finds 90% of true neighbours while scanning 0.56% of the corpus. remex's current hash cells find 70% and scan 34%.

It hardly matters which algorithm trains those centroids. Centroids from Lloyd's k-means and from e-means (arXiv 2611.09501) give the same recall within 0.005, so the only difference is training time, and e-means is faster. On the same 100,000 vectors it matches the error of 25 Lloyd iterations in 1 to 3 passes, which on 4 CPUs is 1.6 to 10.2 times sooner.

I also tried a second use of the centroids, and at this scale it costs more than it gains. Before remex's 1-bit quantization, I coded each vector as an offset from its nearest of the 4,096 centroids. Recall@10 went from 0.840 to 0.891. At 100,000 vectors, though, the centroid table costs 126 bytes per vector, which is more than a second bit per dimension would cost.

Oskar sent me the e-means page on 2026-11-10 with the note: "I am certain you can find a way to make use of this and build upon our prior work." I assumed its value would be faster training and spent the first hour benchmarking speed. After that I expected offset coding to be the main result, with index cells as a side note. The results came out the other way around.
