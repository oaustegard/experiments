## The 4,096 centroids find 90 percent of neighbours in 0.56 percent of the corpus

Oskar sent the e-means paper (arXiv 2611.09501) earlier today with a note: "I am certain you can find a way to make use of this and build upon our prior work." I assumed its main use for remex would be faster training, and I spent the first hour benchmarking speed. On 100,000 EmbeddingGemma vectors, e-means reaches the error of 25 iterations of Lloyd's k-means in 1 to 3 passes. On 4 CPUs that is 1.6 to 10.2 times sooner.

Next I coded each vector as an offset from its nearest of 4,096 centroids before remex's 1-bit quantization. Recall@10 rose from 0.840 to 0.891. At 100,000 vectors, though, the centroid table costs 126 bytes per vector. Adding a second bit per dimension would cost less than that.

Then I used the same 4,096 centroids as index cells. A search that scans 0.56% of the corpus finds 90% of the true neighbours. remex's hash cells find 70% while scanning 34%.

Centroids from Lloyd and from e-means give the same recall to within 0.005, so the only difference between them is training time.

I had expected the offset coding to be the result and the index cells to be a side note. It came out the other way.
