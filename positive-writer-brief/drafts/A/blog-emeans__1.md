## The 4,096 centroids work best as index cells

Oskar sent the e-means page (arXiv 2611.09501) earlier today with one line: "I am certain you can find a way to make use of this and build upon our prior work." I assumed the payoff would be faster training and spent the first hour benchmarking speed. On 100,000 EmbeddingGemma vectors, e-means matches the error of 25 iterations of Lloyd's k-means in 1 to 3 passes. On 4 CPUs that is 1.6 to 10.2 times sooner.

Next I coded each vector as an offset from its nearest of 4,096 centroids before remex's 1-bit quantization. Recall@10 rose from 0.840 to 0.891. At 100,000 vectors, though, the centroid table costs 126 bytes per vector, which is more than adding a second bit per dimension would cost.

Then I used the same 4,096 centroids as index cells. They hold 90% of true neighbours while scanning 0.56% of the corpus. Remex's hash cells hold 70% while scanning 34%. Lloyd and e-means centroids give the same recall to within 0.005, so the choice between the two algorithms comes down to training time.

I had expected the offset coding to be the win and the index cells to be a side note. The results came out the other way.
