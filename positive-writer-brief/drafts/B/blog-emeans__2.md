## E-means centroids make remex's index cells 60 times more selective

Used as index cells, 4,096 centroids hold 90% of true neighbours while scanning 0.56% of the corpus. Remex's hash cells hold 70% of true neighbours while scanning 34%. Both figures come from 100,000 EmbeddingGemma vectors.

Oskar sent the e-means paper (arXiv 2611.09501) on 2026-11-10 with one line: "I am certain you can find a way to make use of this and build upon our prior work." I expected the payoff to be offset coding and treated the index cells as a side note. The results came out the other way.

Training speed is the gain the paper advertises. E-means matches the error of 25 iterations of Lloyd's k-means in 1 to 3 passes. On 4 CPUs that is 1.6 to 10.2 times sooner. Lloyd and e-means centroids give the same recall to within 0.005. You choose between them on training time alone.

Offset coding was the first use I tested. Each vector is coded as an offset from its nearest of the 4,096 centroids before remex quantizes it to 1 bit per dimension. Recall@10 rose from 0.840 to 0.891. The centroid table costs 126 bytes per vector at 100,000 vectors, though. A second bit per dimension costs less than that. At this corpus size, offset coding is the more expensive way to buy recall.

The centroids are more useful for routing queries than for coding vectors. The next step for remex is to replace its hash cells with centroid cells, trained by e-means if training time matters to you and by Lloyd if it does not.
