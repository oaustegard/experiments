## e-means centroids work best as index cells: 90% of neighbours from 0.56% of the corpus

Remex can use 4,096 k-means centroids as index cells. On 100,000 EmbeddingGemma vectors, those cells hold 90% of each query's true neighbours while scanning 0.56% of the corpus. Remex's current hash cells hold 70% of true neighbours and scan 34% of the corpus to do it.

Oskar sent the e-means paper (arXiv 2611.09501) on 2026-11-10 with a note: "I am certain you can find a way to make use of this and build upon our prior work." Muninn expected the gain to come from somewhere else, so the measurements below follow the order of the tests.

The first test was training speed. E-means reaches the error of 25 Lloyd's iterations in 1 to 3 passes. On 4 CPUs that is 1.6 to 10.2 times sooner. The centroids themselves are interchangeable with Lloyd's, and recall stays within 0.005 whichever algorithm trains them. Choosing e-means over Lloyd's changes how long training takes and nothing else.

The second test coded each vector as an offset from its nearest centroid before remex's 1-bit quantization. Recall@10 rose from 0.840 to 0.891. At 100,000 vectors, though, the centroid table costs 126 bytes per vector, which is more than adding a second bit per dimension would cost. At this corpus size the offset coding loses to the plain alternative of spending one more bit per dimension.

Muninn had expected the offset coding to be the main result and the index cells to be a side note. The index cells gave the larger gain. The next step for remex is to replace the hash cells with 4,096 centroids trained by e-means, because e-means produces the same cells in less training time.
