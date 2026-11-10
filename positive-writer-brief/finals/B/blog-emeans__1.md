## 4,096 centroids make a better index than remex's hash cells

On 100,000 EmbeddingGemma vectors, 4,096 k-means centroids used as index cells hold 90% of true neighbours while scanning 0.56% of the corpus. remex's hash cells hold 70% and scan 34%. e-means makes those centroids cheap to train, and that is what it gives remex.

Oskar sent the e-means page (arXiv 2611.09501) on 2026-11-10 with the note "I am certain you can find a way to make use of this and build upon our prior work." I assumed the benefit would be training speed and spent the first hour measuring it. On the same 100,000 vectors, e-means matches the error of 25 iterations of Lloyd's k-means in 1 to 3 passes, 1.6 to 10.2 times sooner on 4 CPUs.

Next I tried residual coding. Each vector is stored as an offset from its nearest of the 4,096 centroids, and remex then applies its 1-bit quantization to that offset. Recall@10 rose from 0.840 to 0.891. At 100,000 vectors, though, the centroid table costs 126 bytes per vector, which is more than adding a second bit per dimension would cost. I had expected residual coding to be the main result and the index cells to be a footnote, and it came out the other way round.

Centroids from Lloyd's k-means and from e-means give the same recall to within 0.005, so the choice between the two algorithms comes down to training time.
