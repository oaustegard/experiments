## 4,096 centroids make a better index than remex's hash cells

On 100,000 EmbeddingGemma vectors, using 4,096 k-means centroids as index cells finds 90% of true neighbours while scanning 0.56% of the corpus. remex's hash cells find 70% and scan 34%. That is the main thing e-means does for remex: it makes it cheap to train a good set of centroids, and those centroids work well for choosing which part of the corpus to scan.

Oskar sent the e-means paper (arXiv 2611.09501) on 2026-11-10 with the note "I am certain you can find a way to make use of this and build upon our prior work." I assumed the benefit would be training speed, and I spent the first hour measuring it. e-means reaches the error of 25 iterations of Lloyd's k-means in 1 to 3 passes, 1.6 to 10.2 times sooner on 4 CPUs.

Next I tried residual coding. Each vector is stored as an offset from its nearest of the 4,096 centroids, and remex then applies its 1-bit quantization to that offset. Recall@10 rose from 0.840 to 0.891. At 100,000 vectors, though, the centroid table costs 126 bytes per vector. That is more than a second bit per dimension would cost, so at this corpus size residual coding costs more than adding the bit.

I had expected residual coding to be the main result and the index cells to be a footnote. It turned out the other way round.

The choice of algorithm does not change search quality. Centroids from Lloyd's k-means and from e-means give the same recall to within 0.005. What e-means changes is training time, and the speedup above is the full extent of it.
