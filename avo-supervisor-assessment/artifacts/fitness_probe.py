import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) if False else ".")
t0 = time.time()
import numpy as np
from remex import Quantizer
sys.path.insert(0, "bench")
from benchmark import make_clustered_embeddings, exact_knn, recall_at_k
t_import = time.time() - t0

corpus = make_clustered_embeddings(10000, 384)
queries = make_clustered_embeddings(200, 384, seed=7)
truth = exact_knn(corpus, queries, 10)
t_data = time.time() - t0 - t_import

t1 = time.time()
q = Quantizer(d=384, bits=int(sys.argv[1]) if len(sys.argv) > 1 else 4, rotation="rht")
enc = q.encode(corpus)
import numpy as np
idx = np.stack([q.search(enc, qv, k=10)[0] for qv in queries])
r = recall_at_k(idx, truth, 10)
t_fit = time.time() - t1
print(f"FITNESS {r:.4f}  import={t_import:.2f}s data+truth={t_data:.2f}s encode+search={t_fit:.2f}s")
