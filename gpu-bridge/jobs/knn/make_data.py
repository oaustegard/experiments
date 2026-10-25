"""Inputs and CPU reference for the knn job: 32,768 docs x 256 dims, 256 queries, exact top-10 by dot product."""
import time, json
import numpy as np
rng = np.random.default_rng(20261025)
N, D, Q, K = 32768, 256, 256, 10
docs = rng.standard_normal((N, D), dtype=np.float32); docs /= np.linalg.norm(docs, axis=1, keepdims=True)
# queries near random docs, so each has a clear nearest neighbour
src = rng.integers(0, N, Q)
qs = docs[src] + 0.03 * rng.standard_normal((Q, D), dtype=np.float32); qs /= np.linalg.norm(qs, axis=1, keepdims=True)
np.save("docs.npy", docs.astype(np.float32)); np.save("queries.npy", qs.astype(np.float32))
t = []
for _ in range(5):
    t0 = time.perf_counter(); s = qs @ docs.T; idx = np.argpartition(-s, K, axis=1)[:, :K]
    idx = np.take_along_axis(idx, np.argsort(-np.take_along_axis(s, idx, 1), 1), 1); t.append(time.perf_counter() - t0)
np.save("ref_idx.npy", idx.astype(np.uint32)); np.save("ref_score.npy", np.take_along_axis(s, idx, 1).astype(np.float32))
print(json.dumps({"cpu_median_ms": round(sorted(t)[2] * 1000, 2), "src_is_top1": float((idx[:, 0] == src).mean())}))
