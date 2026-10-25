"""Inputs and references for knn-f16 (knn-1's docs as float16, same queries) and knn-cached (1,024 new queries)."""
import json, time
import numpy as np
docs = np.load("docs.npy"); qs = np.load("queries.npy")
dh = docs.astype(np.float16); np.save("docs_f16.npy", dh)
dhf = dh.astype(np.float32)
rng = np.random.default_rng(20261026)
src = rng.integers(0, len(docs), 1024)
q2 = docs[src] + 0.03 * rng.standard_normal((1024, 256), dtype=np.float32); q2 = (q2 / np.linalg.norm(q2, axis=1, keepdims=True)).astype(np.float32)
np.save("queries2.npy", q2)
def topk(q, d):
    s = q @ d.T; i = np.argpartition(-s, 10, axis=1)[:, :10]
    return np.take_along_axis(i, np.argsort(-np.take_along_axis(s, i, 1), 1), 1).astype(np.uint32)
np.save("ref_f16_idx.npy", topk(qs, dhf)); np.save("ref2_f16_idx.npy", topk(q2, dhf)); np.save("ref2_f32_idx.npy", topk(q2, docs))
t = []
for _ in range(3):
    t0 = time.perf_counter(); topk(q2, docs); t.append(time.perf_counter() - t0)
print(json.dumps({"numpy_1024q_ms": round(sorted(t)[1] * 1000, 1), "f16_vs_f32_ref_recall": float(np.mean([len(set(a) & set(b)) / 10 for a, b in zip(np.load("ref_f16_idx.npy"), np.load("ref_idx.npy"))]))}))
