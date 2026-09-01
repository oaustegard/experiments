"""If the tiny LM earns nothing, the embedder IS the in-browser classifier.

Which small ONNX-available encoder gives the best no-API accuracy, and at what
download size? Scored on the full WANDS set (860 labels, 468 queries), snapping the
RAW QUERY — no model call at all. The Gemini register-prompt arm is the ceiling a
server round-trip would buy.
"""
import json, sys, time
import numpy as np
from sentence_transformers import SentenceTransformer
sys.path.insert(0, '/home/user/muninn-utilities')
import bench

MODELS = [
    ("sentence-transformers/all-MiniLM-L6-v2",   "Xenova/all-MiniLM-L6-v2",       23.0),
    ("BAAI/bge-small-en-v1.5",                   "Xenova/bge-small-en-v1.5",      33.0),
    ("thenlper/gte-small",                       "Xenova/gte-small",              33.0),
    ("sentence-transformers/all-MiniLM-L12-v2",  "Xenova/all-MiniLM-L12-v2",      33.0),
    ("BAAI/bge-base-en-v1.5",                    "Xenova/bge-base-en-v1.5",      109.0),
]

A = json.load(open("artifacts_lite.json"))
vocab, queries, goldn = A["vocab"], A["queries"], A["gold"]
reg = json.load(open("register_hall.json"))
gold = np.array([vocab.index(c) for c in goldn])


def norm(a): return a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-9, None)


def score(m, texts):
    L = norm(np.asarray(m.encode(vocab, batch_size=128, show_progress_bar=False)))
    Q = norm(np.asarray(m.encode(texts, batch_size=128, show_progress_bar=False)))
    order = np.argsort(-(Q @ L.T), axis=1)[:, :3]
    return (float(np.mean([g == r[0] for r, g in zip(order, gold)])),
            float(np.mean([g in r for r, g in zip(order, gold)])))


out = {}
print(f"{'encoder':34} {'int8 MB':>8} {'query@1':>8} {'query@3':>8} {'+gemini@1':>10}")
print("-" * 74)
for hf, onnx, mb in MODELS:
    try:
        t = time.time()
        m = SentenceTransformer(hf)
        q1, q3 = score(m, queries)
        g1, _ = score(m, reg)
        out[hf] = {"onnx_repo": onnx, "int8_mb": mb, "query_acc1": q1,
                   "query_acc3": q3, "gemini_register_acc1": g1}
        print(f"{hf.split('/')[-1]:34} {mb:8.0f} {q1:8.3f} {q3:8.3f} {g1:10.3f}", flush=True)
        del m
    except Exception as e:                                        # noqa: BLE001
        print(f"{hf.split('/')[-1]:34}  FAILED {type(e).__name__}: {str(e)[:50]}", flush=True)
json.dump(out, open("browser_embedders.json", "w"), indent=1)
