import json, sys, time, numpy as np
import bench
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

vocab, qs = bench.load()
queries = [q for q, _ in qs]
gold = np.array([vocab.index(c) for _, c in qs])
print(f"labels={len(vocab)} queries={len(queries)}", flush=True)

t = time.time()
hall = bench.hallucinate(queries, model="lite")
t_hall = time.time() - t
json.dump({"queries": queries, "gold": [c for _, c in qs], "hall": hall,
           "vocab": vocab}, open("artifacts_lite.json", "w"))
print(f"hallucinated {len(hall)} in {t_hall:.0f}s  e.g. {list(zip(queries[:4], hall[:4]))}", flush=True)

m = SentenceTransformer("all-MiniLM-L6-v2")
t = time.time(); L = m.encode(vocab, batch_size=128, show_progress_bar=False)
Q = m.encode(queries, batch_size=128, show_progress_bar=False)
H = m.encode(hall, batch_size=128, show_progress_bar=False)
print(f"encoded in {time.time()-t:.0f}s", flush=True)

tf = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit(vocab + queries + hall)
Lx, Qx, Hx = tf.transform(vocab).toarray(), tf.transform(queries).toarray(), tf.transform(hall).toarray()

ARMS = {
  "lexical char-ngram: query -> label": (Qx, Lx),
  "lexical char-ngram: hallucination -> label": (Hx, Lx),
  "MiniLM: query -> label  [control]": (Q, L),
  "MiniLM: hallucination -> label  [Doug]": (H, L),
  "MiniLM: (query+hallucination)/2 -> label": ((bench.norm(Q) + bench.norm(H)) / 2, L),
}
out = {}
print(f"\n{'arm':46} {'acc@1':>6} {'acc@3':>6}")
print("-" * 62)
for name, (V, Lv) in ARMS.items():
    idx = bench.snap(V, Lv, k=3)
    a1, a3 = bench.acc(idx, gold, 1), bench.acc(idx, gold, 3)
    out[name] = {"acc@1": a1, "acc@3": a3}
    print(f"{name:46} {a1:6.3f} {a3:6.3f}")
json.dump(out, open("results_lite.json", "w"), indent=1)
print(f"\nhallucination wall-clock: {t_hall:.0f}s for {len(queries)} queries")
