import sys, json, glob, numpy as np, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/austegard/Projects/ane-search")
from common import *; import kb
from remex import Quantizer
import os; B = os.path.dirname(os.path.abspath(__file__))
R = kb.Reader(); F = np.load(f"{B}/float768.npy").astype(np.float32); n = min(R.n, len(F)); F = F[:n]; live = R.live[:n]
Q = {}
for f in glob.glob(f"{B}/queries_*.json"): Q.update({int(k): v for k, v in json.load(open(f)).items() if v})
Q = {i: q for i, q in Q.items() if i < n and live[i]}
emb = Embedder(docs=False); items = sorted(Q.items()); QV = np.stack([emb.query(q) for _, q in items]); tgt = np.array([i for i, _ in items])
unit = lambda X: X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)
def remex(d, bits):
    qz = Quantizer(d=d, bits=bits, seed=0); return qz.decode(qz.encode(F[:, :d])).astype(np.float32)
D = {"float 768": F, "float 256": unit(F[:, :256]), "remex 4-bit x 256 (current)": R.V[:n], "remex 4-bit x 768": remex(768, 4), "remex 8-bit x 256": remex(256, 8), "float 384": unit(F[:, :384])}
def order(s): s = np.where(live, s, -9); return np.lexsort((np.arange(n), -s))
dense = {k: [order(V @ unit(q[None, :V.shape[1]])[0]) for q in QV] for k, V in D.items()}
lexs = [R.keyword(q)[:n] for _, q in items]; lex = [[i for i in order(s) if s[i] > 0] for s in lexs]
def rrf(a, b, wa=1.0, wb=1.0, depth=200, c=60):
    sc = {}
    for r, i in enumerate(a[:depth]): sc[int(i)] = sc.get(int(i), 0) + wa / (c + r)
    for r, i in enumerate(b[:depth]): sc[int(i)] = sc.get(int(i), 0) + wb / (c + r)
    return sorted(sc, key=lambda i: (-sc[i], i))
paths = [m["path"] for m in R.meta]
def report(name, ranks):
    pos = []; fpos = []
    for t, r in zip(tgt, ranks):
        r = list(r[:200]); pos.append(r.index(t) if t in r else 999)
        fpos.append(next((k for k, i in enumerate(r) if paths[i] == paths[t]), 999))
    pos = np.array(pos); fpos = np.array(fpos)
    row = {"hit@1": (pos < 1).mean(), "hit@5": (pos < 5).mean(), "hit@10": (pos < 10).mean(), "mrr@10": np.where(pos < 10, 1 / (pos + 1), 0).mean(), "file@5": (fpos < 5).mean()}
    print(f"{name:42}" + "  ".join(f"{k} {v:.3f}" for k, v in row.items())); return {k: round(float(v), 3) for k, v in row.items()}
out = {"queries": len(items)}; print(len(items), "queries over", int(live.sum()), "passages")
for k in D: out["semantic: " + k] = report("semantic: " + k, dense[k])
out["keyword: bm25s"] = report("keyword: bm25s", lex)
cur = "remex 4-bit x 256 (current)"
for k in (cur, "float 768", "remex 4-bit x 768"):
    out["hybrid 1:1, " + k] = report("hybrid 1:1, " + k, [rrf(a, b) for a, b in zip(dense[k], lex)])
for wb in (0.5, 0.3, 0.15):
    out[f"hybrid 1:{wb}, current codes"] = report(f"hybrid 1:{wb}, current codes", [rrf(a, b, 1, wb) for a, b in zip(dense[cur], lex)])
    out[f"hybrid 1:{wb}, remex 4x768"] = report(f"hybrid 1:{wb}, remex 4x768", [rrf(a, b, 1, wb) for a, b in zip(dense["remex 4-bit x 768"], lex)])
json.dump(out, open(f"{B}/eval.json", "w"), indent=1)
