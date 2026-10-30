"""EG2 text embeddings live in a fixed 512-dim linear subspace (text hidden_size 512,
embedding_dim 768). Rotating into that span is exact for dot-product search; compare
top-10 agreement with the 768-dim reference against Matryoshka prefix truncation."""
import glob, json, sys
from pathlib import Path
import numpy as np

D = sys.argv[1]
X = np.concatenate([np.load(f) for f in sorted(glob.glob(f"{D}/shards/docs_*.npy"))]).astype(np.float64)
Q = np.load(f"{D}/q_dense.npy").astype(np.float64)
_, s, Vt = np.linalg.svd(X, full_matrices=False)
B = Vt[:512]
out = {"sv_ratio_512": float(s[511] / s[0]), "sv_ratio_513": float(s[512] / s[0]),
       "query_residual_max": float(np.linalg.norm(Q - Q @ B.T @ B, axis=1).max())}
top = lambda S: np.argsort(-S, 1)[:, :10]
F = top(Q @ X.T)
rec = lambda A: float(np.mean([len(set(a) & set(b)) / 10 for a, b in zip(A, F)]))
Zd, Zq = X @ B.T, Q @ B.T
for k in [512, 384, 256, 128]:
    Xm = X[:, :k] / np.linalg.norm(X[:, :k], axis=1, keepdims=True)
    out[f"recall10_span_{k}"] = rec(top(Zq[:, :k] @ Zd[:, :k].T))
    out[f"recall10_mrl_{k}"] = rec(top(Q[:, :k] @ Xm.T))
print(json.dumps(out, indent=1))
json.dump(out, open(Path(__file__).resolve().parent / "span512.json", "w"), indent=1)
