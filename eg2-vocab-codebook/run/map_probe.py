"""How much of the trained table's drift from EG2's rows is a linear map? Ridge EG2 row -> trained
row on 90% of the 41,741 kept rows, cosine on the held-out 10%, overall and by frequency band."""
import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from vc import LS, WORK, cut_rows, teacher_table  # noqa: E402

sd = torch.load(LS / ".work/mac/out/l1/mac_k4_v32k/final.pt", map_location="cpu")
Y = sd["tm.embed_tokens.weight"].numpy().astype(np.float64)
rows = cut_rows()
X = np.asarray(teacher_table()[rows], np.float64)
f = np.load(WORK / "freq.npy")[rows]
rng = np.random.default_rng(0)
te = rng.random(len(rows)) < 0.1
def cos(a, b): return (a * b).sum(1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)
print("identity cos (test)", cos(X[te], Y[te]).mean())
Xa = np.hstack([X, np.ones((len(X), 1))])
for lam in [1e-4, 1e-3, 1e-2, 1e-1]:
    for wname, w in [("unif", np.ones(len(X))), ("logf", np.log1p(f) + 1)]:
        W = w[~te, None]
        A = np.linalg.solve(Xa[~te].T @ (Xa[~te] * W) + lam * np.eye(513), Xa[~te].T @ (Y[~te] * W))
        P = Xa[te] @ A
        c = cos(P, Y[te])
        ft = f[te]
        bands = {b: round(float(c[(ft >= lo) & (ft < hi)].mean()), 3) for b, (lo, hi) in
                 {"f<10": (0, 10), "10-100": (10, 100), "100-1k": (100, 1000), ">1k": (1000, 1e18)}.items()}
        print(lam, wname, "test cos", round(float(c.mean()), 4), bands, "train cos", round(float(cos(Xa[~te] @ A, Y[~te]).mean()), 4))
# drift vs frequency
c0 = cos(X, Y)
for lo, hi in [(0, 1), (1, 10), (10, 100), (100, 1000), (1000, 1e18)]:
    m = (f >= lo) & (f < hi)
    print("identity cos by freq", lo, hi, m.sum(), round(float(c0[m].mean()), 3), "norm ratio", round(float((np.linalg.norm(Y[m], axis=1) / np.linalg.norm(X[m], axis=1)).mean()), 3))
