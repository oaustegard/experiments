"""Do operator-rotated codes differ from dense-rotated codes, and does it matter?"""
import numpy as np
from remex import Quantizer
from rhtop import Op
rng = np.random.default_rng(3)
N, Q = 20000, 200
print(f"{'d':>5} {'corpus':>6} {'bits':>4} {'coord flips':>12} {'rows w/ flip':>12} {'R@10 dense':>10} {'R@10 op':>8} {'top10 same':>10}")
for d in (384, 768, 1536, 3072):
    op = Op(d, 42)
    for corpus in ("iso", "aniso"):
        if corpus == "iso":
            X = rng.standard_normal((N, d)).astype(np.float32)
        else:
            ev = np.arange(1, d + 1) ** -1.0
            B, _ = np.linalg.qr(rng.standard_normal((d, d)))
            X = ((rng.standard_normal((N, d)) * np.sqrt(ev)) @ B.T + 0.3 * B[:, 0]).astype(np.float32)
        qs = X[rng.choice(N, Q, replace=False)] + 0.05 * rng.standard_normal((Q, d)).astype(np.float32) * X.std()
        gt = np.argsort(-(qs / np.linalg.norm(qs, axis=1, keepdims=True)) @ (X / np.linalg.norm(X, axis=1, keepdims=True)).T, axis=1)[:, :10]
        for bits in (2, 4, 8):
            qd = Quantizer(d, bits, seed=42, rotation="rht")
            cd = qd.encode(X)
            qo = Quantizer(d, bits, seed=42, rotation="rht")
            qo._rotate_rows = op.rotate_rows
            co = qo.encode(X)
            diff = cd.indices != co.indices
            def rec(qz, c):
                top = np.stack([qz.search(c, qq, k=10)[0] for qq in qs])
                return top, np.mean([len(set(a) & set(b)) / 10 for a, b in zip(top, gt)])
            td, rd = rec(qd, cd); to, ro = rec(qd, co)
            same = np.mean([set(a) == set(b) for a, b in zip(td, to)])
            print(f"{d:>5} {corpus:>6} {bits:>4} {diff.mean():>12.2e} {diff.any(1).mean():>12.4f} {rd:>10.4f} {ro:>8.4f} {same:>10.3f}", flush=True)
