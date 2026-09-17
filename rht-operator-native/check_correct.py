import numpy as np
from remex.rotation import rht_rotation
from rhtop import Op
rng = np.random.default_rng(0)
print("d     rounds B     max|fwd-dense|  max|inv-dense|  max|R@q-dense|  roundtrip")
for d in (6, 12, 100, 384, 768, 1024, 1536, 2048, 3072, 4096):
    R = rht_rotation(d, 42); op = Op(d, 42)
    X = rng.standard_normal((64, d)).astype(np.float32)
    f = np.abs(op.rotate_rows(X) - X @ R.T).max()
    i = np.abs(op.unrotate_rows(X) - X @ R).max()
    q = np.abs(op.rotate_query(X[0]) - R @ X[0]).max()
    rt = np.abs(op.unrotate_rows(op.rotate_rows(X)) - X).max()
    print(f"{d:<5} {op.p['rounds']:<6} {op.p['B']:<5} {f:.2e}        {i:.2e}        {q:.2e}        {rt:.2e}")
