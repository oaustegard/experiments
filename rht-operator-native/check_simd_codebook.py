"""Boundaries from float64 centroids drift with NumPy SIMD level; from float32 centroids they do not.
Run plain, then with NPY_DISABLE_CPU_FEATURES=X86_V4, then with \"X86_V4 X86_V3\"."""
import hashlib, numpy as np
from remex import Quantizer
from remex.codebook import lloyd_max_codebook
from rhtop import Op
sha = lambda a: hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:12]
def fixed(d, bits):
    _, c = lloyd_max_codebook(d, bits)
    return ((c[:-1].astype(np.float64) + c[1:]) / 2).astype(np.float32)
rows = []
for d, bits in ((768, 2), (768, 4), (768, 8), (3072, 4), (3072, 8), (384, 4), (1536, 3)):
    b_old, _ = lloyd_max_codebook(d, bits); b_new = fixed(d, bits)
    rows.append(f"{d}/{bits} old {sha(b_old)} new {sha(b_new)} newdiff {int((b_old != b_new).sum())}")
d = 768
X = np.random.default_rng(7).standard_normal((2000, d)).astype(np.float32)
q = Quantizer(d, 4, seed=42, rotation="rht"); q._rotate_rows = Op(d, 42).rotate_rows
old = sha(q.encode(X).indices); q.boundaries = fixed(d, 4); new = sha(q.encode(X).indices)
print("\n".join(rows)); print("op codes d=768/4: old-bounds", old, "fixed-bounds", new)
