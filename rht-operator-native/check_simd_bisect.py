"""Bisect encode: under NPY_DISABLE_CPU_FEATURES=X86_V4 the rotated vectors match but codes do not."""
import hashlib, numpy as np
from remex import Quantizer
from rhtop import Op
sha = lambda a: hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:12]
d = 768
X = np.random.default_rng(7).standard_normal((2000, d)).astype(np.float32)
q = Quantizer(d, 4, seed=42, rotation="rht"); op = Op(d, 42)
n32 = np.sqrt(np.sum(X.astype(np.float64) ** 2, axis=1)).astype(np.float32)
den = np.maximum(n32, 1e-8)[:, None]
U = X / den
R = op.rotate_rows(U)
I = np.searchsorted(q.boundaries, R).astype(np.uint8)
I2 = np.searchsorted(q.boundaries.astype(np.float32), R).astype(np.uint8)
print("den", den.dtype, sha(den), "| U", U.dtype, sha(U), "| R", sha(R), "| bounds", q.boundaries.dtype, "| idx", sha(I), "| idx(f32 bounds)", sha(I2))
np.save(f"/tmp/U_{__import__('os').environ.get('TAG','x')}.npy", U)
