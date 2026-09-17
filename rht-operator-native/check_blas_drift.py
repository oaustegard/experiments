"""Is dense encode byte-stable across OpenBLAS kernels? Writes codes for one config."""
import sys, os, hashlib
import numpy as np
from remex import Quantizer
from threadpoolctl import threadpool_info
d, bits = int(sys.argv[1]), int(sys.argv[2])
rng = np.random.default_rng(7)
X = rng.standard_normal((20000, d)).astype(np.float32)
q = Quantizer(d, bits, seed=42, rotation="rht")
idx = q.encode(X).indices
tag = os.environ.get("OPENBLAS_CORETYPE", "auto")
np.save(f"/tmp/codes_{tag}_{d}_{bits}.npy", idx)
print(tag, [i.get("architecture") for i in threadpool_info()][0], d, bits, hashlib.sha1(idx.tobytes()).hexdigest()[:12])
