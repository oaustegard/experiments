"""Structured RHT apply via a compiled kernel, on remex.rotation.rht_rotation's plan."""
import ctypes, math, os
from pathlib import Path
import numpy as np
from remex.rotation import _largest_pow2_divisor

HERE = Path(__file__).parent
_f32p = np.ctypeslib.ndpointer(np.float32, flags="C_CONTIGUOUS")
_i32p = np.ctypeslib.ndpointer(np.int32, flags="C_CONTIGUOUS")

def load(variant="native"):
    lib = ctypes.CDLL(str(HERE / f"rht_kernel_{variant}.so"))
    f = lib.rht_apply
    f.restype = ctypes.c_int
    f.argtypes = [_f32p, _f32p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
                  ctypes.c_int64, _i32p, _f32p, ctypes.c_int32, ctypes.c_int32]
    return f

def plan(d, seed=42):
    """Replays rht_rotation's RNG stream: same perms, same signs, same rounds."""
    B = _largest_pow2_divisor(d)
    rng = np.random.default_rng(seed)
    rounds = 1 if B == d else max(2, math.ceil(math.log(d) / math.log(B)))
    perms, ss = [], []
    scale = np.float32(1.0 / math.sqrt(B))
    for _ in range(rounds):
        perms.append(rng.permutation(d).astype(np.int32))
        ss.append(rng.choice(np.array([-1.0, 1.0], np.float32), size=d) * scale)
    return dict(d=d, B=B, rounds=rounds, perms=np.ascontiguousarray(np.stack(perms)),
                ss=np.ascontiguousarray(np.stack(ss).astype(np.float32)))

class Op:
    def __init__(self, d, seed=42, variant="native", threads=None):
        self.p = plan(d, seed); self.f = load(variant)
        self.threads = threads or len(os.sched_getaffinity(0))
    def _run(self, X, mode):
        X = np.ascontiguousarray(X, dtype=np.float32)
        one = X.ndim == 1
        X2 = X.reshape(1, -1) if one else X
        out = np.empty_like(X2)
        p = self.p
        if self.f(X2, out, X2.shape[0], p["d"], p["B"], p["rounds"], p["perms"], p["ss"], mode, self.threads):
            raise MemoryError("rht_apply: scratch allocation failed")
        return out[0] if one else out
    def rotate_rows(self, X):   return self._run(X, 1)   # X @ R.T
    def unrotate_rows(self, X): return self._run(X, 0)   # X @ R
    def rotate_query(self, q):  return self._run(q, 1)   # R @ q
