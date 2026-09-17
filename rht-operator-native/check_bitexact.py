"""Is the operator path bit-identical across compilers/ISAs and against a NumPy reference?"""
import numpy as np
from rhtop import Op, plan

def numpy_ref(X, p, mode):
    """Same elementwise ops, same order as rht_kernel.c."""
    d, B = p["d"], p["B"]
    Y = np.array(X, dtype=np.float32, copy=True)
    n = Y.shape[0]
    def fwht(V):
        V = V.reshape(n, d // B, B)
        h = 1
        while h < B:
            V2 = V.reshape(n, d // B, B // (2 * h), 2, h)
            a = V2[..., 0, :].copy(); b = V2[..., 1, :].copy()
            V2[..., 0, :] = a + b
            V2[..., 1, :] = a - b
            h *= 2
        return V.reshape(n, d)
    rs = range(p["rounds"]) if mode == 0 else reversed(range(p["rounds"]))
    for r in rs:
        perm, ss = p["perms"][r], p["ss"][r]
        if mode == 0:
            Y = fwht(Y[:, perm] * ss)
        else:
            Y = fwht(Y)
            Z = np.empty_like(Y); Z[:, perm] = Y * ss; Y = Z
    return Y

if __name__ == "__main__":
  rng = np.random.default_rng(5)
  for d in (6, 100, 384, 768, 1024, 1536, 3072, 4096):
      X = (rng.standard_normal((500, d)) * rng.lognormal(0, 2, (500, 1))).astype(np.float32)
      outs = {}
      for v in ("native", "nativenc", "sse2", "O0"):
          op = Op(d, 42, v)
          outs[v] = (op.rotate_rows(X), op.unrotate_rows(X))
      p = plan(d, 42)
      outs["numpy"] = (numpy_ref(X, p, 1), numpy_ref(X, p, 0))
      ref = outs["O0"]
      res = {k: all(np.array_equal(a, b) for a, b in zip(v, ref)) for k, v in outs.items()}
      print(d, res)
