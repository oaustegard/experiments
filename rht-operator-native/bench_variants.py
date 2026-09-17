import time, numpy as np
from remex.rotation import rht_rotation
from rhtop import Op, plan
from check_bitexact import numpy_ref
def best(f, r=3):
    f(); return min((lambda t: (f(), time.perf_counter() - t)[1])(time.perf_counter()) for _ in range(r))
rng = np.random.default_rng(2)
print(f"{'d':>5} {'n':>6} {'dense':>9} {'native':>9} {'sse2':>9} {'numpy-ref':>10}  (ms)")
for d in (384, 768, 1024, 3072):
    R = rht_rotation(d, 42); p = plan(d, 42)
    ops = {v: Op(d, 42, v) for v in ("native", "sse2")}
    for n in (1, 10000):
        X = rng.standard_normal((n, d)).astype(np.float32)
        t = [best(lambda: X @ R.T)] + [best(lambda o=o: o.rotate_rows(X)) for o in ops.values()] + [best(lambda: numpy_ref(X, p, 1))]
        print(f"{d:>5} {n:>6} " + " ".join(f"{x*1e3:>9.3f}" for x in t[:3]) + f" {t[3]*1e3:>10.3f}")
