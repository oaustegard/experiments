"""Compiled structured RHT vs dense BLAS, every call shape remex uses."""
import json, sys, time
import numpy as np
from threadpoolctl import threadpool_info
from remex.rotation import rht_rotation
from rhtop import Op, plan

REPEAT = int(sys.argv[1]) if len(sys.argv) > 1 else 7
variant = sys.argv[2] if len(sys.argv) > 2 else "native"

def best(fn, repeat=REPEAT):
    fn()
    ts = []
    for _ in range(repeat):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    return min(ts)

rng = np.random.default_rng(1)
rows = []
print("blas:", [(i["internal_api"], i["num_threads"]) for i in threadpool_info()], "variant:", variant)
print(f"{'d':>5} {'shape':>9} {'dense ms':>10} {'op ms':>10} {'op/dense':>9}")
for d in (384, 768, 1024, 1536, 2048, 3072, 4096):
    t_build_dense = best(lambda: rht_rotation(d, 42), 3)
    t_build_plan = best(lambda: plan(d, 42), 3)
    R = rht_rotation(d, 42); op = Op(d, 42, variant)
    RT = np.ascontiguousarray(R.T)
    rows.append(dict(d=d, shape="build", dense=t_build_dense, op=t_build_plan,
                     dense_bytes=R.nbytes, op_bytes=op.p["perms"].nbytes + op.p["ss"].nbytes))
    q = rng.standard_normal(d).astype(np.float32)
    cells = [("query1d", lambda: R @ q, lambda: op.rotate_query(q))]
    for n in (1, 64, 10000):
        X = rng.standard_normal((n, d)).astype(np.float32)
        cells.append((f"enc n={n}", lambda X=X: X @ R.T, lambda X=X: op.rotate_rows(X)))
    Xd = rng.standard_normal((10000, d)).astype(np.float32)
    cells.append(("dec n=1e4", lambda: Xd @ R, lambda: op.unrotate_rows(Xd)))
    for name, fd, fo in cells:
        rep = 3 if "1e4" in name or "10000" in name else REPEAT
        a, b = best(fd, rep), best(fo, rep)
        rows.append(dict(d=d, shape=name, dense=a, op=b))
        print(f"{d:>5} {name:>9} {a*1e3:>10.3f} {b*1e3:>10.3f} {b/a:>9.3f}")
    print(f"{d:>5} {'build':>9} {t_build_dense*1e3:>10.2f} {t_build_plan*1e3:>10.2f}   "
          f"resident {R.nbytes/1e6:.1f} MB vs {rows[-6]['op_bytes']/1e3:.1f} KB")
    sys.stdout.flush()
json.dump(rows, open(f"results_apply_{variant}.json", "w"), indent=1)
