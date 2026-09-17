"""One-shot run for any machine: correctness, bit-exactness, fingerprints, speed.

Writes ci/<label>.json. Usage: python3 ci_run.py <label> [--quick]
"""
import hashlib, json, os, platform, sys, time
import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits
from remex import Quantizer
from remex.rotation import rht_rotation
from rhtop import Op, plan, ncpu as _ncpu
from check_bitexact import numpy_ref

label = sys.argv[1]
quick = "--quick" in sys.argv
ncpu = _ncpu()
res = dict(label=label, machine=platform.machine(), processor=platform.processor(),
           system=platform.system(), ncpu=ncpu, numpy=np.__version__,
           blas=[{k: i.get(k) for k in ("internal_api", "architecture", "num_threads", "version")} for i in threadpool_info()])
print(json.dumps(res, indent=1))

def sha(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]

# 1. correctness + bit-exactness across build variants, thread counts and NumPy
exact = []
rng = np.random.default_rng(5)
for d in (6, 100, 384, 768, 1024, 1536, 3072, 4096):
    X = (rng.standard_normal((300, d)) * rng.lognormal(0, 2, (300, 1))).astype(np.float32)
    R = rht_rotation(d, 42)
    ref = Op(d, 42, "O0", threads=1)
    outs = {f"{v}/t{t}": Op(d, 42, v, threads=t) for v in ("native", "sse2", "nativenc") for t in ({1, ncpu} if v != "nativenc" else {1})}
    r1, r0 = ref.rotate_rows(X), ref.unrotate_rows(X)
    same = {k: bool(np.array_equal(o.rotate_rows(X), r1) and np.array_equal(o.unrotate_rows(X), r0)) for k, o in outs.items()}
    p = plan(d, 42)
    same["numpy"] = bool(np.array_equal(numpy_ref(X, p, 1), r1) and np.array_equal(numpy_ref(X, p, 0), r0))
    rel = float(np.abs(r1 - X @ R.T).max() / np.abs(X).max())
    exact.append(dict(d=d, all_equal=all(same.values()), variants=same, rel_err_vs_dense=rel))
    print(d, "all bit-equal:", all(same.values()), f"rel err vs dense {rel:.1e}", flush=True)
res["bitexact"] = exact

# 2. fingerprints: same input on every machine (PCG64 + Ziggurat are platform-independent)
fp = []
for d in (768, 3072):
    X = np.random.default_rng(7).standard_normal((2000, d)).astype(np.float32)
    op = Op(d, 42, "native")
    qd = Quantizer(d, 4, seed=42, rotation="rht")
    qo = Quantizer(d, 4, seed=42, rotation="rht"); qo._rotate_rows = op.rotate_rows
    fp.append(dict(d=d, R=sha(rht_rotation(d, 42)), op_rot=sha(op.rotate_rows(X)),
                   dense_rot=sha(X @ rht_rotation(d, 42).T),
                   op_codes=sha(qo.encode(X).indices), dense_codes=sha(qd.encode(X).indices)))
    print("fingerprint", fp[-1], flush=True)
res["fingerprints"] = fp

# 3. speed, at 1 thread and at all threads, both sides limited identically
def best(fn, rep):
    fn(); ts = []
    for _ in range(rep):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    return min(ts)

speed = []
dims = (768, 3072) if quick else (384, 768, 1024, 1536, 2048, 3072, 4096)
for threads in sorted({1, ncpu}):
    with threadpool_limits(limits=threads):
        for d in dims:
            R = rht_rotation(d, 42); op = Op(d, 42, "native", threads=threads)
            q = np.random.default_rng(1).standard_normal(d).astype(np.float32)
            cells = [("query1d", lambda: R @ q, lambda: op.rotate_query(q), 51)]
            for n in (1, 64, 10000):
                X = np.random.default_rng(n).standard_normal((n, d)).astype(np.float32)
                cells.append((f"enc{n}", lambda X=X: X @ R.T, lambda X=X: op.rotate_rows(X), 5 if n > 64 else 51))
            Xd = np.random.default_rng(9).standard_normal((10000, d)).astype(np.float32)
            cells.append(("dec10000", lambda: Xd @ R, lambda: op.unrotate_rows(Xd), 5))
            # Time all dense cells, then all op cells. Both runtimes spin-wait
            # after a call (OpenBLAS workers, libgomp team), so interleaving
            # them makes each side run against the other's spinning threads.
            td = {name: best(fd, rep) for name, fd, fo, rep in cells}
            time.sleep(0.5)
            to = {name: best(fo, rep) for name, fd, fo, rep in cells}
            time.sleep(0.5)
            for name, *_ in cells:
                a, b = td[name], to[name]
                speed.append(dict(threads=threads, d=d, shape=name, dense_s=a, op_s=b))
                print(f"t={threads} d={d:>5} {name:>9} dense {a*1e3:9.3f} ms  op {b*1e3:9.3f} ms  op/dense {b/a:6.3f}", flush=True)
res["speed"] = speed
os.makedirs("ci", exist_ok=True)
json.dump(res, open(f"ci/{label}.json", "w"), indent=1)
