"""Public-API measurement of one installed remex: timing and code fingerprints.

python bench.py <label> <variant> [--sweep]
Writes out/<label>__<variant>.json. Run once per installed version (main vs
branch) under different PYTHONPATHs; compare.py joins them.
"""
import hashlib, json, os, platform, sys, time
import numpy as np
import remex
from remex import Quantizer

label, variant = sys.argv[1], sys.argv[2]
sweep = "--sweep" in sys.argv
out = dict(label=label, variant=variant, remex_file=remex.__file__, system=platform.system(),
           machine=platform.machine(), numpy=np.__version__, python=sys.version.split()[0])
try:
    from remex import _native
    out["native"] = _native.kernel() is not None
    out["native_reason"] = _native.disable_reason()
except ImportError:
    out["native"] = None
try:
    from threadpoolctl import threadpool_info
    out["blas"] = [{k: i.get(k) for k in ("internal_api", "architecture", "num_threads")} for i in threadpool_info()]
except Exception:
    out["blas"] = []
print(json.dumps({k: v for k, v in out.items()}, indent=1), flush=True)

sha = lambda a: hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]

def best(fn, rep):
    fn(); ts = []
    for _ in range(rep):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    return min(ts)

# fingerprints: identical input everywhere
fp = {}
os.makedirs("out/codes", exist_ok=True)
for d in (384, 768, 3072):
    X = np.random.default_rng(7).standard_normal((2000, d)).astype(np.float32)
    for rotation in ("rht", "haar"):
        if rotation == "haar" and d == 3072:
            continue  # 150 s Householder build; nothing new to learn
        for bits in (2, 4, 8):
            q = Quantizer(d, bits, seed=42, rotation=rotation)
            c = q.encode(X)
            key = f"{rotation}/{d}/{bits}"
            fp[key] = dict(bounds=sha(q.boundaries), cents=sha(q.centroids), codes=sha(c.indices))
            np.save(f"out/codes/{label}__{variant}__{rotation}_{d}_{bits}.npy", c.indices)
    print("fingerprints d", d, flush=True)
out["fingerprints"] = fp

# timing through the public API
speed = []
for d in (384, 768, 1024, 1536, 3072):
    t_build = best(lambda: Quantizer(d, 4, seed=42, rotation="rht"), 3)
    q = Quantizer(d, 4, seed=42, rotation="rht")
    row = dict(d=d, build_s=t_build)
    for n in (1, 16, 64, 256, 1024, 10000):
        X = np.random.default_rng(n).standard_normal((n, d)).astype(np.float32)
        row[f"enc{n}_s"] = best(lambda: q.encode(X), 3 if n >= 1024 else 15)
    Xc = np.random.default_rng(0).standard_normal((20000, d)).astype(np.float32)
    comp = q.encode(Xc)
    qq = np.random.default_rng(1).standard_normal(d).astype(np.float32)
    row["search_adc_s"] = best(lambda: q.search_adc(comp, qq, k=10), 5)
    Qb = np.random.default_rng(2).standard_normal((64, d)).astype(np.float32)
    q.search(comp, qq, k=10)  # build the dequant cache outside the timing
    row["search_s"] = best(lambda: q.search(comp, qq, k=10), 15)
    row["search_batch64_s"] = best(lambda: q.search_batch(comp, Qb, k=10), 5)
    speed.append(row)
    print("speed", {k: (round(v * 1e3, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
out["speed"] = speed

if sweep and out.get("native"):
    from remex import rotation as rot
    sw = []
    for cutoff in (1 << 12, 1 << 14, 1 << 16, 1 << 18):
        rot.PARALLEL_MIN_FLOATS = cutoff
        for d in (384, 768):
            op = rot.RHTOperator(d, 42)
            for n in (8, 16, 32, 64, 128, 256, 512):
                X = np.random.default_rng(n).standard_normal((n, d)).astype(np.float32)
                sw.append(dict(cutoff=cutoff, d=d, n=n, s=best(lambda: op.rotate_rows(X), 25)))
    out["sweep"] = sw
    print("sweep done", flush=True)

os.makedirs("out", exist_ok=True)
json.dump(out, open(f"out/{label}__{variant}.json", "w"), indent=1)
